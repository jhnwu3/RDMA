import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
from pyhealth.datasets import BaseDataset

from datasets._brat import parse_ann

logger = logging.getLogger(__name__)

# Raw corpus root (Corpus/Annotated texts/ and Corpus/Original texts/)
_DEFAULT_ROOT = Path(
    "/home/johnwu3/projects/rare_disease/workspace/repos/RDD_Corpus"
)

# "Disability" covers full disease/condition expressions in the RDD scheme.
# Sub-component types (Impairment, Function) and meta-types
# (Speculation, SpecScope, ABR, Neg, Scope) are also written to
# annotations.csv with their types for downstream use.
_DISEASE_TYPES = {"Disability"}

# Subdirectory layout inside the corpus root
_ANN_SUBDIR = Path("Corpus") / "Annotated texts" / "ANN"
_TXT_SUBDIR = Path("Corpus") / "Original texts"
_REL_SUBDIR = Path("Relationships")

# Pipe-delimited column names for positive.csv / negative.csv
_REL_COLS = [
    "pubmed_id", "sentence",
    "rare_disease", "rd_start", "rd_end",
    "disability", "dis_start", "dis_end",
    "speculative", "orphanet_id",
]


class RDDDataset(BaseDataset):
    """Dataset class for the RDD (Rare Disease Dataset) benchmark.

    RDD is a rare-disease NER corpus of biomedical literature abstracts
    annotated with disability and function expressions.  Raw data is BRAT
    standoff files (.txt / .ann pairs) under
    ``Corpus/Annotated texts/ANN/`` and ``Corpus/Original texts/``.

    On first use :meth:`prepare_metadata` is called automatically.  It
    writes four CSV files into *root*:

    * **texts.csv** – one row per document: ``id``, ``text``.
    * **annotations.csv** – one row per entity span: ``id``,
      ``ann_id``, ``annotation_type``, ``start``, ``end``,
      ``annotation``.
    * **relations.csv** – one row per relation: ``id``, ``rel_id``,
      ``relation_type``, ``arg1``, ``arg2``.
    * **relationships.csv** – one row per rare-disease/disability pair
      from ``Relationships/positive.csv`` and ``negative.csv``,
      linked to the document via ``id`` (matched by PubMed ID).

    All entity types are written so downstream tasks can apply their own
    filters.  The default NER task (``RDDNER``) retains only
    ``Disability`` spans.

    Four PyHealth tables are loaded:

    * **texts** – full abstract text per document.
    * **annotations** – entity spans with character offsets and type.
    * **relations** – binary relations between entity spans.
    * **relationships** – sentence-level rare-disease/disability pairs
      with polarity (positive / negative) and speculation flag.

    Args:
        root (str): Directory containing the ``Corpus/`` subdirectory.
            CSVs are also written here.  Defaults to the RDD checkout.
        tables (List[str]): Tables to load.  Defaults to all four.
        dataset_name (Optional[str]): Override the dataset name string.
        config_path (Optional[str]): Path to an alternative YAML config.
        **kwargs: Passed through to
            :class:`~pyhealth.datasets.BaseDataset`.

    Examples:
        >>> from datasets.rdd import RDDDataset
        >>> dataset = RDDDataset()
        >>> dataset.stats()
    """

    def __init__(
        self,
        root: str = str(_DEFAULT_ROOT),
        tables: List[str] = None,
        dataset_name: Optional[str] = "RDD",
        config_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        if config_path is None:
            config_path = (
                Path(__file__).parent / "configs" / "rdd.yaml"
            )

        if tables is None:
            tables = [
                "texts", "annotations", "relations", "relationships",
            ]

        root_path = Path(root)
        required_csvs = (
            "texts.csv",
            "annotations.csv",
            "relations.csv",
            "relationships.csv",
        )
        if not all((root_path / f).exists() for f in required_csvs):
            logger.info(
                "RDD CSVs not found in %s — running prepare_metadata",
                root,
            )
            self.prepare_metadata(root)

        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "rdd",
            config_path=str(config_path),
            **kwargs,
        )

    @staticmethod
    def _build_pubmed_index(txt_dir: Path) -> dict:
        """Return a mapping of pubmed_id -> doc_id from text filenames.

        Text files are named ``{orphanet_id}-{idx}-{pubmed_id}.txt``.
        The PubMed ID is the last dash-separated segment of the stem.
        """
        index: dict = {}
        for p in txt_dir.glob("*.txt"):
            parts = p.stem.rsplit("-", 1)
            if len(parts) == 2:
                pubmed_id = parts[1]
                index.setdefault(pubmed_id, p.stem)
        return index

    @staticmethod
    def _parse_rel_file(
        path: Path, polarity: str, pubmed_index: dict
    ) -> list:
        """Parse one pipe-delimited relationship file.

        Each row has 10 pipe-separated fields (see ``_REL_COLS``).
        The ``id`` column is resolved via *pubmed_index*; rows whose
        PubMed ID is absent from the index are still included with
        ``id`` set to ``None``.

        Args:
            path: Path to positive.csv or negative.csv.
            polarity: ``"positive"`` or ``"negative"``.
            pubmed_index: Mapping of pubmed_id -> doc_id.

        Returns:
            List of row dicts ready for a DataFrame.
        """
        rows = []
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line:
                    continue
                fields = line.split("|")
                if len(fields) < len(_REL_COLS):
                    logger.warning(
                        "Skipping malformed line in %s: %r",
                        path.name,
                        line[:80],
                    )
                    continue
                row = dict(zip(_REL_COLS, fields))
                pubmed_id = row["pubmed_id"]
                row["id"] = pubmed_index.get(pubmed_id)
                if row["id"] is None:
                    logger.debug(
                        "PubMed ID %s not found in text index",
                        pubmed_id,
                    )
                row["polarity"] = polarity
                rows.append(row)
        return rows

    def prepare_metadata(self, root: str) -> None:
        """Parse raw BRAT files and write all four CSVs.

        Reads every .ann file from ``Corpus/Annotated texts/ANN/``
        and pairs it with the matching .txt from
        ``Corpus/Original texts/``.  Also parses
        ``Relationships/positive.csv`` and ``negative.csv``, linking
        each row to its document via PubMed ID.

        Args:
            root (str): Corpus root directory.
        """
        root_path = Path(root)
        ann_dir = root_path / _ANN_SUBDIR
        txt_dir = root_path / _TXT_SUBDIR
        rel_dir = root_path / _REL_SUBDIR

        texts_rows: list = []
        ann_rows: list = []
        rel_rows: list = []

        for ann_path in sorted(ann_dir.glob("*.ann")):
            txt_path = txt_dir / (ann_path.stem + ".txt")
            if not txt_path.exists():
                logger.warning(
                    "No matching .txt for %s — skipping",
                    ann_path.name,
                )
                continue

            doc_id = ann_path.stem
            text = txt_path.read_text(encoding="utf-8", errors="replace")

            entities, relations = parse_ann(ann_path, encoding="utf-8")
            if not entities:
                continue

            texts_rows.append({"id": doc_id, "text": text})
            for e in entities:
                ann_rows.append({"id": doc_id, **e})
            for r in relations:
                rel_rows.append({"id": doc_id, **r})

        texts_df = pd.DataFrame(texts_rows)
        ann_df = pd.DataFrame(ann_rows)
        rel_df = pd.DataFrame(rel_rows)

        texts_df.to_csv(root_path / "texts.csv", index=False)
        ann_df.to_csv(root_path / "annotations.csv", index=False)
        rel_df.to_csv(root_path / "relations.csv", index=False)

        # ── Relationships (positive + negative) ───────────────────────
        pubmed_index = self._build_pubmed_index(txt_dir)

        rel_file_pairs = [
            (rel_dir / "positive.csv", "positive"),
            (rel_dir / "negative.csv", "negative"),
        ]
        ship_rows: list = []
        for rel_path, polarity in rel_file_pairs:
            if rel_path.exists():
                ship_rows.extend(
                    self._parse_rel_file(rel_path, polarity, pubmed_index)
                )
            else:
                logger.warning(
                    "Relationship file not found: %s", rel_path
                )

        ship_df = pd.DataFrame(ship_rows)
        ship_df.to_csv(root_path / "relationships.csv", index=False)

        disease_count = sum(
            1 for a in ann_rows if a["annotation_type"] in _DISEASE_TYPES
        )
        matched = ship_df["id"].notna().sum() if not ship_df.empty else 0
        logger.info(
            "RDD prepare_metadata: %d docs, %d annotations "
            "(%d Disability-type), %d relations, "
            "%d relationships (%d linked to a doc) → %s",
            len(texts_df),
            len(ann_df),
            disease_count,
            len(rel_df),
            len(ship_df),
            matched,
            root_path,
        )
