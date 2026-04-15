import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
from pyhealth.datasets import BaseDataset

from datasets._brat import parse_ann

logger = logging.getLogger(__name__)

# Raw corpus root (contains train/ and test/ BRAT directories)
_DEFAULT_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/RareDis")

# Entity types that represent actual rare-disease mentions.
# SKINRAREDISEASE is a dermatological subtype also kept.
# All other types (DISEASE, SIGN, ANAPHOR, SYMPTOM) are written to
# annotations.csv with their types so downstream tasks can filter.
_DISEASE_TYPES = {"RAREDISEASE", "SKINRAREDISEASE"}

# BRAT split directories relative to root
_SPLIT_DIRS = ["train", "dev", "test"]


def _texts_csv_has_split_column(texts_csv: Path) -> bool:
    """Return True if ``texts.csv`` exists and contains a ``split`` column."""
    if not texts_csv.exists():
        return False
    try:
        columns = pd.read_csv(texts_csv, nrows=0).columns
    except Exception:
        return False
    return "split" in columns


class RareDisDataset(BaseDataset):
    """Dataset class for the RareDis benchmark.

    RareDis is a rare-disease NER corpus drawn from NORD disease
    descriptions.  Raw data is BRAT standoff files (.txt / .ann pairs)
    in ``train/``, ``dev/``, and ``test/`` subdirectories of *root*.

    On first use :meth:`prepare_metadata` is called automatically.  It
    writes three CSV files into *root*:

    * **texts.csv** – one row per document: ``id``, ``split``, ``text``.
    * **annotations.csv** – one row per entity span: ``id``,
      ``ann_id``, ``annotation_type``, ``start``, ``end``,
      ``annotation``.
    * **relations.csv** – one row per relation: ``id``, ``rel_id``,
      ``relation_type``, ``arg1``, ``arg2``.

    Documents that contain zero entities whose ``annotation_type`` is in
    ``_DISEASE_TYPES`` are excluded entirely; all other entity types in
    those documents are still written to annotations.csv for flexibility.

    Three PyHealth tables are loaded:

    * **texts** – full passage text per document.
    * **annotations** – entity spans with character offsets and type.
    * **relations** – binary relations between entity spans.

    Args:
        root (str): Directory with ``train/``, ``dev/``, ``test/`` BRAT dirs.
            CSVs are also written here.  Defaults to the RareDis checkout.
        tables (List[str]): Tables to load.  Defaults to all three.
        dataset_name (Optional[str]): Override the dataset name string.
        config_path (Optional[str]): Path to an alternative YAML config.
        **kwargs: Passed through to
            :class:`~pyhealth.datasets.BaseDataset`.

    Examples:
        >>> from datasets.raredis import RareDisDataset
        >>> dataset = RareDisDataset()
        >>> dataset.stats()
    """

    def __init__(
        self,
        root: str = str(_DEFAULT_ROOT),
        tables: List[str] = None,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        if config_path is None:
            config_path = Path(__file__).parent / "configs" / "raredis.yaml"

        if tables is None:
            tables = ["texts", "annotations", "relations"]

        root_path = Path(root)
        has_all_csvs = all(
            (root_path / f).exists()
            for f in ("texts.csv", "annotations.csv", "relations.csv")
        )
        texts_csv = root_path / "texts.csv"

        if not has_all_csvs:
            logger.info(
                "RareDis CSVs not found in %s — running prepare_metadata",
                root,
            )
            self.prepare_metadata(root)
        elif not _texts_csv_has_split_column(texts_csv):
            logger.info(
                "RareDis texts.csv in %s is missing 'split' — regenerating metadata",
                root,
            )
            self.prepare_metadata(root)

        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "raredis",
            config_path=str(config_path),
            **kwargs,
        )

    def prepare_metadata(self, root: str) -> None:
        """Parse raw BRAT files and write texts/annotations/relations CSVs.

        Reads all .txt / .ann pairs from the ``train/``, ``dev/``, and ``test/``
        subdirectories of *root*.  All entity types are written with their
        BRAT identifiers and character offsets.  Documents are only
        included if they contain at least one annotation whose type is in
        ``_DISEASE_TYPES``.

        Args:
            root (str): Directory containing the BRAT split directories.
        """
        root_path = Path(root)
        texts_rows: list = []
        ann_rows: list = []
        rel_rows: list = []

        for split in _SPLIT_DIRS:
            split_dir = root_path / split
            if not split_dir.exists():
                logger.warning("Split directory not found: %s", split_dir)
                continue

            for txt_path in sorted(split_dir.glob("*.txt")):
                ann_path = txt_path.with_suffix(".ann")
                if not ann_path.exists():
                    continue

                doc_id = txt_path.stem
                text = txt_path.read_text(encoding="utf-8", errors="replace")

                entities, relations = parse_ann(ann_path)

                # Only include documents with at least one disease mention
                if not any(e["annotation_type"] in _DISEASE_TYPES for e in entities):
                    continue

                texts_rows.append({"id": doc_id, "split": split, "text": text})
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

        disease_count = sum(
            1 for a in ann_rows if a["annotation_type"] in _DISEASE_TYPES
        )
        logger.info(
            "RareDis prepare_metadata: %d docs, %d annotations "
            "(%d disease-type), %d relations → %s",
            len(texts_df),
            len(ann_df),
            disease_count,
            len(rel_df),
            root_path,
        )
