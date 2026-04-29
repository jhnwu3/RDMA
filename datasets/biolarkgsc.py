import logging
import random
from pathlib import Path
from typing import List, Optional

import pandas as pd
from pyhealth.datasets import BaseDataset

logger = logging.getLogger(__name__)

# Raw corpus root (contains biolarkgsc_locs.csv)
_DEFAULT_ROOT = Path(
    "/home/johnwu3/projects/rare_disease/workspace/repos/BioLarkGSC"
)

# Source TSV file with character-offset annotations
_LOCS_FILE = "biolarkgsc_locs.csv"


def _parse_labels(labels_str: str, text: str) -> list:
    """Parse the semicolon-separated label string from biolarkgsc_locs.csv.

    Each entry has the form ``HP_XXXXXXX|start:end``.  The surface text is
    extracted by slicing *text*[start:end].  Malformed entries are skipped
    with a warning.

    Args:
        labels_str: Raw labels string from the CSV ``labels`` column.
        text: Full document text used to extract surface strings.

    Returns:
        List of dicts with keys ``ann_id``, ``annotation_type``, ``hpo_id``,
        ``start``, ``end``, ``annotation``.
    """
    entries = []
    for idx, token in enumerate(labels_str.split(";"), start=1):
        token = token.strip()
        if not token:
            continue
        if "|" not in token:
            logger.warning("Skipping malformed label token: %r", token)
            continue
        hpo_id, span = token.split("|", 1)
        if ":" not in span:
            logger.warning("Skipping malformed span in token: %r", token)
            continue
        try:
            start, end = (int(x) for x in span.split(":", 1))
        except ValueError:
            logger.warning("Non-integer offsets in token: %r", token)
            continue
        surface = text[start:end]
        entries.append({
            "ann_id": f"T{idx}",
            "annotation_type": "HP",
            "hpo_id": hpo_id,
            "start": start,
            "end": end,
            "annotation": surface,
        })
    return entries


class BioLarkGSCDataset(BaseDataset):
    """Dataset class for the BioLark GSC benchmark.

    BioLark GSC is a phenotype NER corpus of 228 biomedical abstracts
    annotated with Human Phenotype Ontology (HPO) terms and character
    offsets.  Raw data is a single tab-separated file
    ``biolarkgsc_locs.csv`` containing document IDs, full text, and
    semicolon-separated ``HP_ID|start:end`` labels.

    On first use :meth:`prepare_metadata` is called automatically.  It
    writes two CSV files into *root*:

    * **texts.csv** – one row per document: ``id``, ``text``.
    * **annotations.csv** – one row per HPO mention: ``id``, ``ann_id``,
      ``annotation_type``, ``hpo_id``, ``start``, ``end``, ``annotation``.

    There are no predefined train/dev/test splits in the BioLark GSC
    distribution; all 228 documents are included without a split column.

    Two PyHealth tables are loaded:

    * **texts** – full abstract text per document.
    * **annotations** – HPO entity spans with character offsets.

    Args:
        root (str): Directory containing ``biolarkgsc_locs.csv``.
            CSVs are also written here.  Defaults to the BioLarkGSC checkout.
        tables (List[str]): Tables to load.  Defaults to both.
        dataset_name (Optional[str]): Override the dataset name string.
        config_path (Optional[str]): Path to an alternative YAML config.
        **kwargs: Passed through to
            :class:`~pyhealth.datasets.BaseDataset`.

    Examples:
        >>> from datasets.biolarkgsc import BioLarkGSCDataset
        >>> dataset = BioLarkGSCDataset()
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
            config_path = (
                Path(__file__).parent / "configs" / "biolarkgsc.yaml"
            )

        if tables is None:
            tables = ["texts", "annotations"]

        root_path = Path(root)
        required_csvs = ("texts.csv", "annotations.csv")
        texts_path = root_path / "texts.csv"
        needs_regen = not all((root_path / f).exists() for f in required_csvs)
        if not needs_regen and texts_path.exists():
            _df = pd.read_csv(texts_path, nrows=1)
            if "split" not in _df.columns:
                needs_regen = True
        if needs_regen:
            logger.info(
                "BioLarkGSC CSVs not found in %s — running prepare_metadata",
                root,
            )
            self.prepare_metadata(root)
        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "biolarkgsc",
            config_path=str(config_path),
            **kwargs,
        )

    def prepare_metadata(self, root: str) -> None:
        """Parse the BioLark GSC TSV and write texts/annotations CSVs.

        Reads ``biolarkgsc_locs.csv`` from *root*, parses HPO mentions
        with character offsets, and extracts surface strings from the
        document text.

        Args:
            root (str): Directory containing ``biolarkgsc_locs.csv``.
        """
        root_path = Path(root)
        src = root_path / _LOCS_FILE

        df = pd.read_csv(src, sep="\t", dtype=str)

        texts_rows: list = []
        ann_rows: list = []

        for _, row in df.iterrows():
            doc_id = str(row["id"])
            text = str(row["text"])
            labels_str = str(row.get("labels", ""))

            texts_rows.append({"id": doc_id, "text": text})

            if labels_str and labels_str.lower() != "nan":
                for entry in _parse_labels(labels_str, text):
                    ann_rows.append({"id": doc_id, **entry})

        # Deterministic 80/10/10 split (no predefined splits in BioLarkGSC)
        doc_ids = [r["id"] for r in texts_rows]
        rng = random.Random(42)
        shuffled = doc_ids[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * 0.8)
        n_dev = int(n * 0.1)
        split_map = {}
        for doc_id in shuffled[:n_train]:
            split_map[doc_id] = "train"
        for doc_id in shuffled[n_train : n_train + n_dev]:
            split_map[doc_id] = "dev"
        for doc_id in shuffled[n_train + n_dev :]:
            split_map[doc_id] = "test"
        for r in texts_rows:
            r["split"] = split_map[r["id"]]

        texts_df = pd.DataFrame(texts_rows)
        ann_df = pd.DataFrame(ann_rows)

        texts_df.to_csv(root_path / "texts.csv", index=False)
        ann_df.to_csv(root_path / "annotations.csv", index=False)

        logger.info(
            "BioLarkGSC prepare_metadata: %d docs, %d HP annotations → %s",
            len(texts_df),
            len(ann_df),
            root_path,
        )
