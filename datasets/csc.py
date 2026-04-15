import json
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
from pyhealth.datasets import BaseDataset

logger = logging.getLogger(__name__)

# Raw corpus root (contains phenotype_mining_benchmark.json)
_DEFAULT_ROOT = Path(
    "/home/johnwu3/projects/rare_disease/workspace/repos/CSC"
)

# Source JSON file name
_JSON_FILE = "phenotype_mining_benchmark.json"


class CSCDataset(BaseDataset):
    """Dataset class for the CSC phenotype mining benchmark.

    The CSC benchmark consists of 116 clinical case reports annotated
    with Human Phenotype Ontology (HPO) phenotype terms.  Raw data is a
    single JSON file ``phenotype_mining_benchmark.json`` where each
    top-level key is a document ID and each value contains a
    ``clinical_text`` field and a ``phenotypes`` list of
    ``{phenotype_name, hpo_id}`` dicts.

    Unlike BioLark GSC and RareDis, the CSC annotations do **not**
    include character offsets into the text.

    On first use :meth:`prepare_metadata` is called automatically.  It
    writes two CSV files into *root*:

    * **texts.csv** – one row per document: ``id``, ``text``.
    * **phenotypes.csv** – one row per annotation: ``id``,
      ``phenotype_name``, ``hpo_id``.

    HPO IDs are stored in their original colon-separated form
    (``HP:XXXXXXX``).  Documents with zero phenotypes are included so
    they can serve as negative examples for downstream tasks.

    Two PyHealth tables are loaded:

    * **texts** – full clinical case text per document.
    * **phenotypes** – phenotype annotations with HPO IDs and names.

    Args:
        root (str): Directory containing ``phenotype_mining_benchmark.json``.
            CSVs are also written here.  Defaults to the CSC checkout.
        tables (List[str]): Tables to load.  Defaults to both.
        dataset_name (Optional[str]): Override the dataset name string.
        config_path (Optional[str]): Path to an alternative YAML config.
        **kwargs: Passed through to
            :class:`~pyhealth.datasets.BaseDataset`.

    Examples:
        >>> from datasets.csc import CSCDataset
        >>> dataset = CSCDataset()
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
            config_path = Path(__file__).parent / "configs" / "csc.yaml"

        if tables is None:
            tables = ["texts", "phenotypes"]

        root_path = Path(root)
        required_csvs = ("texts.csv", "phenotypes.csv")
        if not all((root_path / f).exists() for f in required_csvs):
            logger.info(
                "CSC CSVs not found in %s — running prepare_metadata",
                root,
            )
            self.prepare_metadata(root)

        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "csc",
            config_path=str(config_path),
            **kwargs,
        )

    def prepare_metadata(self, root: str) -> None:
        """Parse the CSC JSON and write texts/phenotypes CSVs.

        Reads ``phenotype_mining_benchmark.json`` from *root* and
        flattens the nested phenotype lists into a long-form CSV.
        All documents are written, including those with no phenotypes.

        Args:
            root (str): Directory containing ``phenotype_mining_benchmark.json``.
        """
        root_path = Path(root)
        src = root_path / _JSON_FILE

        with open(src, encoding="utf-8") as fh:
            data = json.load(fh)

        texts_rows: list = []
        phenotype_rows: list = []

        for doc_id, doc in data.items():
            text = doc.get("clinical_text", "")
            texts_rows.append({"id": doc_id, "text": text})

            for ph in doc.get("phenotypes", []):
                phenotype_rows.append({
                    "id": doc_id,
                    "phenotype_name": ph.get("phenotype_name", ""),
                    "hpo_id": ph.get("hpo_id", ""),
                })

        texts_df = pd.DataFrame(texts_rows)
        phenotypes_df = pd.DataFrame(phenotype_rows)

        texts_df.to_csv(root_path / "texts.csv", index=False)
        phenotypes_df.to_csv(root_path / "phenotypes.csv", index=False)

        logger.info(
            "CSC prepare_metadata: %d docs, %d phenotype annotations → %s",
            len(texts_df),
            len(phenotypes_df),
            root_path,
        )
