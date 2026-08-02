# RDMA benchmark data

This directory holds the benchmark data RDMA is evaluated on. Three corpora are
redistributed here under their original licenses; the rest is either produced by
us or must be obtained separately.

**Every dataset below belongs to its original authors.** If you use one, cite the
original paper, not just RDMA. Full entries are in [`../CITATIONS.md`](../CITATIONS.md).

## What ships here

| Directory / file | Dataset | Docs | License | Redistributed? |
|---|---|---|---|---|
| `biolarkgsc/` | BioLark GSC+ | 228 | CC BY 4.0 | yes |
| `csc/` | CSC phenotype-mining benchmark | 116 | MIT | yes |
| `raredis/` | RareDis | 1011 | CC BY 4.0 | yes |
| `rare_disease_mining/` | our MIMIC-III rare-disease annotations | 117–312 | see below | annotations only, **no note text** |
| `initial_diff_diagnosis_benchmark.json` | our MIMIC-IV differential-diagnosis benchmark | 145 | see below | annotations only, **no note text** |
| `annotation_tool_input.json` | input for `../annotation_tool.html` | 117 | see below | annotations only, **no note text** |
| — | RDD Corpus | — | see upstream | **no** — download separately |

Loaders default to these paths, so the first three benchmarks work on a fresh
clone with no configuration:

```python
from datasets.biolarkgsc import BioLarkGSCDataset
from datasets.csc import CSCDataset
from datasets.raredis import RareDisDataset

BioLarkGSCDataset()   # 228 documents
CSCDataset()          # 116 documents
RareDisDataset()      # 1011 documents
```

On first use each loader runs `prepare_metadata()` and writes derived
`texts.csv` / `annotations.csv` / `phenotypes.csv` / `relations.csv` into its own
directory. Those are gitignored — only the source files are committed.

---

## BioLark GSC+ — `biolarkgsc/`

228 PubMed abstracts annotated with Human Phenotype Ontology terms and character
offsets. Originally annotated by Groza et al. (2015); GSC+ is the corrected and
extended version by Lobo, Lamurias & Couto (2017). The packaging we use is the
Mendeley release by Shankai Yan, distributed with PhenoRerank.

- Source: <https://data.mendeley.com/datasets/v4t59p8w4z/3> (DOI `10.17632/v4t59p8w4z.3`)
- License: **CC BY 4.0** — redistribution permitted with attribution
- Files: `biolarkgsc_locs.csv` (tab-separated; `id`, `text`, and `HP_XXXXXXX|start:end` labels), `biolarkgsc.csv`
- Cite: Lobo et al. 2017, and Groza et al. 2015 for the original GSC

## CSC — `csc/`

116 published clinical case reports annotated with HPO phenotype terms. Derived
from the evaluation set distributed with RAG-HPO (Posey Lab, Baylor College of
Medicine). We converted the upstream spreadsheet to JSON; annotations are
unchanged. Note that CSC annotations carry **no character offsets**.

- Source: <https://github.com/PoseyPod/RAG-HPO>
- License: **MIT** — redistribution permitted with attribution
- File: `phenotype_mining_benchmark.json` — `{doc_id: {clinical_text, phenotypes: [{phenotype_name, hpo_id}]}}`
- Cite: Garcia et al. 2025, *Genome Medicine*

## RareDis — `raredis/`

Rare-disease NER and relation corpus built from National Organization for Rare
Disorders (NORD) disease descriptions, in BRAT standoff format. Entity types:
`RAREDISEASE`, `SKINRAREDISEASE`, `DISEASE`, `SIGN`, `SYMPTOM`, `ANAPHOR`.

- Source: <https://github.com/isegura/NLP4RARE-CM-UC3M>
- License: **CC BY 4.0**, © 2024 Isabel Segura-Bedmar — redistribution permitted with attribution
- Files: `train/`, `dev/`, `test/`, each holding `.txt` / `.ann` pairs
- Cite: Martínez-deMiguel et al. 2022, *Journal of Biomedical Informatics*

## RDD Corpus — not included

The RDD corpus is **not** redistributed here. Download it from its upstream
source and point the loader at it:

```python
from datasets.rdd import RDDDataset
RDDDataset(root="/path/to/RDD_Corpus")
```

The loader expects `Corpus/Annotated texts/ANN`, `Corpus/Original texts/`, and
`Relationships/`. See the corpus's own `README.txt` for authorship and license.

---

## MIMIC-derived files — annotations only

`rare_disease_mining/`, `initial_diff_diagnosis_benchmark.json` and
`annotation_tool_input.json` are **our** annotations over MIMIC clinical notes.

MIMIC-III and MIMIC-IV are distributed by PhysioNet under a Credentialed Health
Data Use Agreement that forbids republishing note text. So these files contain
**no clinical note text**. Every `context` field has been removed by
[`../scripts/data/strip_mimic_text.py`](../scripts/data/strip_mimic_text.py);
what remains is our annotation labels plus the identifiers needed to join back
to the source notes.

| File | Corpus | Contents |
|---|---|---|
| `rare_disease_mining/mimic3_mining_rdma_human_annotations.json` | MIMIC-III | 117 notes; human-reviewed gold labels. Keyed by `NOTEEVENTS.ROW_ID`; each entry carries `note_details = {subject_id, hadm_id, category, chartdate, charttime}` |
| `rare_disease_mining/rd_annos_public.json` | MIMIC-III | 312 notes; original annotations from acadTags/Rare-disease-identification |
| `rare_disease_mining/filtered_rd_annos_public.json` | MIMIC-III | 117 notes; keyword-filtered subset |
| `rare_disease_mining/reannoted_rd_annos.json` | MIMIC-III | 333 human re-annotation decisions, joined by `document_id` |
| `annotation_tool_input.json` | MIMIC-III | predictions vs. gold for the annotation UI, joined by `document_id` |
| `initial_diff_diagnosis_benchmark.json` | **MIMIC-IV** | 145 subjects; ORPHA codes, disease entities and matched phenotypes. Keyed by `subject_id` |

Note the last one is MIMIC-**IV**, not MIMIC-III — it uses `___`
de-identification and 8-digit `subject_id` keys. It also holds `lab_info`
entries; the numeric patient values were removed, keeping `lab_name`, `units`
and `direction`.

Entity labels are kept verbatim even when they contain a `___` placeholder
(e.g. `"K___-Feil syndrome"`), since the placeholder is exactly where MIMIC's own
de-identification removed PHI, and these labels *are* the benchmark.

### Rebuilding the context windows

If you hold PhysioNet credentials, you can reconstruct the removed contexts from
your own copy:

```bash
python scripts/data/rehydrate_mimic_text.py \
    --mimic3-root      /path/to/mimic-iii-clinical-database-1.4 \
    --mimic4-note-root /path/to/mimic-iv-note/2.2/note \
    --out rehydrated/
```

This re-locates each annotated entity in its source note and re-extracts a
window around it. Because the originals stored snippets without character
offsets, the result is **equivalent but not byte-identical** to ours — the
enclosing sentences match, the window edges may not. The script reports any
entity it could not relocate.

Get the data at <https://physionet.org/content/mimiciii/> and
<https://physionet.org/content/mimic-iv-note/>. To evaluate on MIMIC-III you also
need the notes themselves; pass `--mimic3_root` to the MIMIC tasks.

### Verifying before you publish

```bash
python scripts/data/check_public_data_leakage.py
```

Walks every JSON here and exits non-zero if it finds a residual `context` key, an
over-long string, or a MIMIC de-identification marker. Run it after regenerating
anything in this directory.

---

## Ontologies

RDMA matches against the Human Phenotype Ontology and Orphanet. Neither is
redistributed here; the prebuilt embedding stores are a separate download (see
the top-level `readme.md`).

- **HPO** — <https://hpo.jax.org>, CC BY 4.0
- **Orphanet / ORDO** — <https://www.orphadata.com>, CC BY 4.0
- **UMLS** — used for some concept mappings. **License-restricted**: you must
  obtain your own UMLS Metathesaurus License from the NLM at
  <https://uts.nlm.nih.gov/uts/signup-login>.
