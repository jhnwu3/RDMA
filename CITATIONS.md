# Credits and citations

RDMA builds on datasets, baseline systems, models and ontologies created by
other groups. This file records every one of them. **If you use RDMA, please
cite the underlying work you actually relied on**, not only our paper.

Dataset provenance and licenses are detailed in
[`public_data/README.md`](public_data/README.md).

---

## RDMA

```bibtex
@article{rdma2025,
  title  = {RDMA: Rare Disease Mining Agents},
  note   = {arXiv:2507.15867},
  url    = {https://arxiv.org/abs/2507.15867},
  year   = {2025}
}
```

---

## Datasets

### BioLark GSC / GSC+
Original gold standard corpus of 228 PubMed abstracts annotated with HPO terms,
later corrected and extended as GSC+.

```bibtex
@article{lobo2017identifying,
  title   = {Identifying Human Phenotype Terms by Combining Machine Learning and Validation Rules},
  author  = {Lobo, Manuel and Lamurias, Andre and Couto, Francisco M},
  journal = {BioMed Research International},
  volume  = {2017},
  pages   = {8565739},
  year    = {2017},
  doi     = {10.1155/2017/8565739}
}
```

Cite Groza et al. (2015) for the original GSC annotations. The packaged release
used here is distributed on Mendeley Data (DOI `10.17632/v4t59p8w4z.3`, CC BY
4.0) alongside PhenoRerank:

```bibtex
@article{yan2022phenorerank,
  title   = {PhenoRerank: A re-ranking model for phenotypic concept recognition
             pre-trained on human phenotype ontology},
  author  = {Yan, Shankai and Lu, Zhiyong},
  journal = {Journal of Biomedical Informatics},
  year    = {2022}
}
```

### RareDis
Rare-disease NER and relation corpus built from NORD disease descriptions.
CC BY 4.0, © 2024 Isabel Segura-Bedmar.

```bibtex
@article{martinezdemiguel2022raredis,
  title   = {The RareDis corpus: A corpus annotated with rare diseases, their
             signs and symptoms},
  author  = {Mart{\'i}nez-deMiguel, Claudia and Segura-Bedmar, Isabel and
             Chac{\'o}n-Solano, Esteban and Guerrero-Aspizua, Sara},
  journal = {Journal of Biomedical Informatics},
  volume  = {125},
  pages   = {103961},
  year    = {2022},
  doi     = {10.1016/j.jbi.2021.103961}
}
```

### CSC phenotype-mining benchmark
Derived from the evaluation set distributed with RAG-HPO (MIT). See the RAG-HPO
entry under *Baselines*.

### RDD Corpus
Not redistributed with RDMA. See its own `README.txt` for authorship and
license.

### MIMIC-III and MIMIC-IV
Our rare-disease annotations are over MIMIC clinical notes. The notes themselves
are not redistributed; see `public_data/README.md`.

```bibtex
@article{johnson2016mimiciii,
  title   = {MIMIC-III, a freely accessible critical care database},
  author  = {Johnson, Alistair E W and Pollard, Tom J and Shen, Lu and
             Lehman, Li-wei H and Feng, Mengling and Ghassemi, Mohammad and
             Moody, Benjamin and Szolovits, Peter and Celi, Leo Anthony and
             Mark, Roger G},
  journal = {Scientific Data},
  volume  = {3},
  pages   = {160035},
  year    = {2016},
  doi     = {10.1038/sdata.2016.35}
}
```

Also cite MIMIC-IV and PhysioNet per the instructions on their PhysioNet
project pages.

The MIMIC-III rare-disease annotations we build on originate from
[acadTags/Rare-disease-identification](https://github.com/acadTags/Rare-disease-identification).

---

## Baselines

Each entry lists the paper, the upstream code, and where our wrapper lives.

### PhenoGPT
LLM-based phenotype concept recognition. Wrappers:
`baselines/{biolarkgsc,csc}/phenogpt.py`, `rdma/hporag/phenogpt.py`.
Upstream: <https://github.com/WGLab/PhenoGPT>

```bibtex
@article{yang2024phenogpt,
  title   = {Enhancing phenotype recognition in clinical notes using large
             language models: PhenoBCBERT and PhenoGPT},
  author  = {Yang, Jingye and Liu, Cong and Deng, Wendy and Wu, Da and
             Weng, Chunhua and Zhou, Yunyun and Wang, Kai},
  journal = {Patterns},
  volume  = {5},
  number  = {1},
  pages   = {100887},
  year    = {2024},
  doi     = {10.1016/j.patter.2023.100887}
}
```

### PhenoGPT2
Multimodal fine-tuned LLM for phenotype extraction and normalization.
Wrappers: `baselines/{biolarkgsc,csc}/phenogpt2.py`.

```bibtex
@inproceedings{phenogpt2,
  title     = {PhenoGPT2: A Multimodal Fine-tuned Large Language Model for
               Phenotype Extraction and Normalization from Clinical Text and
               Facial Images},
  booktitle = {Proceedings of the 16th ACM International Conference on
               Bioinformatics, Computational Biology, and Health Informatics
               (ACM-BCB)},
  year      = {2025},
  doi       = {10.1145/3765612.3767763}
}
```

### RAG-HPO
Retrieval-augmented HPO term assignment. Also the source of the CSC benchmark.
Wrappers: `baselines/{biolarkgsc,csc}/raghpo.py`.
Upstream: <https://github.com/PoseyPod/RAG-HPO> (MIT)

```bibtex
@article{garcia2025raghpo,
  title   = {Improving automated deep phenotyping through large language models
             using retrieval-augmented generation},
  author  = {Garcia, Branden T and Westerfield, Lauren and Yelemali, Priya and
             others},
  journal = {Genome Medicine},
  volume  = {17},
  number  = {1},
  pages   = {91},
  year    = {2025},
  doi     = {10.1186/s13073-025-01521-w}
}
```

### FastHPOCR
Dictionary-based HPO concept recognition. Wrappers:
`baselines/{biolarkgsc,csc}/fasthpocr.py`.

```bibtex
@article{groza2024fasthpocr,
  title   = {FastHPOCR: pragmatic, fast, and accurate concept recognition using
             the human phenotype ontology},
  author  = {Groza, Tudor and Gration, Dylan and Baynam, Gareth and
             Robinson, Peter N},
  journal = {Bioinformatics},
  volume  = {40},
  number  = {7},
  pages   = {btae406},
  year    = {2024},
  doi     = {10.1093/bioinformatics/btae406}
}
```

### BioBERT-MRC
Span NER as machine reading comprehension, over BioBERT. Wrappers:
`baselines/*/biobert_mrc.py`, trainer `baselines/raredis/biobert_mrc_trainer.py`,
model `models/biobert_span_ner.py`. Cite both:

```bibtex
@inproceedings{li2020unified,
  title     = {A Unified MRC Framework for Named Entity Recognition},
  author    = {Li, Xiaoya and Feng, Jingrong and Meng, Yuxian and Han, Qinghong
               and Wu, Fei and Li, Jiwei},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association for
               Computational Linguistics (ACL)},
  pages     = {5849--5859},
  year      = {2020},
  url       = {https://aclanthology.org/2020.acl-main.519/}
}

@article{lee2020biobert,
  title   = {BioBERT: a pre-trained biomedical language representation model for
             biomedical text mining},
  author  = {Lee, Jinhyuk and Yoon, Wonjin and Kim, Sungdong and Kim, Donghyeon
             and Kim, Sunkyu and So, Chan Ho and Kang, Jaewoo},
  journal = {Bioinformatics},
  volume  = {36},
  number  = {4},
  pages   = {1234--1240},
  year    = {2020},
  doi     = {10.1093/bioinformatics/btz682}
}
```

### Bio_ClinicalBERT token NER
Wrappers: `baselines/*/bioclinicalbert_ner.py`, model
`models/bioclinicalbert_ner.py`. Checkpoint:
[`emilyalsentzer/Bio_ClinicalBERT`](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT).

```bibtex
@inproceedings{alsentzer2019publicly,
  title     = {Publicly Available Clinical BERT Embeddings},
  author    = {Alsentzer, Emily and Murphy, John and Boag, William and
               Weng, Wei-Hung and Jin, Di and Naumann, Tristan and
               McDermott, Matthew},
  booktitle = {Proceedings of the 2nd Clinical Natural Language Processing
               Workshop},
  pages     = {72--78},
  year      = {2019},
  url       = {https://aclanthology.org/W19-1909/}
}
```

### i2b2 / Stanza clinical NER
Clinical entity extraction with Stanza's i2b2 model, linked to HPO by embedding
similarity. Wrappers: `baselines/*/i2b2.py`; extractor `StanzaEntityExtractor`
in `rdma/hporag/entity.py`.

```bibtex
@article{zhang2021biomedical,
  title   = {Biomedical and clinical English model packages for the Stanza
             Python NLP library},
  author  = {Zhang, Yuhao and Zhang, Yuhui and Qi, Peng and
             Manning, Christopher D and Langlotz, Curtis P},
  journal = {Journal of the American Medical Informatics Association},
  volume  = {28},
  number  = {9},
  pages   = {1892--1899},
  year    = {2021},
  doi     = {10.1093/jamia/ocab090}
}
```

### RDRAG, Dictionary, Zero-shot
Our own baselines, implemented for this paper — no external citation.

- **RDRAG** — LLM extraction plus embedding-based ontology matching, without the
  verifier and supervisor agents (`baselines/{raredis,rdd,mimic3_rd_mining_code}/rdrag.py`)
- **Dictionary** — retrieval and string matching, no LLM
  (`baselines/*/dict.py`, `baselines/*/dictionary_hpo.py`)
- **Zero-shot** — direct LLM prompting (`baselines/*/zeroshot.py`)

---

## Frameworks, models and tools

### PyHealth
The dataset/task framework RDMA's benchmark layer plugs into.

```bibtex
@inproceedings{yang2023pyhealth,
  title     = {PyHealth: A Deep Learning Toolkit for Healthcare Applications},
  author    = {Yang, Chaoqi and Wu, Zhenbang and Jiang, Patrick and Lin, Zhen
               and Gao, Junyi and Danek, Benjamin P and Sun, Jimeng},
  booktitle = {Proceedings of the 29th ACM SIGKDD Conference on Knowledge
               Discovery and Data Mining (KDD)},
  pages     = {5788--5789},
  year      = {2023},
  doi       = {10.1145/3580305.3599178}
}
```

### Embedding models
- **MedEmbed** — [`abhinand/MedEmbed-small-v0.1`](https://huggingface.co/abhinand/MedEmbed-small-v0.1), the default retriever
- **BGE** — [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5); cite Xiao et al., *C-Pack*, SIGIR 2024, doi `10.1145/3626772.3657878`
- **MedCPT** — [`ncbi/MedCPT`](https://huggingface.co/ncbi/MedCPT-Query-Encoder):

```bibtex
@article{jin2023medcpt,
  title   = {MedCPT: Contrastive Pre-trained Transformers with large-scale
             PubMed search logs for zero-shot biomedical information retrieval},
  author  = {Jin, Qiao and Kim, Won and Chen, Qingyu and Comeau, Donald C and
             Yeganova, Lana and Wilbur, W John and Lu, Zhiyong},
  journal = {Bioinformatics},
  volume  = {39},
  number  = {11},
  pages   = {btad651},
  year    = {2023},
  doi     = {10.1093/bioinformatics/btad651}
}
```

### NLP libraries
- **spaCy** / **scispaCy** — biomedical NLP pipelines. Cite Neumann et al.,
  *ScispaCy: Fast and Robust Models for Biomedical Natural Language Processing*,
  BioNLP 2019.
- **negspacy** — negation detection, implementing the NegEx algorithm
  (Chapman et al., 2001).
- **Stanza** — see the i2b2 baseline entry above.
- **FAISS** — similarity search. Cite Johnson et al., *Billion-scale similarity
  search with GPUs*, IEEE Transactions on Big Data, 2019.

### Ontologies
- **Human Phenotype Ontology** — <https://hpo.jax.org>, CC BY 4.0. Cite the
  current HPO release paper (Köhler et al. / Gargano et al., *Nucleic Acids
  Research*).
- **Orphanet / ORDO** — <https://www.orphadata.com>, CC BY 4.0.
- **UMLS** — © National Library of Medicine. License-restricted; obtain your own
  from <https://uts.nlm.nih.gov/uts/signup-login>.

### Evaluated LLMs
Results reported in the paper use Llama-3 (8B/70B), Mistral-24B, Qwen3-32B,
Nemotron-120B and GPT-5 via Azure OpenAI. Cite each model's own release
documentation as appropriate.
