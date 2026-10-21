# FactCheck

> **Research use only. Not a medical device.**
> This software was developed in the HEREDITARY project as a research prototype. It has not been assessed or certified under the EU Medical Device Regulation (EU) 2017/745. It must not be used to diagnose, treat, monitor or make any other decision about individual patients.

## 1. Overview

| Field | Value |
|---|---|
| Name | FactCheck benchmark (code, RAG dataset and Mock API) |
| Version | No tagged version (main branch) |
| Type | Software or library; Dataset |
| Owner partner and contact | University of Padua (UNIPD), Stefano Marchesin, stefano.marchesin@unipd.it |
| License | MIT (see LICENSE file) |
| Repository and DOI | https://github.com/FactCheck-AI/FactCheck, https://github.com/FactCheck-AI/FactCheck-MockAPI, https://huggingface.co/datasets/FactCheck-AI/FactCheck (no Zenodo DOI); Mock API deployed at https://factcheck-api.dei.unipd.it |
| Release date | 24 May 2025 (last update 22 July 2025) |
| Related task or deliverable | T4.1 |
| How to cite | F. Shami, S. Marchesin and G. Silvello. Benchmarking Large Language Models for Knowledge Graph Validation. Proc. 29th International Conference on Extending Database Technology, EDBT 2026, Tampere, Finland, pp. 551-565. OpenProceedings.org, 2026. doi:10.48786/EDBT.2026.45 |

## 2. Intended use

**Primary intended use.** Benchmark to evaluate LLMs for KG fact validation along three dimensions: internal knowledge (direct and guided iterative prompting), external evidence through retrieval-augmented generation, and multi-model consensus. It includes 13,530 facts from three KGs, a RAG corpus of 130,820 generated questions and 2,090,305 web documents, and a mock search API for reproducible retrieval.

**Intended users.** Researchers and developers working on KG quality and LLM evaluation.

**Out-of-scope uses.** This release must not be used for
- clinical diagnosis, prognosis or treatment of individual patients;
- decisions about individuals, such as insurance, employment or access to care;
- re-identification of individuals or linkage with other data sources;
- populations, devices or data types not described in Section 3;
- use of LLM verdicts as ground truth for real-world KG validation.

## 3. Data provenance

| Dataset | Source partner | Type (public, synthetic, pseudonymized, anonymized) | Population (size, age, sex, country, condition) | Ethics approval or access conditions |
|---|---|---|---|---|
| FactBench, English subset (2,800 facts) | External: Gerber et al., 2015 | Public | Not applicable (KG triples; no human population) | Publicly available |
| YAGO (1,386 facts) | External: Ojha and Talukdar, EMNLP 2017 | Public | Not applicable (KG triples; no human population) | Publicly available |
| DBpedia 2015-10 (9,344 facts) | UNIPD: Marchesin, Silvello and Alonso, HCOMP 2024 | Public (derived from anonymized annotations) | Not applicable (KG triples; no human population) | Publicly available |
| RAG corpus: LLM-generated questions and web documents retrieved through Google Search | UNIPD, built for this benchmark | Public web content | Not applicable | Third-party content; terms of the original websites apply; hosted on Hugging Face |

This repository and the released files contain no personal data, pseudonymized records, identifiers or pseudonymization keys.

## 6. Known limitations and risks

- **Generalizability.** English content only; Google Search was queried with English language settings and US geolocation. The KGs are general-domain.
- **Known biases and confounders.** A fact is considered true if the KG snapshot supports it. Class balance differs across datasets (proportion of correct facts 0.54 in FactBench, 0.99 in YAGO, 0.85 in DBpedia); on YAGO, LLMs biased towards "true" may obtain inflated scores. Wikipedia pages are filtered from the evidence of DBpedia and FactBench facts to avoid circular verification.
- **Failure modes.** The evaluated LLMs are not stable or reliable enough for real-world KG validation. RAG gives inconsistent improvements at higher computational cost, and consensus strategies do not consistently outperform single models. About 13% of the retrieved documents have empty text.
- **Privacy risks.** No trained model. The RAG corpus consists of publicly accessible web pages collected as retrieved. The Mock API stores the username and email used to request an API key, with usage counters.

## 7. Responsible use and reporting

- **Terms of use.** MIT license. Research use only, as stated in the disclaimer above. The web documents in the RAG corpus remain subject to the terms of their original websites.
- **Contact for misuse or vulnerability reports.**

## 8. Acknowledgement

Funded by the European Union under grant agreement No 101137074 (HEREDITARY). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Health and Digital Executive Agency (HaDEA). Neither the European Union nor the granting authority can be held responsible for them.
