# Third-party software, model weights, and data

The code in this repository is released under the MIT License (see `LICENSE`). It does **not** include any
third-party source code, model weights, or data. Running it installs or downloads the components below, and each
of them is governed by its own license. In particular, **two dependencies restrict commercial use** (SDV family and
the TabPFN-2.5 weights). Check those terms before you use this code outside non-commercial research.

## Dependencies with additional terms

| Component | Used for | License | What it means for users |
|---|---|---|---|
| `sdv` 1.29.1, `ctgan` 0.11.1, `rdt` 1.18.2, `copulas` 0.12.3 (DataCebo) | CTGAN, TVAE and CopulaGAN base generators | Business Source License 1.1 | Use is permitted except for offering a commercial "Synthetic Data Service" to third parties. Each release changes to the MIT License four years after its release date. |
| `tabpfn` 6.0.6 (Prior Labs) | TabPFN classifier and the refinement teacher | Prior Labs License 1.1 (Apache-2.0 with an added attribution clause, §10) | If you distribute work that contains TabPFN source or weights, include the license and display "Built with PriorLabs-TabPFN". If you distribute an AI model trained, fine-tuned or distilled with TabPFN outputs, its name must begin with "TabPFN". Internal benchmarking without distribution needs no attribution. |
| TabPFN v2 classifier weights (`Prior-Labs/TabPFN-v2-clf`, `tabpfn-v2-classifier-finetuned-zk73skhh.ckpt`) | Refinement teacher (Phase I), the only TabPFN model that GRID distils | Prior Labs License 1.1 (the license file of that Hugging Face repository) | As above, with no restriction to non-commercial use. Downloaded automatically into the TabPFN cache. |
| TabPFN-2.5 classifier weights (`Prior-Labs/tabpfn_2_5`, `tabpfn-v2.5-classifier-v2.5_default.ckpt`) | Only one of the seven downstream classifiers (the "TabPFN" arm); its outputs are used only as evaluation predictions | TabPFN-2.5 License v1.1 | Research, testing, evaluation and internal benchmarking only. The weights, their derivatives and their outputs may not be used for commercial or production purposes without a commercial license from Prior Labs, and the outputs may not be used to distil a competing model. Downloaded automatically. |

Built with PriorLabs-TabPFN.

**Distillation in GRID and the TabPFN naming clause.** Phase I distils the TabPFN v2 teacher into a small student
network inside every training fold. The student is used only to compute the refinement gradient and is discarded when
the fold ends. This repository therefore distributes no model trained or distilled with TabPFN, and the naming clause
of §10 is not triggered by the repository itself. It applies if you distribute or make available such a model.
This includes the student network. It arguably also includes classifiers trained on GRID-refined samples, which
are produced with the outputs of the distilled teacher. Such a model must carry a name beginning with "TabPFN".
Internal benchmarking and testing without external communication are not distribution under §10.

**Avoiding the non-commercial TabPFN-2.5 weights.** GRID itself needs only the v2 weights. Removing `"TabPFN"` from
`CLF_NAMES` in `grid/config.py` drops the only component that uses the TabPFN-2.5 weights. The reported results
include that classifier.

## Other dependencies (permissive licenses)

| Package (version used) | License |
|---|---|
| numpy 2.2.6, pandas 2.2.2, scipy 1.15.3, scikit-learn 1.6.0, joblib 1.5.2 | BSD-3-Clause |
| torch 2.9.1 | BSD-3-Clause |
| imbalanced-learn 0.14.0 (SMOTE, ADASYN) | MIT |
| pytabkit 1.7.2 (RealMLP, TabM, FT-Transformer, RF and XGBoost wrappers) | Apache-2.0 |
| xgboost 3.1.0 | Apache-2.0 |
| lightning 2.5.5 (pytabkit dependency) | Apache-2.0 |
| tqdm 4.67.1 | MPL-2.0 and MIT |
| PyYAML 6.0.3 | MIT |
| ucimlrepo 0.0.7 (data download) | MIT |

## Public clinical datasets (downloaded by `experiments/public/fetch_data.py`, not redistributed)

| File | Source | License | Citation |
|---|---|---|---|
| `2_breast_cancer_wdbc` | UCI ML Repository, id 17 | CC BY 4.0 | Wolberg W, Mangasarian O, Street N, Street W (1993). Breast Cancer Wisconsin (Diagnostic). https://doi.org/10.24432/C5DW2B |
| `10_wpbc` | UCI ML Repository, id 16 | CC BY 4.0 | Wolberg W, Street W, Mangasarian O (1995). Breast Cancer Wisconsin (Prognostic). https://doi.org/10.24432/C5GK50 |
| `6_ilpd` | UCI ML Repository, id 225 | CC BY 4.0 | Ramana B, Venkateswarlu N (2022). ILPD (Indian Liver Patient Dataset). https://doi.org/10.24432/C5D02C |
| `8_thoracic_surgery` | UCI ML Repository, id 277 | CC BY 4.0 | Lubicz M, Pawelczyk K, Rzechonek A, Kolodziej J (2014). Thoracic Surgery Data. https://doi.org/10.24432/C5Z60N |
| `9_heart_failure_clinical_records` | UCI ML Repository, id 519 | CC BY 4.0 | Heart Failure Clinical Records (2020). https://doi.org/10.24432/C5Z89R |
| `3_diabetes` | OpenML data_id 37 (Pima Indians Diabetes) | listed as "Public" on OpenML | Smith JW et al. (1988), Proc. Annu. Symp. Comput. Appl. Med. Care; originally distributed by the UCI ML Repository, which no longer hosts it |

The derived analysis files are produced by `experiments/public/fetch_data.py` (column renaming, coding of categorical strings,
removal of rows with missing values and of follow-up time where noted in that script).

## Data that are not part of this repository

The primary cohort of the paper comes from the Parkinson's Progression Markers Initiative (PPMI). PPMI data are
available to qualified investigators through https://www.ppmi-info.org and may not be redistributed. This repository
contains no PPMI data, no data derived from PPMI, and no models trained on PPMI data.
