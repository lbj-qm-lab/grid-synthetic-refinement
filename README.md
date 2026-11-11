# GRID: Gradient-based Refinement of Imbalanced Synthetic Data

![Python 3.10](https://img.shields.io/badge/python-3.10-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Code for the paper *Predicting levodopa-induced dyskinesia from serial dopamine transporter imaging with
boundary-refined synthetic data*.

GRID refines synthetic minority-class samples after they are generated, whatever the generator, and adds the refined
samples to the training data of a downstream classifier. It has three phases, all fitted inside each training fold.

| Phase | Step | Code |
|---|---|---|
| I | **Knowledge distillation.** A teacher classifier is fitted on the training data and its decision boundary is distilled into a small differentiable surrogate. | `grid/refinement/teacher.py`, `refiner.py` |
| II | **Class-conditional manifold learning.** One autoencoder per class is fitted on the real training samples of that class. | `grid/refinement/refiner.py` |
| III | **Refinement.** Each synthetic minority sample is moved for K Adam steps along the gradient of the surrogate's minority-class loss, with the autoencoder reconstruction error as a penalty that keeps it close to the real minority samples. | `grid/refinement/refiner.py` |

The PPMI cohort of the paper cannot be redistributed. The scripts here run the same experiment on the six public
clinical datasets of the paper, or on any other CSV.

## Repository structure

```
main.py                    entry point: runs the experiment on one CSV
grid/
  config.py                all settings: cross-validation, strategies, generators, GRID, classifiers
  generators/              SMOTE, ADASYN, ROS; CTGAN, TVAE, CopulaGAN (SDV)
  refinement/              GRID: teacher, surrogate, per-class autoencoders, refinement loop
  classifiers/             the seven downstream classifiers
  pipeline/                augmentation, nested cross-validation, result files
  data/                    CSV and metadata loading
experiments/public/        public datasets: download script, per-dataset metadata, run script
```

## Installation

Python 3.10 and an NVIDIA GPU.

```bash
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128   # match your CUDA version
pip install -r requirements.txt
```

If the TabPFN weights download asks for authentication, follow https://docs.priorlabs.ai/how-to-access-gated-models.

## Usage

A run is 5 repeats of 5 x 5 nested cross-validation on one dataset. In every inner fold:

1. Each of six generators (SMOTE, ADASYN, ROS, CTGAN, TVAE, CopulaGAN) is fitted on the inner-training split and
   oversamples the minority class to parity with the majority class.
2. For GRID, the teacher, surrogate and autoencoders are fitted on the same split and the synthetic samples are refined.
3. Each of seven classifiers (logistic regression, random forest, XGBoost, RealMLP, FT-Transformer, TabM, TabPFN) is
   trained under three strategies: the inner-training split alone (Original), with the unrefined synthetic samples
   (Synthetic Aug.), and with the refined samples (GRID).
4. The five inner models are averaged on the held-out outer fold, giving one out-of-fold probability per row and repeat.

Generator outputs are cached, so a rerun of the same command skips step 1.

### Public clinical datasets

```bash
python experiments/public/fetch_data.py   # downloads the data -> data/public/*.csv
bash experiments/public/run_public.sh      # runs all six       -> results/public/<dataset>/

DATASETS="10_wpbc" bash experiments/public/run_public.sh   # one dataset only
```

| Dataset | File | Source | N | Imbalance ratio |
|---|---|---|---|---|
| WDBC | `2_breast_cancer_wdbc` | UCI 17 | 569 | 1.68 |
| Diabetes | `3_diabetes` | OpenML 37 | 768 | 1.87 |
| Heart Failure | `9_heart_failure_clinical_records` | UCI 519 | 299 | 2.11 |
| ILPD | `6_ilpd` | UCI 225 | 579 | 2.51 |
| WPBC | `10_wpbc` | UCI 16 | 194 | 3.22 |
| Thoracic Surgery | `8_thoracic_surgery` | UCI 277 | 470 | 5.71 |

The deep-generator hyperparameters selected for each dataset are in `experiments/public/metadata/`.

### Your own data

```bash
python main.py --data my_data.csv [--metadata my_data.yaml]     # -> results/my_data/
```

The CSV holds numeric predictors and a binary `target` column (1 = minority class), with no missing values. The
optional YAML, in the format of `experiments/public/metadata/`, lists the categorical predictors and the CTGAN,
TVAE and CopulaGAN hyperparameters; without it, all predictors are treated as continuous and the generator defaults
of `grid/config.py` are used. `python main.py -h` lists the other options.

### Output

| File | Content |
|---|---|
| `ALL_PREDICTIONS_REPEATED_*.csv` | out-of-fold predicted probability of every row, per strategy, classifier and repeat |
| `METRICS_REPEATED_*.csv` | AUROC and Brier score of each outer fold |
| `SUMMARY_REPEATED_*.csv` | mean and SD of the fold-level metrics, as a quick check (the paper scores the repeat-averaged predictions) |
| `ALL_AUG_DATA_REPEATED_*.csv` | the augmented training sets, with synthetic rows flagged |

The `Exp` column names the training condition: `1. Baseline` is Original, `2-x` and `3-x` are Synthetic Aug., and
`4-x` and `5-x` are GRID.

## License

The code is released under the MIT License (`LICENSE`). Two dependencies restrict commercial use: the SDV generators
(Business Source License 1.1) and the TabPFN-2.5 weights used by the downstream TabPFN classifier (TabPFN-2.5
License). The TabPFN v2 teacher is under the Prior Labs License 1.1. The full terms, and the licenses and citations
of the public datasets, are in `THIRD_PARTY_NOTICES.md`.

Built with PriorLabs-TabPFN.

## Citation

```bibtex
@unpublished{bu2026grid,
  title  = {Predicting levodopa-induced dyskinesia from serial dopamine transporter imaging
            with boundary-refined synthetic data},
  author = {Bu, Daehyuk and Song, Yoo Sung and Jeong, Eun Hye and Lee, Kee Yeun},
  year   = {2026},
  note   = {Manuscript submitted for publication}
}
```