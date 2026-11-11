"""Run the GRID experiment on one dataset with the settings of the paper (grid/config.py).

    python main.py --data my_data.csv [--metadata my_data.yaml]

Step 1 fills the generator-output cache. Step 2 runs the repeated nested cross-validation of the 13 arms x 7
classifiers, outer folds in parallel, and writes
<save_dir>/<run_name>/{METRICS,ALL_PREDICTIONS,ALL_AUG_DATA,SUMMARY}_REPEATED_*.csv.
--seed, --n_repeats and --n_folds override grid/config.py.
"""
import argparse
import os
import warnings

from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from grid.config import CV, EXPERIMENTS
from grid.data.loading import detect_int_cols, get_int_bounds, load_dataset, load_metadata
from grid.pipeline.aggregate import aggregate_and_save
from grid.pipeline.nested_cv import process_nested_fold

warnings.filterwarnings('ignore')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="CSV with numeric predictors and a binary `target` column")
    ap.add_argument("--metadata", default=None, help="dataset YAML (format of experiments/public/metadata/)")
    ap.add_argument("--run_name", default=None, help="output folder name (default: data file name)")
    ap.add_argument("--save_dir", default="results")
    ap.add_argument("--cache_dir", default="cache/generator")
    ap.add_argument("--seed", type=int, default=CV["seed"])
    ap.add_argument("--n_repeats", type=int, default=CV["n_repeats"])
    ap.add_argument("--n_folds", type=int, default=CV["n_outer_folds"], help="outer folds")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_jobs", type=int, default=5, help="outer folds run in parallel (default 5)")
    a = ap.parse_args()

    if not os.path.isfile(a.data):
        raise SystemExit(f"[stop] {a.data} not found")
    run = a.run_name or os.path.splitext(os.path.basename(a.data))[0]
    cache_dir = os.path.join(a.cache_dir, f"{run}_seed{a.seed}")
    save_dir = os.path.join(a.save_dir, run)
    # inherited by the worker processes: fixes their string-hash order
    os.environ["PYTHONHASHSEED"] = str(a.seed)

    X, y = load_dataset(a.data)
    cat_cols, cat_allowed, tuned_params = load_metadata(a.metadata, X)
    int_cols = detect_int_cols(X, cat_cols)
    int_bounds = get_int_bounds(X, int_cols)
    num_cols = [c for c in X.columns if c not in cat_cols]
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    print(f"=== {run}: {a.data} (metadata {a.metadata}) — {len(X)} rows, {len(num_cols)} continuous, "
          f"{len(cat_cols)} categorical, integer-valued {int_cols}")

    tasks = []
    for rep_idx in range(a.n_repeats):
        skf = StratifiedKFold(n_splits=a.n_folds, shuffle=True, random_state=a.seed + rep_idx)
        for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
            tasks.append((rep_idx, fold_idx, tr_idx, te_idx, a.seed + rep_idx * a.n_folds + fold_idx))

    def run_folds(gen_cache_only):
        return Parallel(n_jobs=a.n_jobs)(
            delayed(process_nested_fold)(
                rep_idx, fold_idx, tr_idx, te_idx, X, y, EXPERIMENTS,
                a.device, task_seed,
                cat_cols, cat_allowed,
                int_cols, int_bounds, scaler, num_cols,
                cache_dir, gen_cache_only, tuned_params
            )
            for rep_idx, fold_idx, tr_idx, te_idx, task_seed in tqdm(tasks, desc="outer folds")
        )

    print(f"    step 1/2: generator cache -> {cache_dir} (existing outputs are reused)", flush=True)
    run_folds(gen_cache_only=True)

    print(f"    step 2/2: nested CV, {len(tasks)} outer folds, {a.n_jobs} in parallel -> {save_dir}", flush=True)
    os.makedirs(save_dir, exist_ok=True)
    results = run_folds(gen_cache_only=False)
    aggregate_and_save([(t[0], t[1], res) for t, res in zip(tasks, results)], save_dir)


if __name__ == "__main__":
    main()
