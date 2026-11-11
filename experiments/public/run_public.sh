#!/bin/bash
# run_public.sh — main.py on the six public datasets, each with its metadata YAML.
# Run after `python experiments/public/fetch_data.py`.
#
#   bash experiments/public/run_public.sh
#   DATASETS="9_heart_failure_clinical_records" bash experiments/public/run_public.sh
#   NJ=3 bash experiments/public/run_public.sh          # fewer outer folds in parallel
#
# Outputs: results/public/<dataset>/ (see main.py). Re-running skips finished datasets and reuses the generator cache.
set -euo pipefail
cd "$(dirname "$0")/../.."

DATASETS=${DATASETS:-"9_heart_failure_clinical_records 10_wpbc 8_thoracic_surgery \
2_breast_cancer_wdbc 6_ilpd 3_diabetes"}
NJ=${NJ:-5}

# pytabkit creates an empty lightning_logs/version_N folder at every predict(); a large folder slows predict() down.
# This janitor empties it while the run is alive (results are unaffected).
janitor () {
    while kill -0 "$1" 2>/dev/null; do
        if [ -d lightning_logs ] && [ "$(ls -f lightning_logs | wc -l)" -gt 2000 ]; then
            t="lightning_logs_trash_$(date +%s)"
            mv lightning_logs "$t" 2>/dev/null && find "$t" -depth -type d -empty -delete 2>/dev/null || true
        fi
        sleep 30
    done
}
janitor $$ &
trap 'kill $! 2>/dev/null || true' EXIT

for ds in $DATASETS; do
    if ls "results/public/$ds"/METRICS_REPEATED_*.csv >/dev/null 2>&1; then
        echo "=== $ds: done already — skipped"; continue
    fi
    python main.py --data "data/public/${ds}.csv" --metadata "experiments/public/metadata/${ds}.yaml" \
        --save_dir results/public --n_jobs "$NJ"
done
