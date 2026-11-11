import os

import pandas as pd
from time import strftime


def aggregate_and_save(done, save_dir):
    all_metrics, all_preds, all_aug_data = [], [], []

    print("Aggregating results...")
    for rep_idx, fold_idx, fold_res in done:
        if not fold_res:
            continue

        m, p, a = fold_res

        for metric_dict in m:
            metric_dict['Repeat'] = rep_idx
            all_metrics.append(metric_dict)

        for pred_df in p:
            pred_df['Repeat'] = rep_idx
            all_preds.append(pred_df)

        for aug_df in a:
            aug_df['Repeat'] = rep_idx
            all_aug_data.append(aug_df)

    now = strftime("%Y%m%d-%H%M%S")

    if all_metrics:
        res_df = pd.DataFrame(all_metrics)
        key_cols = ['Exp', 'Classifier', 'Repeat', 'OuterFold']
        cols = key_cols + [c for c in res_df.columns if c not in key_cols]
        res_df = res_df[cols]

        res_df.to_csv(os.path.join(save_dir, f"METRICS_REPEATED_{now}.csv"), index=False)

        summ = res_df.groupby(['Exp', 'Classifier'])[['AUC', 'Brier']].agg(['mean', 'std'])
        summ.to_csv(os.path.join(save_dir, f"SUMMARY_REPEATED_{now}.csv"))

        print("\n=== Final Repeated Nested CV Summary ===")
        print(summ)

    if all_preds:
        full_preds_df = pd.concat(all_preds, ignore_index=True)
        full_preds_df.to_csv(os.path.join(save_dir, f"ALL_PREDICTIONS_REPEATED_{now}.csv"), index=False)
        print(f"Saved All Predictions: {len(full_preds_df)} rows")

    if all_aug_data:
        full_aug_df = pd.concat(all_aug_data, ignore_index=True)
        full_aug_df.to_csv(os.path.join(save_dir, f"ALL_AUG_DATA_REPEATED_{now}.csv"), index=False)
        print(f"Saved Augmented Data: {len(full_aug_df)} rows")
