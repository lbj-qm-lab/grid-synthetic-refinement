import gc

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from grid.classifiers import registry as CLF
from grid.config import CLF_NAMES, CV, INNER_SEED_STRIDE
from grid.pipeline.augment import run_pipeline
from grid.refinement.teacher import make_teacher
from grid.utils.seed import seed_basic


def process_nested_fold(rep_idx, fold_idx, tr_idx, te_idx, X, y, experiments, device_str, base_seed,
                        cat_cols, cat_allowed, int_cols=None, int_bounds=None, global_scaler=None, num_cols_order=None,
                        gen_cache_dir=None, gen_cache_only=False, tuned_params=None):
    step_seed = base_seed + fold_idx
    seed_basic(step_seed)

    X_outer_tr, X_outer_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_outer_tr, y_outer_te = y.iloc[tr_idx], y.iloc[te_idx]

    all_cols = X_outer_tr.columns.tolist()
    real_cat_cols = [c for c in cat_cols if c in all_cols]
    num_cols = [c for c in all_cols if c not in real_cat_cols]
    new_order = num_cols + real_cat_cols

    X_outer_tr = X_outer_tr[new_order]
    X_outer_te = X_outer_te[new_order]

    n_num_features = len(num_cols)
    cat_idx = [i + n_num_features for i in range(len(real_cat_cols))]

    for c in real_cat_cols:
        X_outer_tr[c] = X_outer_tr[c].astype("category")
        X_outer_te[c] = X_outer_te[c].astype("category")

    fold_metrics = []
    fold_preds = []
    fold_aug_data = []

    clf_names = CLF_NAMES

    teacher_cache = {}  # inner_k -> teacher, shared by all GRID arms

    for exp in experiments:
        print(f"Fold {fold_idx} | Exp: {exp['name']} | Starting Nested Loop...")

        clf_preds_accum = {cn: np.zeros(len(X_outer_te)) for cn in clf_names}

        seed_basic(step_seed)
        inner_skf = StratifiedKFold(n_splits=CV["n_inner_folds"], shuffle=True, random_state=step_seed)

        for inner_k, (in_tr_idx, in_val_idx) in enumerate(inner_skf.split(X_outer_tr, y_outer_tr)):

            X_in_tr = X_outer_tr.iloc[in_tr_idx].copy()
            y_in_tr = y_outer_tr.iloc[in_tr_idx].copy()
            X_in_val = X_outer_tr.iloc[in_val_idx].copy()
            y_in_val = y_outer_tr.iloc[in_val_idx].copy()

            current_inner_seed = step_seed + (inner_k * INNER_SEED_STRIDE)

            use_refinement_for_exp = exp.get('grid', False) and (not gen_cache_only)
            precomputed_teacher = None
            if use_refinement_for_exp:
                if inner_k not in teacher_cache:
                    seed_basic(current_inner_seed)
                    teacher_cache[inner_k] = make_teacher(
                        X_in_tr, y_in_tr,
                        seed=current_inner_seed,
                        cat_idx=cat_idx,
                        device=device_str,
                    )
                precomputed_teacher = teacher_cache[inner_k]

            if gen_cache_only:
                seed_basic(current_inner_seed)
                run_pipeline(
                    X_in_tr, y_in_tr, exp, device_str, current_inner_seed,
                    cat_cols, cat_allowed,
                    step=f"rep{rep_idx}_fold{fold_idx}_inner{inner_k}", refine=False,
                    int_cols=int_cols, int_bounds=int_bounds,
                    global_scaler=global_scaler, num_cols_order=num_cols_order,
                    gen_cache_dir=gen_cache_dir, tuned_params=tuned_params,
                )
                continue

            seed_basic(current_inner_seed)
            X_aug, y_aug = run_pipeline(
                X_in_tr, y_in_tr, exp, device_str, current_inner_seed,
                cat_cols, cat_allowed,
                step=f"rep{rep_idx}_fold{fold_idx}_inner{inner_k}",
                int_cols=int_cols, int_bounds=int_bounds,
                global_scaler=global_scaler, num_cols_order=num_cols_order,
                precomputed_teacher=precomputed_teacher,
                gen_cache_dir=gen_cache_dir, tuned_params=tuned_params,
            )

            n_aug_count = len(X_aug) - len(X_in_tr)
            aug_df = X_aug.copy()
            aug_df['target'] = y_aug.values
            aug_df['Exp'] = exp['name']
            aug_df['OuterFold'] = fold_idx
            aug_df['InnerFold'] = inner_k
            aug_df['is_syn'] = [0] * len(X_in_tr) + [1] * n_aug_count
            fold_aug_data.append(aug_df)

            for clf_name in clf_names:
                model = CLF.build(clf_name, current_inner_seed, device_str)
                if clf_name == "TabPFN":
                    model.fit(X_aug, y_aug)
                    final_prob = model.predict_proba(X_outer_te)[:, 1]
                elif clf_name == "LogisticReg":
                    model.fit(X_aug, y_aug)
                    final_prob = model.predict_proba(X_outer_te)[:, 1]
                else:  # PyTabKit: inner-validation split for early stopping only
                    model.fit(X_aug, y_aug, X_val=X_in_val, y_val=y_in_val, cat_col_names=real_cat_cols)
                    final_prob = model.predict_proba(X_outer_te)[:, 1]
                clf_preds_accum[clf_name] += final_prob

                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        for clf_name in clf_names:
            final_prob = clf_preds_accum[clf_name] / CV["n_inner_folds"]
            fold_metrics.append({
                "Exp": exp["name"], "Classifier": clf_name, "OuterFold": fold_idx,
                "AUC": roc_auc_score(y_outer_te, final_prob),
                "Brier": brier_score_loss(y_outer_te, final_prob),
            })
            fold_preds.append(pd.DataFrame({
                "row": X_outer_te.index,  # row number in the input CSV
                "y_true": y_outer_te.values,
                "y_prob": final_prob,
                "Exp": exp["name"],
                "Classifier": clf_name,
                "OuterFold": fold_idx,
            }))

    return fold_metrics, fold_preds, fold_aug_data
