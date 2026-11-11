import os
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from grid.data.loading import get_target_counts, cat_cols_to_indices, snap_integers_global
from grid.generators import oversampling as OS
from grid.generators.deep import generate_base_data
from grid.config import GRID_PARAMS
from grid.refinement.refiner import ConditionalRefiner
from grid.utils.seed import seed_basic

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def run_pipeline(X_train, y_train, method_config, device, seed, cat_cols, cat_allowed, step, refine=True,
                 int_cols=None, int_bounds=None, global_scaler=None, num_cols_order=None,
                 precomputed_teacher=None, gen_cache_dir=None, tuned_params=None):
    algo = method_config["algo"]
    if algo == "base":
        return X_train, y_train
    use_refinement = method_config.get("grid", False) and refine

    target_counts = get_target_counts(y_train)
    cat_idx = cat_cols_to_indices(X_train, cat_cols)

    # Generator output (cached: it does not depend on the refinement settings)
    cache_path = None
    if gen_cache_dir is not None:
        os.makedirs(gen_cache_dir, exist_ok=True)
        cache_path = os.path.join(gen_cache_dir, f"{algo}__{step}.pkl")

    if cache_path is not None and os.path.exists(cache_path):
        payload = pd.read_pickle(cache_path)
        if payload["seed"] != seed or list(payload["X"].columns) != list(X_train.columns):
            raise ValueError(f"{cache_path}: made with another seed or other columns — delete it or use another "
                             f"--cache_dir")
        X_gen = payload["X"].copy()
        y_gen = payload["y"].copy()
    else:
        if algo == "ros":
            X_gen, y_gen = OS.ros(X_train, y_train, target_counts, seed)
        elif algo == "smote":
            X_gen, y_gen = OS.smote(X_train, y_train, target_counts, cat_idx, cat_allowed, seed)
        elif algo == "adasyn":
            X_gen, y_gen = OS.adasyn(X_train, y_train, target_counts, cat_allowed, seed)
        else:  # CTGAN / TVAE / CopulaGAN
            X_gen, y_gen = generate_base_data(
                X_train, y_train, target_counts, algo, device, seed, cat_cols=cat_cols, cat_allowed=cat_allowed,
                tuned_params=tuned_params,
            )
        if X_gen.empty:
            raise RuntimeError(f"{algo} returned no synthetic samples ({step}, seed {seed})")

        if cache_path is not None:
            tmp_path = f"{cache_path}.tmp{os.getpid()}"
            pd.to_pickle({"X": X_gen, "y": y_gen, "seed": seed}, tmp_path)
            os.replace(tmp_path, cache_path)  # atomic write

    # GRID refinement, continuous predictors min-max scaled to [0, 1] on the training set
    if use_refinement:
        _num_cols = [c for c in X_train.columns if c not in cat_cols]
        _mm = None
        _teacher_inv = None
        X_train_ref, X_gen_ref = X_train, X_gen
        if len(_num_cols) > 0:
            _mm = MinMaxScaler().fit(X_train[_num_cols].astype(float))

            def _to01(df):
                d = df.copy()
                d[_num_cols] = _mm.transform(d[_num_cols].astype(float))
                return d

            X_train_ref = _to01(X_train)
            X_gen_ref = _to01(X_gen)

            # the teacher was fitted in the model space
            _npos = [X_train.columns.get_loc(c) for c in _num_cols]

            def _teacher_inv(arr, _npos=_npos, _mm=_mm):
                out = np.array(arr, dtype=float, copy=True)
                out[:, _npos] = _mm.inverse_transform(out[:, _npos])
                return out

        seed_basic(seed)

        refiner = ConditionalRefiner(device=device, **GRID_PARAMS)
        refiner.fit(X_train_ref, y_train, precomputed_teacher, seed=seed, cat_idx=cat_idx, cat_cols=cat_cols,
                    cat_allowed=cat_allowed, teacher_inverse=_teacher_inv)
        X_gen = refiner.optimize(X_gen_ref, y_gen)

        if _mm is not None:  # [0, 1] -> model space
            X_gen = X_gen.copy()
            X_gen[_num_cols] = _mm.inverse_transform(X_gen[_num_cols].astype(float))

        if int_cols:
            X_gen = snap_integers_global(X_gen, int_cols, int_bounds, global_scaler, num_cols_order)

    X_out = pd.concat([X_train, X_gen]).reset_index(drop=True)
    y_out = pd.concat([y_train, y_gen]).reset_index(drop=True)
    for c in cat_cols:
        X_out[c] = X_out[c].astype("category")
    return X_out, y_out
