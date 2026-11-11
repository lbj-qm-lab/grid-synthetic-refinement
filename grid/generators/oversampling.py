import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE, SMOTENC

from grid.data.loading import snap_categoricals


def ros(X_train, y_train, target_counts, seed):
    rng = np.random.default_rng(seed)
    X_parts, y_parts = [], []
    for c, n_target in target_counts.items():
        n_need = n_target - (y_train == c).sum()
        if n_need <= 0:
            continue
        X_c = X_train[y_train == c]
        idx = rng.choice(len(X_c), size=n_need, replace=True)
        X_parts.append(X_c.iloc[idx].reset_index(drop=True))
        y_parts.append(pd.Series([c] * n_need))
    if not X_parts:
        return pd.DataFrame(), pd.Series(dtype=y_train.dtype)
    return (pd.concat(X_parts, axis=0).reset_index(drop=True),
            pd.concat(y_parts, axis=0).reset_index(drop=True).astype(int))


def _synthetic_rows(sampler, X_train, y_train, cat_allowed):
    X_res, y_res = sampler.fit_resample(X_train, y_train)
    n_train = len(X_train)
    X_gen = X_res.iloc[n_train:].copy().reset_index(drop=True)
    y_gen = y_res.iloc[n_train:].copy().reset_index(drop=True)
    return snap_categoricals(X_gen, cat_allowed), y_gen


def smote(X_train, y_train, target_counts, cat_idx, cat_allowed, seed):
    if cat_idx:
        sampler = SMOTENC(categorical_features=cat_idx, sampling_strategy=target_counts, random_state=seed)
    else:
        sampler = SMOTE(sampling_strategy=target_counts, random_state=seed)
    return _synthetic_rows(sampler, X_train, y_train, cat_allowed)


def adasyn(X_train, y_train, target_counts, cat_allowed, seed):
    sampler = ADASYN(sampling_strategy=target_counts, random_state=seed)
    return _synthetic_rows(sampler, X_train, y_train, cat_allowed)
