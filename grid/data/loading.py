import numpy as np
import pandas as pd
import yaml


def load_dataset(path):
    print(f"[Data] Loading from {path}...")
    df = pd.read_csv(path)
    if 'target' not in df.columns:
        raise ValueError(f"{path}: no `target` column")
    X, y = df.drop(columns=['target']), df['target']

    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        raise ValueError(f"{path}: non-numeric predictors {non_numeric} — encode them as numbers first")
    missing = [c for c in df.columns if df[c].isna().any()]
    if missing:
        raise ValueError(f"{path}: missing values in {missing} — remove or impute them first")
    if not set(y.unique()) <= {0, 1}:
        raise ValueError(f"{path}: `target` must be 0/1, found {sorted(y.unique())}")
    return X, y.astype(int)


def load_metadata(path, X):
    if path is None:
        return [], {}, None
    with open(path) as f:
        meta = yaml.safe_load(f) or {}
    categorical = meta.get("categorical") or {}
    cat_cols = list(categorical)
    cat_allowed = {c: np.array(v) for c, v in categorical.items()}

    absent = [c for c in cat_cols if c not in X.columns]
    if absent:
        raise ValueError(f"{path}: categorical columns {absent} are not in the data")
    for c in cat_cols:
        outside = sorted(set(X[c].unique().tolist()) - set(cat_allowed[c].tolist()))
        if outside:
            raise ValueError(f"{path}: `{c}` has values {outside} outside its allowed codes {cat_allowed[c].tolist()}")

    tuned_params = {algo: entry["params"] for algo, entry in (meta.get("generators") or {}).items()} or None
    return cat_cols, cat_allowed, tuned_params


def detect_int_cols(X_df, cat_cols):
    int_cols = []
    for c in X_df.columns:
        if c in cat_cols:
            continue
        arr = X_df[c].to_numpy(dtype=float)
        if X_df[c].dtype.kind in ('i', 'u') or np.all(arr == np.round(arr)):
            int_cols.append(c)
    return int_cols


def get_int_bounds(X_df, int_cols):
    bounds = {}
    for c in int_cols:
        arr = X_df[c].to_numpy(dtype=float)
        bounds[c] = (int(np.floor(arr.min())), int(np.ceil(arr.max())))
    return bounds


def snap_integers_global(df, int_cols, int_bounds, global_scaler, num_cols_order):
    sc = np.asarray(global_scaler.scale_, dtype=float)
    s_arr = 1.0 / sc
    o_arr = -np.asarray(global_scaler.mean_, dtype=float) / sc
    df = df.copy()
    name_to_j = {c: i for i, c in enumerate(num_cols_order)}
    for c in int_cols:
        j = name_to_j[c]
        s, o = float(s_arr[j]), float(o_arr[j])
        lo, hi = int_bounds[c]
        x_orig = np.clip(np.round((df[c].to_numpy(dtype=float) - o) / s), lo, hi)
        df[c] = s * x_orig + o
    return df


def get_target_counts(y):
    counts = y.value_counts().to_dict()
    max_c = max(counts.values())
    return {c: max_c for c in counts}


def cat_cols_to_indices(X_df, cat_cols):
    return [X_df.columns.get_loc(c) for c in cat_cols]


def snap_categoricals(df, cat_allowed):
    df = df.copy()
    for c, allowed in cat_allowed.items():
        x = df[c].to_numpy(dtype=float)
        idx = np.argmin(np.abs(x[:, None] - allowed[None, :]), axis=1)
        df[c] = allowed[idx].astype(int)
    return df
