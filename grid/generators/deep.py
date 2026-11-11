import logging
import warnings

import numpy as np
import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.sampling import Condition
from sdv.single_table import CopulaGANSynthesizer, CTGANSynthesizer, TVAESynthesizer

from grid.config import DEEP_GENERATOR_DEFAULTS, DEEP_SAMPLING
from grid.data.loading import snap_categoricals
from grid.utils.seed import seed_basic

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
for _name in ["sdv", "sdv.single_table", "ctgan", "rdt", "copulas"]:
    logging.getLogger(_name).setLevel(logging.ERROR)

SYNTHESIZERS = {"ctgan": CTGANSynthesizer, "tvae": TVAESynthesizer, "copulagan": CopulaGANSynthesizer}


def generate_base_data(X_ctx, y_ctx, target_counts, algo, device, seed, cat_cols, cat_allowed, tuned_params=None):
    if algo not in SYNTHESIZERS:
        raise ValueError(f"unknown deep generator {algo!r} (expected one of {list(SYNTHESIZERS)})")

    counts = y_ctx.value_counts().to_dict()
    needed = {c: target_counts[c] - counts.get(c, 0) for c in target_counts if target_counts[c] > counts.get(c, 0)}
    if not needed:
        return pd.DataFrame(), pd.Series(dtype=y_ctx.dtype)

    # class as a categorical column (string labels for conditional sampling)
    df = X_ctx.copy()
    for col in cat_cols:
        if str(df[col].dtype) == "category":
            df[col] = df[col].astype(object)
    df["__y__"] = y_ctx.values.astype(np.int64).astype(str)

    meta = SingleTableMetadata()
    meta.detect_from_dataframe(df)
    for col in cat_cols:
        meta.update_column(column_name=col, sdtype="categorical")
    meta.update_column(column_name="__y__", sdtype="categorical")

    kw = dict(tuned_params[algo]) if tuned_params and algo in tuned_params else dict(DEEP_GENERATOR_DEFAULTS[algo])
    for k in ("compress_dims", "decompress_dims", "generator_dim", "discriminator_dim"):
        if k in kw:
            kw[k] = tuple(kw[k])
    kw["cuda"] = str(device).startswith("cuda")

    seed_basic(seed)
    synth = SYNTHESIZERS[algo](metadata=meta, **kw)
    seed_basic(seed)
    synth.fit(df)

    X_parts, y_parts = [], []
    for c, n_need in needed.items():
        cond = Condition(num_rows=int(n_need), column_values={"__y__": str(int(c))})
        df_s = synth.sample_from_conditions([cond], batch_size=DEEP_SAMPLING["batch_size"],
                                            max_tries_per_batch=DEEP_SAMPLING["max_tries_per_batch"])
        if len(df_s) < n_need:
            raise RuntimeError(f"{algo}: conditional sampling returned {len(df_s)} of {n_need} rows of class {c} "
                               f"(seed {seed})")
        y_parts.append(pd.to_numeric(df_s["__y__"], errors="coerce").fillna(-1).astype(int))
        X_parts.append(df_s.drop(columns=["__y__"]).copy())

    X_pool = snap_categoricals(pd.concat(X_parts, axis=0).reset_index(drop=True), cat_allowed)
    y_pool = pd.concat(y_parts, axis=0).reset_index(drop=True).astype(int)

    X_new, y_new = [], []
    for c, n in needed.items():
        X_new.append(X_pool[y_pool == c].sample(n=n, replace=False, random_state=seed))
        y_new.append(pd.Series([c] * n))
    return pd.concat(X_new).reset_index(drop=True), pd.concat(y_new).reset_index(drop=True)
