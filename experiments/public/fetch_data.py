import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

UCI_ID = {"2_breast_cancer_wdbc": 17, "6_ilpd": 225, "8_thoracic_surgery": 277,
          "9_heart_failure_clinical_records": 519, "10_wpbc": 16}
DATASETS = ["2_breast_cancer_wdbc", "3_diabetes", "6_ilpd", "8_thoracic_surgery",
            "9_heart_failure_clinical_records", "10_wpbc"]

PIMA_COLS = {"preg": "Pregnancies", "plas": "Glucose", "pres": "BloodPressure", "skin": "SkinThickness",
             "insu": "Insulin", "mass": "BMI", "pedi": "DiabetesPedigreeFunction", "age": "Age"}
THORACIC_CODES = {
    "DGN": {"DGN1": 0, "DGN2": 1, "DGN3": 2, "DGN4": 3, "DGN5": 4, "DGN6": 5, "DGN8": 6},
    "PRE6": {"PRZ0": 0, "PRZ1": 1, "PRZ2": 2},
    "PRE14": {"OC11": 0, "OC12": 1, "OC13": 2, "OC14": 3},
}
THORACIC_TF = ["PRE7", "PRE8", "PRE9", "PRE10", "PRE11", "PRE17", "PRE19", "PRE25", "PRE30", "PRE32"]


def fetch_uci(i: int) -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo
    return fetch_ucirepo(id=i).data.original.copy()


def build(ds: str) -> pd.DataFrame:
    if ds == "2_breast_cancer_wdbc":
        # drop ID; malignant = 1; features renamed f1..f30
        r = fetch_uci(UCI_ID[ds])
        y = (r.pop("Diagnosis") == "M").astype(int)
        X = r.drop(columns=["ID"])
        X.columns = [f"f{k}" for k in range(1, X.shape[1] + 1)]

    elif ds == "10_wpbc":
        # drop ID and Time (follow-up); 4 rows with missing lymph-node status dropped; recurrence = 1
        r = fetch_uci(UCI_ID[ds]).drop(columns=["ID", "Time"]).dropna().reset_index(drop=True)
        y = (r.pop("Outcome") == "R").astype(int)
        X = r.astype(float)

    elif ds == "3_diabetes":
        # OpenML 37 (Pima); tested_positive = 1
        from sklearn.datasets import fetch_openml
        r = fetch_openml(data_id=37, as_frame=True, parser="auto").frame
        y = (r.pop("class").astype(str) == "tested_positive").astype(int)
        X = r.rename(columns=PIMA_COLS)
        for c in X.columns:  # integer-valued columns written as integers
            if np.allclose(X[c], np.round(X[c])):
                X[c] = X[c].round().astype(np.int64)

    elif ds == "6_ilpd":
        # 4 rows with missing A/G ratio dropped; Female = 0, Male = 1; non-patient (Selector 2) = 1
        r = fetch_uci(UCI_ID[ds]).dropna().reset_index(drop=True)
        y = (r.pop("Selector") == 2).astype(int)
        r["Gender"] = r["Gender"].map({"Female": 0, "Male": 1}).astype(np.int64)
        X = r.rename(columns={"A/G Ratio": "A_G_Ratio"})

    elif ds == "8_thoracic_surgery":
        # categorical strings coded one-to-one; death within 1 year (Risk1Yr T) = 1
        r = fetch_uci(UCI_ID[ds])
        y = (r.pop("Risk1Yr") == "T").astype(int)
        for c, m in THORACIC_CODES.items():
            r[c] = r[c].map(m).astype(np.int64)
        for c in THORACIC_TF:
            r[c] = r[c].map({"F": 0, "T": 1}).astype(np.int64)
        X = r

    elif ds == "9_heart_failure_clinical_records":
        # drop follow-up time (it encodes the outcome); death = 1
        r = fetch_uci(UCI_ID[ds]).drop(columns=["time"])
        y = r.pop("death_event").astype(int)
        X = r

    else:
        raise ValueError(ds)

    if X.isna().any().any():
        raise ValueError(f"{ds}: missing values remain")
    out = X.reset_index(drop=True).copy()
    out["target"] = y.reset_index(drop=True).astype(np.int64)
    return out


def sha256(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/public")
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(HERE, "data_sha256.json")) as fh:
        ref = json.load(fh)
    for ds in args.datasets:
        path = os.path.join(args.out, f"{ds}.csv")
        df = build(ds)
        df.to_csv(path, index=False)
        h = sha256(path)
        ok = h == ref[ds]
        status = "matches the paper" if ok else f"DIFFERS: {h}"
        print(f"{ds:34s} n={len(df):4d}  events={int(df.target.sum()):3d}  sha256 {status}")


if __name__ == "__main__":
    main()
