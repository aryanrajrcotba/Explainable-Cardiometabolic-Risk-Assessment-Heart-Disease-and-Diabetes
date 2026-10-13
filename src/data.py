"""Data loading and dataset configuration.

Two public datasets are used:
  * Heart    - UCI Heart Disease (Cleveland + Hungarian + Switzerland + VA Long Beach).
               We rebuild the well-known "918-row Kaggle" table from the four raw UCI files
               so that we can KEEP the source (hospital) as a column, which the Kaggle CSV drops.
  * Diabetes - Pima Indians Diabetes (768 rows). Zeros in Glucose, BloodPressure,
               SkinThickness, Insulin and BMI are physiologically impossible, so they are
               "hidden missing values" and are converted to NaN.

Everything else in the project asks this module for data + a config dict, so the
dataset-specific knowledge (column types, which features are modifiable, UI ranges)
lives in exactly one place.
"""
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42  # one global seed, used everywhere for reproducibility

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

UCI_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
UCI_FILES = {  # file name -> source label kept in the "Source" column
    "processed.cleveland.data": "Cleveland",
    "processed.hungarian.data": "Hungarian",
    "processed.switzerland.data": "Switzerland",
    "processed.va.data": "VA Long Beach",
}
PIMA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"


# --------------------------------------------------------------------------- download
def download_raw():
    """Download raw files into data/raw if they are not already there."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for fname in UCI_FILES:
        if not (RAW_DIR / fname).exists():
            urllib.request.urlretrieve(UCI_BASE + fname, RAW_DIR / fname)
    if not (RAW_DIR / "pima.csv").exists():
        urllib.request.urlretrieve(PIMA_URL, RAW_DIR / "pima.csv")


# --------------------------------------------------------------------------- heart
def load_heart():
    """Return the combined heart-disease table with readable column names."""
    download_raw()
    uci_cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
    parts = []
    for fname, source in UCI_FILES.items():
        part = pd.read_csv(RAW_DIR / fname, header=None, names=uci_cols, na_values="?")
        part["Source"] = source
        parts.append(part)
    raw = pd.concat(parts, ignore_index=True)

    # Re-code the numeric UCI codes into the readable Kaggle-style categories.
    df = pd.DataFrame({
        "Age": raw["age"],
        "Sex": raw["sex"].map({1: "M", 0: "F"}),
        "ChestPainType": raw["cp"].map({1: "TA", 2: "ATA", 3: "NAP", 4: "ASY"}),
        "RestingBP": raw["trestbps"],
        "Cholesterol": raw["chol"],
        "FastingBS": raw["fbs"],
        "RestingECG": raw["restecg"].map({0: "Normal", 1: "ST", 2: "LVH"}),
        "MaxHR": raw["thalach"],
        "ExerciseAngina": raw["exang"].map({1: "Y", 0: "N"}),
        "Oldpeak": raw["oldpeak"],
        "ST_Slope": raw["slope"].map({1: "Up", 2: "Flat", 3: "Down"}),
        # UCI "num" is 0 (healthy) .. 4 (severe). Standard practice: any disease = 1.
        "HeartDisease": (raw["num"] > 0).astype(int),
        "Source": raw["Source"],
    })
    # Cholesterol = 0 and RestingBP = 0 are impossible values (hidden missing, same problem as Pima).
    df.loc[df["Cholesterol"] == 0, "Cholesterol"] = np.nan
    df.loc[df["RestingBP"] == 0, "RestingBP"] = np.nan
    # Two rows are exact duplicates; dropping them gives the 918 rows of the Kaggle version.
    return df.drop_duplicates().reset_index(drop=True)


# --------------------------------------------------------------------------- diabetes
def load_diabetes():
    """Return the Pima table; hidden zeros are turned into NaN."""
    download_raw()
    cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
            "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    df = pd.read_csv(RAW_DIR / "pima.csv", header=None, names=cols)
    for c in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        df[c] = df[c].replace(0, np.nan)
    return df


# --------------------------------------------------------------------------- config
# "modifiable": feature -> (direction we allow it to move, healthy bound).
#   'down' = risk factor we want lowered, never below the healthy bound.
#   'up'   = protective factor we want raised.
# Anything NOT in "modifiable" (Age, Sex, ...) is frozen in counterfactuals.
# "ui": feature -> (min, max, step) for the Streamlit sidebar.
CONFIG = {
    "heart": {
        "title": "Heart disease",
        "target": "HeartDisease",
        "numeric": ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"],
        "categorical": ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"],
        "binary_numeric": ["FastingBS"],  # numeric 0/1 flag; treated as categorical by DiCE
        "modifiable": {
            "RestingBP": ("down", 110),
            "Cholesterol": ("down", 170),
            "FastingBS": ("down", 0),
            "MaxHR": ("up", None),
            "Oldpeak": ("down", 0.0),   # exercise-test ST depression; improves with fitness / treatment
        },
        "ui": {"Age": (20, 90, 1), "RestingBP": (80, 200, 1), "Cholesterol": (100, 600, 1),
               "MaxHR": (60, 210, 1), "Oldpeak": (-1.0, 6.5, 0.1)},
        "unknown_ok": ["Cholesterol"],
        "age_col": "Age", "sex_col": "Sex",
    },
    "diabetes": {
        "title": "Diabetes",
        "target": "Outcome",
        "numeric": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
                    "BMI", "DiabetesPedigreeFunction", "Age"],
        "categorical": [],
        "binary_numeric": [],
        "modifiable": {
            "BMI": ("down", 22.0),
            "Glucose": ("down", 90),
            "BloodPressure": ("down", 70),
        },
        "ui": {"Pregnancies": (0, 17, 1), "Glucose": (40, 250, 1), "BloodPressure": (30, 130, 1),
               "SkinThickness": (5, 100, 1), "Insulin": (10, 850, 1), "BMI": (15.0, 70.0, 0.1),
               "DiabetesPedigreeFunction": (0.05, 2.5, 0.01), "Age": (18, 90, 1)},
        "unknown_ok": ["SkinThickness", "Insulin"],
        "age_col": "Age", "sex_col": None,  # Pima is female-only, so no sex column
    },
}


def load_dataset(name):
    """name in {'heart','diabetes'} -> (DataFrame, config)."""
    df = load_heart() if name == "heart" else load_diabetes()
    return df, CONFIG[name]


def feature_cols(cfg):
    """Model input columns, in a fixed order (numeric first, then categorical)."""
    return cfg["numeric"] + cfg["categorical"]


def split_xy(df, cfg):
    """Split a table into features X (model inputs only) and label y."""
    return df[feature_cols(cfg)].copy(), df[cfg["target"]].astype(int).copy()


def train_test_split_df(df, cfg, test_size=0.2):
    """One fixed stratified hold-out split. The test part is touched ONLY for final reporting."""
    return train_test_split(df, test_size=test_size, stratify=df[cfg["target"]],
                            random_state=SEED)


def fill_missing(row_df, ref_df, cfg):
    """Fill NaNs in a few rows with the reference median/mode (needed by DiCE, which hates NaN)."""
    out = row_df.copy()
    for c in feature_cols(cfg):
        if out[c].isna().any():
            fill = ref_df[c].median() if c in cfg["numeric"] else ref_df[c].mode().iloc[0]
            out[c] = out[c].fillna(fill)
    return out


if __name__ == "__main__":
    for ds in ("heart", "diabetes"):
        d, c = load_dataset(ds)
        print(ds, d.shape, "positive rate = %.3f" % d[c["target"]].mean())
        print(d.isna().sum()[lambda s: s > 0].to_dict())
