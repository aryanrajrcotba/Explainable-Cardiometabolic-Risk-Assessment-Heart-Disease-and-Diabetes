"""Actionable counterfactuals with DiCE: "what would lower this patient's risk?"

Rules that make the suggestions *actionable* rather than just mathematically valid:
  * Only features listed in CONFIG[ds]["modifiable"] may change (BMI, glucose, cholesterol,
    blood pressure, fasting blood sugar flag, max heart rate). Age, sex and everything else are frozen
    (DiCE's `features_to_vary`).
  * Risk factors may only move DOWN towards a healthy bound, protective factors only UP, and by a
    realistic amount (see permitted_range below).
  * "Lower risk" means: below the tuned screening threshold of the calibrated model, not just 0.5.
  * After rounding the values to realistic precision we RE-SCORE every counterfactual with the real
    model and keep it only if it still crosses the threshold.
"""
import warnings

import dice_ml
import numpy as np
import pandas as pd

from src.data import SEED, feature_cols, fill_missing

ROUND = {"BMI": 1, "Oldpeak": 1}        # decimals; every other modifiable feature is rounded to an integer
MAX_UP = 0.15               # a protective factor (e.g. MaxHR) may rise by at most 15%


class ThresholdModel:
    """Wraps the calibrated model so that probability 0.5 == our tuned threshold.

    DiCE flips the predicted class at 0.5. By stretching probabilities piecewise-linearly
    (threshold -> 0.5) DiCE's notion of "opposite class" becomes exactly "below our threshold".
    """

    def __init__(self, model, threshold, flag_cols=()):
        self.model, self.t, self.flag_cols = model, threshold, flag_cols

    def predict_proba(self, X):
        X = X.copy()
        for c in self.flag_cols:                 # DiCE hands over 0/1 flags as strings: back to numbers
            X[c] = X[c].astype(float)
        p = self.model.predict_proba(X)[:, 1]
        t = self.t
        q = np.where(p <= t, 0.5 * p / t, 0.5 + 0.5 * (p - t) / (1 - t))
        return np.column_stack([1 - q, q])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def permitted_ranges(cfg, row, train_X):
    """Per-patient allowed range of every modifiable feature."""
    rng = {}
    for f, (direction, bound) in cfg["modifiable"].items():
        q = float(row[f])
        if f in cfg["binary_numeric"]:                       # 0/1 flag: allowed values as strings (DiCE)
            rng[f] = ["0.0", "1.0"] if (direction == "down" and q == 1) else [str(float(q))]
        elif direction == "down":
            rng[f] = [min(bound, q), q]                      # never above today, never below healthy
        else:
            hi = min(q * (1 + MAX_UP), float(train_X[f].max()))
            if f == "MaxHR":                                 # never above the age-predicted maximum
                hi = min(hi, max(q, 220 - float(row["Age"])))
            rng[f] = [q, hi]
    return rng


def _round(f, v):
    return round(float(v), ROUND.get(f, 0))


def suggest(cfg, art, X_train, y_train, patient, total=4, seed=SEED):
    """Counterfactuals for one patient (1-row DataFrame of raw features).

    Returns (list_of_options, message). Each option is
        {"changes": {feature: (current, suggested)}, "new_risk": float}
    sorted by fewest changes first. The list is empty when nothing is needed / possible.
    """
    model, thr = art["calibrated"], art["threshold"]
    feats = feature_cols(cfg)
    patient = fill_missing(patient[feats], X_train, cfg).reset_index(drop=True)   # DiCE cannot take NaN
    risk = float(model.predict_proba(patient)[0, 1])
    if risk < thr:
        return [], f"Estimated risk ({risk:.0%}) is already below the screening threshold ({thr:.0%})."

    # DiCE learns feature ranges from the training data (missing values filled the simple way).
    train = fill_missing(X_train[feats], X_train, cfg).copy()
    train[cfg["target"]] = y_train.values
    flags = cfg["binary_numeric"]
    for c in flags:                                   # DiCE treats categorical values as strings
        train[c] = train[c].astype(float).astype(str)
    continuous = [c for c in cfg["numeric"] if c not in cfg["binary_numeric"]]
    data = dice_ml.Data(dataframe=train, continuous_features=continuous, outcome_name=cfg["target"])
    dice_model = dice_ml.Model(model=ThresholdModel(model, thr, flags), backend="sklearn")
    ranges = permitted_ranges(cfg, patient.iloc[0], X_train)
    # a feature whose allowed range has a single value cannot change: freeze it
    ranges = {f: r for f, r in ranges.items() if len(set(r)) > 1}
    vary = list(ranges)
    if not vary:
        return [], "None of the modifiable factors can move any further in a healthy direction."

    query = patient.copy()
    for c in flags:
        query[c] = query[c].astype(float).astype(str)
    options = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for method in ("random", "genetic"):          # random is fast; genetic is the fallback
            try:
                explainer = dice_ml.Dice(data, dice_model, method=method)
                extra = {"random_seed": seed} if method == "random" else {}
                res = explainer.generate_counterfactuals(
                    query, total_CFs=total, desired_class="opposite", features_to_vary=vary,
                    permitted_range=ranges, verbose=False, **extra)
                cfs = res.cf_examples_list[0].final_cfs_df
            except Exception:                         # DiCE raises when it finds nothing
                cfs = None
            if cfs is not None and len(cfs):
                options = _clean(cfs, patient, cfg, model, thr)
            if options:
                break
    if not options:
        return [], ("No realistic change to the modifiable factors was found that brings the risk "
                    "below the threshold. Discuss other options with a clinician.")
    return options, f"Found {len(options)} option(s) that bring the estimated risk below {thr:.0%}."


def _clean(cfs, patient, cfg, model, thr):
    """Round, drop unchanged features, re-score with the real model, keep only valid options."""
    out, seen = [], set()
    for _, r in cfs.iterrows():
        cand = patient.copy()
        for f in cfg["modifiable"]:
            if f in r.index:
                cand.loc[0, f] = _round(f, r[f]) if f not in cfg["binary_numeric"] else float(r[f])
        new_risk = float(model.predict_proba(cand)[0, 1])
        changes = {f: (float(patient.loc[0, f]), float(cand.loc[0, f])) for f in cfg["modifiable"]
                   if abs(float(patient.loc[0, f]) - float(cand.loc[0, f])) > 1e-9}
        key = tuple(sorted(changes.items()))
        if changes and new_risk < thr and key not in seen:
            seen.add(key)
            out.append({"changes": changes, "new_risk": new_risk})
    return sorted(out, key=lambda o: (len(o["changes"]), o["new_risk"]))[:3]


if __name__ == "__main__":
    # quick demo: counterfactuals for the first few flagged patients of the test set
    import sys
    from src.data import load_dataset, split_xy, train_test_split_df
    from src.models import load_artifact
    ds = sys.argv[1] if len(sys.argv) > 1 else "heart"
    df, cfg = load_dataset(ds)
    tr, te = train_test_split_df(df, cfg)
    X_tr, y_tr = split_xy(tr, cfg)
    X_te, _ = split_xy(te, cfg)
    art = load_artifact(ds)
    flagged = X_te[art["calibrated"].predict_proba(X_te)[:, 1] >= art["threshold"]]
    for i in range(3):
        opts, msg = suggest(cfg, art, X_tr, y_tr, flagged.iloc[[i]])
        print(msg)
        for o in opts:
            print("  ", {f: f"{a:g} -> {b:g}" for f, (a, b) in o["changes"].items()}, f"-> risk {o['new_risk']:.0%}")
