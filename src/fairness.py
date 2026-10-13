"""Fairness and generalisation checks.

1. Subgroup performance (sex, age group): uses OUT-OF-FOLD predictions for the WHOLE dataset
   (5-fold, the tuned hyper-parameters, the tuned threshold) so that every subgroup is big enough
   to say something. Each patient is scored by a model that never saw them.
2. Cross-source generalisation (heart only): the dataset is a mixture of four hospitals.
   a) train on Cleveland, test on Hungarian / Switzerland / VA Long Beach  (the requested test)
   b) leave-one-source-out for every source.
   We compare with the in-source score (5-fold CV inside the training hospital) and report the drop honestly.
   NOTE: Statlog is a subset of Cleveland, so it is not a separate source here.

Run:  python -m src.fairness
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from src.data import load_dataset, split_xy
from src.models import RESULTS, choose_threshold, cv_splitter, load_artifact, metrics_at

AGE_BINS = [0, 45, 55, 65, 200]
AGE_LABELS = ["<45", "45-54", "55-64", "65+"]


def group_metrics(y, p, thr, groups):
    """One row per subgroup: size, prevalence, recall, precision, F1, FNR, ROC-AUC."""
    rows = []
    order = list(groups.cat.categories) if hasattr(groups, "cat") else sorted(groups.dropna().unique())
    for g in order:
        m = (groups == g).values
        if m.sum() < 5:
            continue
        r = metrics_at(y[m], p[m], thr)
        rows.append({"group": g, "n": int(m.sum()), "prevalence": float(y[m].mean()),
                     "recall": r["recall"], "precision": r["precision"], "f1": r["f1"],
                     "false_negative_rate": 1 - r["recall"],
                     "roc_auc": r["roc_auc"] if len(set(y[m])) == 2 else np.nan})
    return pd.DataFrame(rows).set_index("group")


def subgroup_report(ds):
    df, cfg = load_dataset(ds)
    X, y = split_xy(df, cfg)
    art = load_artifact(ds)
    thr = art["threshold"]
    # out-of-fold calibrated probabilities for everyone
    cal = CalibratedClassifierCV(clone(art["pipeline"]), method=art["calibration_method"], cv=cv_splitter())
    p = cross_val_predict(cal, X, y, cv=cv_splitter(), method="predict_proba")[:, 1]
    y = y.values
    out = {"overall": group_metrics(y, p, thr, pd.Series("all", index=df.index))}
    out["age_group"] = group_metrics(y, p, thr, pd.cut(df[cfg["age_col"]], AGE_BINS, labels=AGE_LABELS))
    if cfg["sex_col"]:
        out["sex"] = group_metrics(y, p, thr, df[cfg["sex_col"]])
    tables = []
    for name, t in out.items():
        t = t.copy(); t.insert(0, "attribute", name); tables.append(t)
    full = pd.concat(tables).round(3)
    full.to_csv(RESULTS / f"{ds}_fairness_subgroups.csv")
    print(f"\n--- {ds}: subgroup performance at threshold {thr:.3f} (out-of-fold, whole dataset) ---")
    print(full.to_string())
    plot_subgroups(out, ds)
    return full


def plot_subgroups(out, ds):
    keys = [k for k in out if k != "overall"]
    fig, ax = plt.subplots(1, len(keys), figsize=(5.5 * len(keys), 4), squeeze=False)
    base = out["overall"].iloc[0]
    for a, k in zip(ax[0], keys):
        t = out[k]
        x = np.arange(len(t))
        a.bar(x - 0.2, t.recall, 0.4, label="recall", color="tab:red")
        a.bar(x + 0.2, t.precision, 0.4, label="precision", color="tab:blue")
        a.axhline(base.recall, color="tab:red", ls=":", lw=1)
        a.set_xticks(x); a.set_xticklabels([f"{i}\n(n={n})" for i, n in zip(t.index, t.n)])
        a.set(ylim=(0, 1.05), title=f"{ds}: by {k.replace('_', ' ')}"); a.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(RESULTS / f"{ds}_fairness_subgroups.png", dpi=130); plt.close()


# --------------------------------------------------------------------------- cross-source
def _fit_predict(art, X_tr, y_tr, X_te):
    """Fit the final model's recipe (same hyper-parameters) on one source, predict another.
    The threshold is re-derived on the TRAINING source's own out-of-fold predictions."""
    cal = CalibratedClassifierCV(clone(art["pipeline"]), method=art["calibration_method"], cv=cv_splitter())
    oof = cross_val_predict(cal, X_tr, y_tr, cv=cv_splitter(), method="predict_proba")[:, 1]
    thr = choose_threshold(y_tr.values, oof)
    cal.fit(X_tr, y_tr)
    return oof, thr, cal.predict_proba(X_te)[:, 1]


def cross_source_report():
    df, cfg = load_dataset("heart")
    X, y = split_xy(df, cfg)
    art = load_artifact("heart")
    src = df["Source"]
    rows = []

    def row(setting, train_src, test_src, y_te, p_te, thr, in_src=None):
        r = metrics_at(y_te.values, p_te, thr)
        rows.append({"setting": setting, "train": train_src, "test": test_src, "n_test": len(y_te),
                     "test_prevalence": y_te.mean(), "roc_auc": r["roc_auc"], "recall": r["recall"],
                     "precision": r["precision"], "accuracy": (r["tp"] + r["tn"]) / len(y_te),
                     "in_source_roc_auc": in_src[0], "in_source_recall": in_src[1],
                     "in_source_accuracy": in_src[2]})

    # a) train on Cleveland only, test on each other hospital
    tr = (src == "Cleveland").values
    oof, thr, _ = _fit_predict(art, X[tr], y[tr], X[~tr].head(5))
    inr = metrics_at(y[tr].values, oof, thr)
    in_src = (inr["roc_auc"], inr["recall"], (inr["tp"] + inr["tn"]) / tr.sum())
    for s in ["Hungarian", "Switzerland", "VA Long Beach"]:
        te = (src == s).values
        _, _, p = _fit_predict(art, X[tr], y[tr], X[te])
        row("train Cleveland", "Cleveland", s, y[te], p, thr, in_src)

    # b) leave-one-source-out
    for s in src.unique():
        te = (src == s).values
        oof, thr, p = _fit_predict(art, X[~te], y[~te], X[te])
        inr = metrics_at(y[~te].values, oof, thr)
        row("leave-one-source-out", "all others", s, y[te], p, thr,
            (inr["roc_auc"], inr["recall"], (inr["tp"] + inr["tn"]) / (~te).sum()))

    res = pd.DataFrame(rows)
    res["auc_drop"] = res.in_source_roc_auc - res.roc_auc
    res["accuracy_drop"] = res.in_source_accuracy - res.accuracy
    res["recall_drop"] = res.in_source_recall - res.recall
    res = res.round(3)
    res.to_csv(RESULTS / "heart_cross_source.csv", index=False)
    print("\n--- heart: cross-source generalisation (threshold re-derived on each training set) ---")
    print(res.to_string(index=False))
    plot_cross_source(res)
    return res


def plot_cross_source(res):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    a = res[res.setting == "train Cleveland"]
    x = np.arange(len(a))
    ax[0].bar(x - 0.2, a.in_source_accuracy, 0.4, label="in-source (Cleveland CV)", color="tab:green")
    ax[0].bar(x + 0.2, a.accuracy, 0.4, label="other hospital", color="tab:orange")
    ax[0].set_xticks(x); ax[0].set_xticklabels(a.test); ax[0].set_ylim(0, 1)
    ax[0].set(ylabel="Accuracy", title="Train on Cleveland -> test elsewhere"); ax[0].legend(fontsize=8)
    b = res[res.setting == "leave-one-source-out"]
    x = np.arange(len(b))
    ax[1].bar(x - 0.2, b.in_source_roc_auc, 0.4, label="in-source (training hospitals CV)", color="tab:green")
    ax[1].bar(x + 0.2, b.roc_auc, 0.4, label="held-out hospital", color="tab:orange")
    ax[1].set_xticks(x); ax[1].set_xticklabels(b.test, rotation=15); ax[1].set_ylim(0, 1)
    ax[1].set(ylabel="ROC-AUC", title="Leave-one-source-out"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(RESULTS / "heart_cross_source.png", dpi=130); plt.close()


if __name__ == "__main__":
    subgroup_report("heart")
    subgroup_report("diabetes")
    cross_source_report()
