"""Stage 2 - leakage-free pipelines, model comparison, calibration, cost-sensitive threshold.

Run:  python -m src.models --dataset heart      (or diabetes / both)

Protocol (kept deliberately simple):
  1. One stratified 80/20 hold-out split (fixed seed). The 20% test part is used ONLY for reporting.
  2. On the 80% train part, every model is tuned with GridSearchCV (stratified 5-fold).
     The imputer/scaler/encoder are INSIDE the sklearn Pipeline, so they are re-fitted on the
     training folds only in every CV split -> no information leaks from validation folds.
  3. The final model = best CV ROC-AUC. Its probabilities are calibrated (Platt or isotonic,
     whichever gives the lower out-of-fold Brier score).
  4. The decision threshold is chosen on OUT-OF-FOLD predictions of the training data
     (>= 90% recall, then best precision), never on the test set.
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, log_loss, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.data import SEED, ROOT, load_dataset, split_xy, train_test_split_df

RESULTS = ROOT / "results"
MODEL_DIR = RESULTS / "models"
MIN_RECALL = 0.90     # clinical requirement: catch at least 90% of true patients
FN_COST, FP_COST = 5, 1  # illustrative: a missed patient is assumed 5x worse than a false alarm


def cv_splitter():
    """The single CV splitter used everywhere (shuffled + seeded + stratified)."""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


# --------------------------------------------------------------------------- pipeline
def build_pipeline(estimator, cfg):
    """preprocessing + model in ONE object so CV re-fits preprocessing on each training fold."""
    num = Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("sc", StandardScaler())])
    transformers = [("num", num, cfg["numeric"])]
    if cfg["categorical"]:
        cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
        transformers.append(("cat", cat, cfg["categorical"]))
    pre = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("pre", pre), ("clf", estimator)])


# model name -> (estimator, hyper-parameter grid). Grids are small on purpose (tiny datasets).
def model_zoo():
    return {
        "Logistic Regression": (LogisticRegression(max_iter=2000, random_state=SEED),
                                {"clf__C": [0.01, 0.1, 1, 10]}),
        "Random Forest": (RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1),
                          {"clf__max_depth": [4, 8, None], "clf__min_samples_leaf": [1, 3, 5]}),
        "XGBoost": (XGBClassifier(eval_metric="logloss", subsample=0.8, random_state=SEED,
                                  n_jobs=1, verbosity=0),
                    {"clf__n_estimators": [100, 300], "clf__max_depth": [2, 3, 4],
                     "clf__learning_rate": [0.03, 0.1]}),
        "SVM (RBF)": (SVC(probability=True, random_state=SEED),
                      {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01, 0.1]}),
        "MLP": (MLPClassifier(max_iter=2000, random_state=SEED),
                {"clf__hidden_layer_sizes": [(16,), (32, 16)], "clf__alpha": [1e-3, 1e-2, 1e-1]}),
    }


# --------------------------------------------------------------------------- metrics
def metrics_at(y, p, thr=0.5):
    """All reported metrics for probabilities p at decision threshold thr."""
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "brier": brier_score_loss(y, p), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def choose_threshold(y, p, min_recall=MIN_RECALL):
    """Highest-precision threshold that still keeps recall >= min_recall.

    Why: a false negative (sending a sick patient home) is far costlier than a false positive
    (an extra check-up), so we first FIX the recall we need and only then maximise precision.
    Ties in precision are broken towards the higher threshold (fewer needless alarms).
    """
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]            # these arrays are one longer than thr
    ok = rec >= min_recall
    if not ok.any():                           # cannot reach the target: fall back to max recall
        return float(thr[np.argmax(rec)])
    best_prec = prec[ok].max()
    return float(thr[ok & (prec >= best_prec - 1e-12)].max())


def expected_cost(y, p, thr):
    pred = (p >= thr).astype(int)
    fn = ((pred == 0) & (y == 1)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    return (FN_COST * fn + FP_COST * fp) / len(y)


# --------------------------------------------------------------------------- tuning
SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "precision": "precision",
           "recall": "recall", "f1": "f1"}


def tune_all(X, y, cfg):
    """GridSearchCV for every model. Returns {name: fitted GridSearchCV}."""
    fitted = {}
    for name, (est, grid) in model_zoo().items():
        gs = GridSearchCV(build_pipeline(est, cfg), grid, scoring=SCORING, refit="roc_auc",
                          cv=cv_splitter(), n_jobs=-1)
        gs.fit(X, y)
        fitted[name] = gs
        print(f"  tuned {name:<20s} CV ROC-AUC = {gs.best_score_:.3f}  {gs.best_params_}")
    return fitted


def cv_table(fitted):
    """Mean +- std of every CV metric at the best hyper-parameters of each model."""
    rows = []
    for name, gs in fitted.items():
        i, r = gs.best_index_, gs.cv_results_
        row = {"model": name}
        for m in SCORING:
            row[m] = f"{r[f'mean_test_{m}'][i]:.3f} ± {r[f'std_test_{m}'][i]:.3f}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def pick_final(fitted):
    """Best CV ROC-AUC. SVM/MLP only have approximate (Kernel) SHAP, so if one of them wins by
    less than 0.01 AUC we prefer the best of LR / RF / XGBoost (exact, fast explanations)."""
    score = {n: gs.best_score_ for n, gs in fitted.items()}
    best = max(score, key=score.get)
    if best in ("SVM (RBF)", "MLP"):
        exact = {n: s for n, s in score.items() if n not in ("SVM (RBF)", "MLP")}
        alt = max(exact, key=exact.get)
        if score[best] - exact[alt] < 0.01:
            print(f"  {best} won by <0.01 AUC; using {alt} for exact SHAP explanations")
            return alt
    return best


# --------------------------------------------------------------------------- calibration
def choose_calibration(pipe, X, y):
    """Compare none / Platt (sigmoid) / isotonic by out-of-fold Brier score on the TRAIN data."""
    oof, brier = {}, {}
    oof["none"] = cross_val_predict(clone(pipe), X, y, cv=cv_splitter(), method="predict_proba")[:, 1]
    for m in ("sigmoid", "isotonic"):
        cal = CalibratedClassifierCV(clone(pipe), method=m, cv=cv_splitter())
        oof[m] = cross_val_predict(cal, X, y, cv=cv_splitter(), method="predict_proba")[:, 1]
    for m, p in oof.items():
        brier[m] = brier_score_loss(y, p)
    best = min(("sigmoid", "isotonic"), key=brier.get)
    print("  OOF Brier:", {k: round(v, 4) for k, v in brier.items()}, "->", best)
    return best, oof[best], brier


# --------------------------------------------------------------------------- plots
def plot_comparison(fitted, X_te, y_te, ds, cvt):
    # ROC and PR curves on the hold-out test set
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, gs in fitted.items():
        p = gs.best_estimator_.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, p)
        pr, rc, _ = precision_recall_curve(y_te, p)
        ax[0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_te, p):.2f})")
        ax[1].plot(rc, pr, label=f"{name} ({average_precision_score(y_te, p):.2f})")
    ax[0].plot([0, 1], [0, 1], "k:", lw=1)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (test set)")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall (test set)")
    for a in ax:
        a.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(RESULTS / f"{ds}_roc_pr.png", dpi=130); plt.close()


def plot_calibration(fitted, X_te, y_te, final_name, p_uncal, p_cal, ds):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, gs in fitted.items():
        p = gs.best_estimator_.predict_proba(X_te)[:, 1]
        frac, mean = calibration_curve(y_te, p, n_bins=6, strategy="quantile")
        ax[0].plot(mean, frac, "o-", label=name, ms=4)
    ax[0].plot([0, 1], [0, 1], "k:", label="perfect")
    ax[0].set(xlabel="Mean predicted probability", ylabel="Observed fraction positive",
              title="Calibration of all models (test set)")
    for p, lab in ((p_uncal, f"{final_name} uncalibrated"), (p_cal, f"{final_name} calibrated")):
        frac, mean = calibration_curve(y_te, p, n_bins=6, strategy="quantile")
        ax[1].plot(mean, frac, "o-", label=f"{lab} (Brier {brier_score_loss(y_te, p):.3f})")
    ax[1].plot([0, 1], [0, 1], "k:", label="perfect")
    ax[1].set(xlabel="Mean predicted probability", ylabel="Observed fraction positive",
              title="Final model: before vs after calibration")
    for a in ax:
        a.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(RESULTS / f"{ds}_calibration.png", dpi=130); plt.close()


def plot_threshold(y, p, thr, ds):
    """Left: precision & recall vs threshold. Right: expected cost vs threshold (why 0.5 is wrong)."""
    grid = np.linspace(0.02, 0.98, 97)
    prec = [precision_score(y, p >= t, zero_division=0) for t in grid]
    rec = [recall_score(y, p >= t, zero_division=0) for t in grid]
    cost = [expected_cost(y, p, t) for t in grid]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(grid, prec, label="precision"); ax[0].plot(grid, rec, label="recall")
    ax[0].axhline(MIN_RECALL, color="gray", ls=":", label=f"recall target {MIN_RECALL:.0%}")
    ax[0].axvline(thr, color="red", ls="--", label=f"chosen threshold {thr:.2f}")
    ax[0].axvline(0.5, color="black", ls=":", label="default 0.5")
    ax[0].set(xlabel="Threshold", title="Precision / recall vs threshold (train OOF)"); ax[0].legend(fontsize=8)
    ax[1].plot(grid, cost, color="purple")
    ax[1].axvline(thr, color="red", ls="--"); ax[1].axvline(0.5, color="black", ls=":")
    ax[1].set(xlabel="Threshold", ylabel="Average cost per patient",
              title=f"Expected cost if a missed patient costs {FN_COST}x a false alarm")
    plt.tight_layout(); plt.savefig(RESULTS / f"{ds}_threshold.png", dpi=130); plt.close()


# --------------------------------------------------------------------------- main
def slug(name):
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")


def make_deployable(name, gs, X_tr, y_tr, X_te, y_te, ds, is_final):
    """Calibrate one tuned model, choose its cost-sensitive threshold, fit the final objects."""
    best_pipe = gs.best_estimator_
    method, oof_cal, briers = choose_calibration(best_pipe, X_tr, y_tr)
    # threshold from TRAIN out-of-fold calibrated probabilities (test set never used)
    thr = choose_threshold(y_tr.values, oof_cal)
    calibrated = CalibratedClassifierCV(clone(best_pipe), method=method, cv=cv_splitter()).fit(X_tr, y_tr)
    pipeline = clone(best_pipe).fit(X_tr, y_tr)   # un-calibrated twin, used for explanations
    p_cal = calibrated.predict_proba(X_te)[:, 1]
    p_raw = pipeline.predict_proba(X_te)[:, 1]
    print(f"  {name:<20s} calibration={method:<8s} threshold={thr:.3f}")
    if is_final:
        plot_threshold(y_tr.values, oof_cal, thr, ds)
    joblib.dump({"dataset": ds, "model_name": name, "calibrated": calibrated, "pipeline": pipeline,
                 "threshold": thr, "calibration_method": method, "best_params": gs.best_params_},
                MODEL_DIR / f"{ds}__{slug(name)}.joblib", compress=3)
    return {"calibration_method": method, "oof_brier": briers, "threshold": thr,
            "test_at_0.5": {k: float(v) for k, v in metrics_at(y_te, p_cal, 0.5).items()},
            "test_at_tuned_threshold": {k: float(v) for k, v in metrics_at(y_te, p_cal, thr).items()},
            "best_params": gs.best_params_}, p_raw, p_cal


def run(ds):
    print(f"\n=== {ds} ===")
    RESULTS.mkdir(exist_ok=True); MODEL_DIR.mkdir(exist_ok=True)
    df, cfg = load_dataset(ds)
    train_df, test_df = train_test_split_df(df, cfg)
    X_tr, y_tr = split_xy(train_df, cfg)
    X_te, y_te = split_xy(test_df, cfg)
    print(f"  train {X_tr.shape}, test {X_te.shape}")

    # 1-2. tune all five models, compare
    fitted = tune_all(X_tr, y_tr, cfg)
    cvt = cv_table(fitted)
    test_rows = {n: metrics_at(y_te, gs.best_estimator_.predict_proba(X_te)[:, 1])
                 for n, gs in fitted.items()}
    test_t = pd.DataFrame(test_rows).T[["roc_auc", "pr_auc", "precision", "recall", "f1", "brier"]].round(3)
    cvt.to_csv(RESULTS / f"{ds}_cv_table.csv"); test_t.to_csv(RESULTS / f"{ds}_test_table.csv")
    print("\nCV (5-fold, train part):\n", cvt.to_string(), "\n\nTest (hold-out, threshold 0.5):\n", test_t.to_string())
    plot_comparison(fitted, X_te, y_te, ds, cvt)

    # 3-5. calibration + cost-sensitive threshold + deployable object for EVERY model
    final_name = pick_final(fitted)
    print("  final model (best CV ROC-AUC rule):", final_name)
    summary = {"dataset": ds, "final_model": final_name, "min_recall_target": MIN_RECALL, "models": {}}
    for name, gs in fitted.items():
        info, p_raw, p_cal = make_deployable(name, gs, X_tr, y_tr, X_te, y_te, ds, name == final_name)
        summary["models"][name] = info
        if name == final_name:
            plot_calibration(fitted, X_te, y_te, name, p_raw, p_cal, ds)
    json.dump(summary, open(RESULTS / f"{ds}_final_metrics.json", "w"), indent=2, default=str)

    # table: every model at its own 90%-recall threshold, on the hold-out test set
    thr_t = pd.DataFrame({n: {**{k: v for k, v in i["test_at_tuned_threshold"].items()
                                 if k in ("precision", "recall", "f1", "fn", "fp")},
                              "threshold": i["threshold"]} for n, i in summary["models"].items()}).T.round(3)
    thr_t.to_csv(RESULTS / f"{ds}_test_at_tuned_threshold.csv")
    print("\nTest at the tuned (>=90% train-OOF recall) threshold:\n", thr_t.to_string())


def load_artifact(ds, model=None):
    """Load a saved deployable. model=None -> the final (best CV) model."""
    if model is None:
        model = json.load(open(RESULTS / f"{ds}_final_metrics.json"))["final_model"]
    return joblib.load(MODEL_DIR / f"{ds}__{slug(model)}.joblib")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="both", choices=["heart", "diabetes", "both"])
    a = ap.parse_args()
    for d in (["heart", "diabetes"] if a.dataset == "both" else [a.dataset]):
        run(d)
