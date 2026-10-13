"""Explanation Reliability Score (ERS) + the research experiment (E).

IDEA
  An explanation is only worth trusting if (1) independent explanation methods agree on
  which features matter for THIS patient, and (2) the explanation would not change if we
  re-ran the stochastic method or re-trained the model on slightly different data.
  The ERS turns both checks into one 0-100 number per prediction.

FORMULA (every weight is a named constant below; nothing is tuned on the data)
  For two importance vectors a, b (absolute attributions over the p original features):
      pair_score(a, b) = 0.5 * max(0, Spearman(|a|, |b|))  +  0.5 * |top3(a) ∩ top3(b)| / 3
  consistency(list of vectors) = mean of pair_score over all pairs in the list.

  Agreement      = consistency([SHAP, LIME(mean of runs), local permutation])
  LIME stability = consistency([LIME run 1, ..., LIME run R])           (R = 5 reseeded runs)
  Model stability= consistency([SHAP from B bootstrap re-fitted models])  (B = 5)
  Stability      = 0.5 * LIME stability + 0.5 * Model stability

  ERS = 100 * (0.5 * Agreement + 0.5 * Stability)

  Level (the badge): relative to what is typical for this model. The cut-offs are the 33rd and
  67th percentile of the ERS of N_REFERENCE random TRAINING patients (test set untouched):
  Low = least reliable third, High = most reliable third. Why relative: with fixed cut-offs
  (70/40) ~99% of patients were "High", so the badge carried no information. The 0-100 ERS
  itself stays absolute. If no reference file exists, the fixed cut-offs below are the fallback.

Run the experiment:  python -m src.ers --dataset heart
"""
import argparse
import itertools
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from src import explain as ex
from src.data import SEED, ROOT, load_dataset, split_xy, train_test_split_df
from src.models import MODEL_DIR, RESULTS, load_artifact, slug

# ----------------------------------------------------------------- documented constants
TOP_K = 3
W_RANK, W_TOP = 0.5, 0.5            # inside pair_score
W_AGREE, W_STABLE = 0.5, 0.5        # inside ERS
N_LIME_RUNS = 5                     # R
N_BOOTSTRAP = 5                     # B
HIGH_CUT, LOW_CUT = 70, 40          # fallback fixed cut-offs
N_REFERENCE = 150                   # training patients used to set the relative cut-offs
METHODS = ("SHAP", "LIME", "Permutation")


# --------------------------------------------------------------------------- the score
def top_k_set(v, k=TOP_K):
    """Indices of the k largest |attributions| (stable tie-break by position)."""
    return set(np.argsort(-np.abs(v), kind="stable")[:k])


def pair_score(a, b):
    """Similarity in [0,1] of two attribution vectors: rank correlation + top-3 overlap."""
    rho = spearmanr(np.abs(a), np.abs(b)).correlation
    rho = 0.0 if np.isnan(rho) else max(0.0, float(rho))    # constant vectors -> 0; negatives -> 0
    overlap = len(top_k_set(a) & top_k_set(b)) / TOP_K
    return W_RANK * rho + W_TOP * overlap


def consistency(vectors):
    """Mean pair_score over all pairs. One vector alone is trivially consistent (1.0)."""
    pairs = list(itertools.combinations(vectors, 2))
    return 1.0 if not pairs else float(np.mean([pair_score(a, b) for a, b in pairs]))


def reliability_level(ers, cuts=None):
    """cuts = (low_cut, high_cut); None -> fixed fallback cut-offs."""
    low, high = cuts if cuts else (LOW_CUT, HIGH_CUT)
    return "High" if ers >= high else ("Medium" if ers >= low else "Low")


def compute_ers(shap_v, lime_runs, perm_v, boot_shaps):
    """ERS for ONE patient. All inputs are 1-D arrays over the original features
    (lime_runs and boot_shaps are lists of such arrays). Returns a dict of all components."""
    lime_mean = np.mean(lime_runs, axis=0)
    agreement = consistency([shap_v, lime_mean, perm_v])
    lime_stab = consistency(lime_runs)
    boot_stab = consistency(boot_shaps)
    stability = 0.5 * lime_stab + 0.5 * boot_stab
    ers = 100.0 * (W_AGREE * agreement + W_STABLE * stability)
    out = {"ERS": ers, "level": reliability_level(ers), "agreement": agreement,
           "lime_stability": lime_stab, "model_stability": boot_stab, "stability": stability}
    vecs = dict(zip(METHODS, [shap_v, lime_mean, perm_v]))
    for m1, m2 in itertools.combinations(METHODS, 2):          # pairwise detail for display
        r = spearmanr(np.abs(vecs[m1]), np.abs(vecs[m2])).correlation
        out[f"rho_{m1}_{m2}"] = 0.0 if np.isnan(r) else float(r)
        out[f"top3_{m1}_{m2}"] = len(top_k_set(vecs[m1]) & top_k_set(vecs[m2])) / TOP_K
    out["all3_top3_overlap"] = len(top_k_set(shap_v) & top_k_set(lime_mean) & top_k_set(perm_v))
    out["same_top1"] = int(len({int(np.argmax(np.abs(v))) for v in vecs.values()}) == 1)
    return out


# --------------------------------------------------------------------------- bootstrap models
def make_bootstrap_pipes(pipe, X_train, y_train, n=N_BOOTSTRAP, seed=SEED):
    """n copies of the model (same hyper-parameters) re-fitted on bootstrap resamples of the train set."""
    rng = np.random.RandomState(seed)
    pipes = []
    for _ in range(n):
        idx = rng.choice(len(X_train), len(X_train), replace=True)
        pipes.append(clone(pipe).fit(X_train.iloc[idx], y_train.iloc[idx]))
    return pipes


def get_bootstrap_pipes(ds, model, pipe, X_train, y_train):
    """Cached on disk so the Streamlit app does not need to refit them every start-up."""
    path = MODEL_DIR / f"{ds}__{slug(model)}__bootstrap.joblib"
    if path.exists():
        return joblib.load(path)
    pipes = make_bootstrap_pipes(pipe, X_train, y_train)
    joblib.dump(pipes, path, compress=3)
    return pipes


# --------------------------------------------------------------------------- batch computation
def explain_and_score(pipe, boot_pipes, X_train, X_eval, n_lime=N_LIME_RUNS, n_perm=50,
                      seed=SEED, bg_size=200, cuts=None):
    """Compute SHAP, LIME, permutation attributions and the ERS for every row of X_eval.

    Returns (table, attrs). table: DataFrame with the ERS and its components, one row per patient.
    attrs: dict of arrays [n_patients, n_features] for 'SHAP', 'LIME', 'Permutation'
           (plus 'base' and 'unit' for the SHAP waterfall).
    """
    bg = X_train.sample(min(bg_size, len(X_train)), random_state=seed)
    shap_v, base, unit = ex.shap_values(pipe, bg, X_eval)
    boot_v = [ex.shap_values(bp, bg, X_eval)[0] for bp in boot_pipes]
    lime_ex = ex.make_lime(pipe, X_train)

    rows, lime_all, perm_all = [], [], []
    for i in range(len(X_eval)):
        x_row = X_eval.iloc[[i]]
        runs = [ex.lime_values(pipe, lime_ex, x_row, seed=seed + r, num_samples=1000) for r in range(n_lime)]
        perm = ex.permutation_values(pipe, bg, x_row, n_draws=n_perm, seed=seed)
        rows.append(compute_ers(shap_v[i], runs, perm, [b[i] for b in boot_v]))
        lime_all.append(np.mean(runs, axis=0))
        perm_all.append(perm)
    attrs = {"SHAP": shap_v, "LIME": np.array(lime_all), "Permutation": np.array(perm_all),
             "base": base, "unit": unit}
    table = pd.DataFrame(rows, index=X_eval.index)
    table["level"] = table.ERS.map(lambda e: reliability_level(e, cuts))
    return table, attrs


def get_cutoffs(ds, model, pipe, boot_pipes, X_train):
    """Relative High/Medium/Low cut-offs (33rd/67th percentile of ERS on reference TRAIN patients).
    Computed once and cached in results/."""
    path = RESULTS / f"{ds}_{slug(model)}_ers_cutoffs.json"
    if path.exists():
        c = json.load(open(path))
        return c["low"], c["high"]
    ref = X_train.sample(min(N_REFERENCE, len(X_train)), random_state=SEED + 1)
    tab, _ = explain_and_score(pipe, boot_pipes, X_train, ref)
    low, high = (float(np.percentile(tab.ERS, q)) for q in (100 / 3, 200 / 3))
    json.dump({"low": low, "high": high, "n_reference": len(ref), "ref_mean": float(tab.ERS.mean())},
              open(path, "w"), indent=1)
    return low, high


# --------------------------------------------------------------------------- experiment E
def run_experiment(ds, model_name=None):
    """Research experiment: do the methods agree, and is low ERS linked to model errors?"""
    df, cfg = load_dataset(ds)
    train_df, test_df = train_test_split_df(df, cfg)
    X_tr, y_tr = split_xy(train_df, cfg)
    X_te, y_te = split_xy(test_df, cfg)
    art = load_artifact(ds, model_name)
    model_name = art["model_name"]
    pipe, thr = art["pipeline"], art["threshold"]
    print(f"\n=== ERS experiment: {ds} / {model_name} (n_test={len(X_te)}) ===")

    boot = get_bootstrap_pipes(ds, model_name, pipe, X_tr, y_tr)
    cuts = get_cutoffs(ds, model_name, pipe, boot, X_tr)
    print(f"  relative cut-offs from reference train patients: Low < {cuts[0]:.1f} <= Medium < {cuts[1]:.1f} <= High")
    table, attrs = explain_and_score(pipe, boot, X_tr, X_te, cuts=cuts)

    # was the (calibrated) model right for each patient, at the tuned threshold?
    p = art["calibrated"].predict_proba(X_te)[:, 1]
    table["proba"] = p
    table["y_true"] = y_te.values
    table["y_pred"] = (p >= thr).astype(int)
    table["wrong"] = (table["y_pred"] != table["y_true"]).astype(int)
    table["margin"] = np.abs(p - thr)         # simple baseline: distance from the decision threshold
    slug_ = f"{ds}_{slug(model_name)}"
    table.to_csv(RESULTS / f"{slug_}_ers_test.csv")

    s = summarise(table, ds, model_name)
    s["level_cutoffs"] = {"low": cuts[0], "high": cuts[1]}
    plot_experiment(table, s, slug_, ds, model_name)
    json.dump(s, open(RESULTS / f"{slug_}_ers_summary.json", "w"), indent=2)
    return table, s


def auc_or_nan(y, score):
    return float(roc_auc_score(y, score)) if len(set(y)) == 2 else float("nan")


def summarise(t, ds, model_name):
    s = {"dataset": ds, "model": model_name, "n": int(len(t)), "n_wrong": int(t.wrong.sum())}
    # --- how often do the methods agree?
    s["agreement_rates"] = {
        "all three share the same #1 feature": float(t.same_top1.mean()),
        "all three share >=2 of their top-3 features": float((t.all3_top3_overlap >= 2).mean()),
        "mean pairwise Spearman >= 0.7": float((t[[c for c in t if c.startswith("rho_")]].mean(axis=1) >= 0.7).mean()),
    }
    s["mean_pairwise"] = {f"{a} vs {b}": {"spearman": float(t[f"rho_{a}_{b}"].mean()),
                                          "top3_overlap": float(t[f"top3_{a}_{b}"].mean())}
                          for a, b in itertools.combinations(METHODS, 2)}
    s["ers_distribution"] = {"mean": float(t.ERS.mean()), "std": float(t.ERS.std()),
                             "min": float(t.ERS.min()), "max": float(t.ERS.max()),
                             "levels": t.level.value_counts().to_dict()}
    # --- is low ERS linked to errors?
    right, wrong = t[t.wrong == 0].ERS, t[t.wrong == 1].ERS
    s["ers_correct_mean"], s["ers_wrong_mean"] = float(right.mean()), float(wrong.mean())
    s["mannwhitney_p_wrong_lower"] = (float(mannwhitneyu(wrong, right, alternative="less").pvalue)
                                      if len(wrong) > 0 and len(right) > 0 else float("nan"))
    s["auc_error_detection"] = {                       # AUROC for spotting wrong predictions (0.5 = useless)
        "ERS (low = suspicious)": auc_or_nan(t.wrong, -t.ERS),
        "agreement only": auc_or_nan(t.wrong, -t.agreement),
        "stability only": auc_or_nan(t.wrong, -t.stability),
        "baseline: closeness to threshold": auc_or_nan(t.wrong, -t.margin),
    }
    s["spearman_ers_vs_margin"] = float(spearmanr(t.ERS, t.margin).correlation)
    s["error_rate_by_level"] = {lv: {"n": int((t.level == lv).sum()),
                                     "error_rate": float(t[t.level == lv].wrong.mean())}
                                for lv in ("High", "Medium", "Low") if (t.level == lv).any()}
    # --- sensitivity of the finding to the (arbitrary) 50/50 weights
    s["weight_sensitivity_auc"] = {
        f"w_agree={w}": auc_or_nan(t.wrong, -(100 * (w * t.agreement + (1 - w) * t.stability)))
        for w in (0.0, 0.25, 0.5, 0.75, 1.0)}
    return s


def plot_experiment(t, s, slug_, ds, model_name):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    # 1) ERS of correct vs wrong predictions
    data = [t[t.wrong == 0].ERS, t[t.wrong == 1].ERS]
    ax[0].boxplot(data, labels=[f"correct (n={len(data[0])})", f"wrong (n={len(data[1])})"], showfliers=False)
    rng = np.random.RandomState(0)
    for i, d in enumerate(data, start=1):
        ax[0].scatter(i + rng.uniform(-0.15, 0.15, len(d)), d, alpha=0.5, s=14,
                      color=["tab:blue", "tab:red"][i - 1])
    for c in s["level_cutoffs"].values():             # relative Low/Medium/High cut-offs
        ax[0].axhline(c, color="gray", ls=":")
    ax[0].set(ylabel="ERS (0-100)", title="ERS vs correctness")
    # 2) error rate per reliability level
    lv = s["error_rate_by_level"]
    ax[1].bar(list(lv), [v["error_rate"] for v in lv.values()],
              color=[{"High": "tab:green", "Medium": "tab:orange", "Low": "tab:red"}[k] for k in lv])
    for i, (k, v) in enumerate(lv.items()):
        ax[1].text(i, v["error_rate"] + 0.005, f"n={v['n']}", ha="center")
    ax[1].set(ylabel="Share of wrong predictions", title="Error rate by ERS level")
    # 3) ERS vs closeness to threshold (is ERS just a proxy for model confidence?)
    ax[2].scatter(t.margin, t.ERS, c=t.wrong.map({0: "tab:blue", 1: "tab:red"}), alpha=0.6, s=18)
    ax[2].set(xlabel="|probability - threshold|", ylabel="ERS",
              title=f"ERS vs model confidence (Spearman {s['spearman_ers_vs_margin']:.2f})")
    fig.suptitle(f"{ds} / {model_name}   AUROC(ERS->error) = {s['auc_error_detection']['ERS (low = suspicious)']:.2f}",
                 y=1.02)
    plt.tight_layout(); plt.savefig(RESULTS / f"{slug_}_ers_vs_correctness.png", dpi=130, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="both", choices=["heart", "diabetes", "both"])
    ap.add_argument("--models", nargs="*", default=[None, "Random Forest", "XGBoost"],
                    help="model names; omit for final + Random Forest + XGBoost")
    a = ap.parse_args()
    for d in (["heart", "diabetes"] if a.dataset == "both" else [a.dataset]):
        for m in a.models:
            _, summ = run_experiment(d, m)
            print(json.dumps({k: summ[k] for k in ("agreement_rates", "ers_correct_mean", "ers_wrong_mean",
                                                   "mannwhitney_p_wrong_lower", "auc_error_detection",
                                                   "error_rate_by_level")}, indent=1))
