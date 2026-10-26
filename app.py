"""Streamlit app: cardiometabolic risk + explanation + how much to trust the explanation.

Run:  streamlit run app.py        (train the models first:  python -m src.models)
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src import counterfactuals as cfm
from src import data as D
from src import ers
from src import explain as ex
from src.models import RESULTS, load_artifact, slug

st.set_page_config(page_title="Explainable Cardiometabolic Risk", page_icon="🫀", layout="wide")

MODEL_CHOICES = ["Logistic Regression", "Random Forest", "XGBoost"]   # models with exact SHAP
LEVEL_COLOUR = {"High": "#2e7d32", "Medium": "#ef6c00", "Low": "#c62828"}
CAT_OPTIONS = {   # categorical feature -> {stored value: label shown in the sidebar}
    "Sex": {"M": "Male", "F": "Female"},
    "ChestPainType": {"ASY": "Asymptomatic", "ATA": "Atypical angina", "NAP": "Non-anginal pain", "TA": "Typical angina"},
    "RestingECG": {"Normal": "Normal", "ST": "ST-T abnormality", "LVH": "Left ventricular hypertrophy"},
    "ExerciseAngina": {"N": "No", "Y": "Yes"},
    "ST_Slope": {"Up": "Upsloping", "Flat": "Flat", "Down": "Downsloping"},
}

STYLE = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1250px;}
.hero {padding: 22px 28px; border-radius: 18px; margin-bottom: 14px;
       background: linear-gradient(120deg, #3a0f24 0%, #1b2140 55%, #0f3b4a 100%);
       border: 1px solid rgba(255,255,255,.08);}
.hero h1 {margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -.5px;}
.hero p {margin: 6px 0 0; opacity: .8; font-size: 1rem;}
.card {padding: 16px 18px; border-radius: 14px; background: #171b26; border: 1px solid rgba(255,255,255,.07); height: 100%;}
.card h5 {margin: 0 0 8px; font-size: .8rem; text-transform: uppercase; letter-spacing: 1px; opacity: .65;}
.pill {display:inline-block; padding: 3px 12px; border-radius: 999px; font-weight: 700; font-size: .85rem;}
div[data-testid="stMetric"] {background:#171b26; border:1px solid rgba(255,255,255,.07); padding:10px 14px; border-radius:12px;}
button[data-baseweb="tab"] {font-weight: 600;}
#MainMenu, footer {visibility: hidden;}
</style>
"""


def gauge_html(risk, thr):
    """Semicircle risk gauge (pure SVG): needle = risk, tick = screening threshold."""
    import math
    def pt(p, r):
        a = math.pi * (1 - p)
        return 120 + r * math.cos(a), 120 - r * math.sin(a)
    x, y = pt(min(max(risk, 0), 1), 78)
    tx1, ty1 = pt(thr, 84); tx2, ty2 = pt(thr, 104)
    col = "#e53935" if risk >= thr else "#43a047"
    return f"""<svg viewBox="0 0 240 150" width="100%">
    <defs><linearGradient id="g" x1="0" x2="1"><stop offset="0" stop-color="#43a047"/><stop offset=".5" stop-color="#fbc02d"/>
    <stop offset="1" stop-color="#e53935"/></linearGradient></defs>
    <path d="M 20 120 A 100 100 0 0 1 220 120" fill="none" stroke="url(#g)" stroke-width="16" stroke-linecap="round" opacity=".9"/>
    <line x1="{tx1:.1f}" y1="{ty1:.1f}" x2="{tx2:.1f}" y2="{ty2:.1f}" stroke="white" stroke-width="3"/>
    <line x1="120" y1="120" x2="{x:.1f}" y2="{y:.1f}" stroke="{col}" stroke-width="5" stroke-linecap="round"/>
    <circle cx="120" cy="120" r="8" fill="{col}"/>
    <text x="120" y="108" text-anchor="middle" font-size="26" font-weight="800" fill="white" dy="-34">{risk:.0%}</text>
    <text x="20" y="140" font-size="9" fill="#aaa" text-anchor="middle">0%</text>
    <text x="220" y="140" font-size="9" fill="#aaa" text-anchor="middle">100%</text>
    <text x="{tx2:.1f}" y="{ty2-4:.1f}" font-size="8" fill="white" text-anchor="middle">threshold {thr:.0%}</text></svg>"""


def label(f):
    """Sidebar label: readable name, first letter capitalised, acronyms upper-cased."""
    words = [{"ecg": "ECG", "st": "ST", "bmi": "BMI"}.get(w, w) for w in ex.NAMES[f][0].split()]
    return " ".join(words)[0].upper() + " ".join(words)[1:]


# --------------------------------------------------------------------------- cached heavy objects
@st.cache_resource(show_spinner="Loading model and reference data...")
def load_bundle(ds, model_name):
    df, cfg = D.load_dataset(ds)
    train_df, test_df = D.train_test_split_df(df, cfg)
    X_tr, y_tr = D.split_xy(train_df, cfg)
    X_te, y_te = D.split_xy(test_df, cfg)
    art = load_artifact(ds, model_name)
    boot = ers.get_bootstrap_pipes(ds, art["model_name"], art["pipeline"], X_tr, y_tr)
    cuts = ers.get_cutoffs(ds, art["model_name"], art["pipeline"], boot, X_tr)
    return dict(cfg=cfg, X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te, art=art, boot=boot, cuts=cuts)


def final_model_name(ds):
    return json.load(open(RESULTS / f"{ds}_final_metrics.json"))["final_model"]


@st.cache_data(show_spinner=False)
def assess(ds, model_name, values):
    """Everything computed for one patient. `values` is a tuple of (feature, value) pairs (cache key)."""
    b = load_bundle(ds, model_name)
    cfg, art = b["cfg"], b["art"]
    row = pd.DataFrame([dict(values)])[D.feature_cols(cfg)]
    for c in cfg["numeric"]:
        row[c] = row[c].astype(float)
    risk = float(art["calibrated"].predict_proba(row)[0, 1])
    table, attrs = ers.explain_and_score(art["pipeline"], b["boot"], b["X_tr"], row, cuts=b["cuts"])
    feats = ex.original_features(art["pipeline"])
    text = ex.plain_language(attrs["SHAP"][0], row.iloc[0], risk, art["threshold"], feats,
                            avg_risk=float(b["y_tr"].mean()))
    options, cf_msg = [], ""
    if risk >= art["threshold"]:
        options, cf_msg = cfm.suggest(cfg, art, b["X_tr"], b["y_tr"], row)
    return dict(row=row, risk=risk, ers=table.iloc[0].to_dict(), attrs=attrs, feats=feats,
                text=text, options=options, cf_msg=cf_msg)


@st.cache_data(show_spinner=False)
def train_risks(ds, model_name):
    b = load_bundle(ds, model_name)
    return b["art"]["calibrated"].predict_proba(b["X_tr"])[:, 1]


def nearest_patients(bundle, row, k=15):
    """k most similar training patients (z-scored numeric distance + categorical mismatch)."""
    cfg, X = bundle["cfg"], bundle["X_tr"]
    num = [c for c in cfg["numeric"] if c not in cfg["binary_numeric"]]
    sd = X[num].std().replace(0, 1)
    med = X[num].median()
    d = (((X[num].fillna(med) - row[num].iloc[0].fillna(med)) / sd) ** 2).sum(axis=1)
    for c in cfg["categorical"] + cfg["binary_numeric"]:
        d = d + (X[c] != row[c].iloc[0]).astype(float) * 1.5
    idx = d.sort_values().index[:k]
    return X.loc[idx], bundle["y_tr"].loc[idx], np.sqrt(d.loc[idx])


def build_report(ds_label, art, res, thr):
    e = res["ers"]
    top = np.argsort(-np.abs(res["attrs"]["SHAP"][0]))[:5]
    lines = [f"# Cardiometabolic risk report - {ds_label}", "",
             "> Educational tool, not a diagnosis.", "",
             f"- Estimated risk: **{res['risk']:.0%}** (screening threshold {thr:.0%}) -> "
             f"{'HIGHER' if res['risk'] >= thr else 'LOWER'}",
             f"- Model: {art['model_name']} ({art['calibration_method']}-calibrated)",
             f"- Explanation reliability (ERS): **{e['ERS']:.0f}/100 ({e['level']})**", "",
             "## Patient values"]
    lines += [f"- {ex.NAMES[f][0]}: {_short(res['row'].iloc[0][f])}" for f in res["feats"]]
    lines += ["", "## Top drivers (SHAP)"]
    lines += [f"- {ex.NAMES[res['feats'][i]][0]}: {'raises' if res['attrs']['SHAP'][0][i] > 0 else 'lowers'} risk"
              for i in top]
    lines += ["", "## In plain language", res["text"]]
    return "\n".join(lines)


@st.cache_data(show_spinner=False)
def global_importance(ds, model_name):
    b = load_bundle(ds, model_name)
    return ex.global_permutation(b["art"]["pipeline"], b["X_te"], b["y_te"])


# --------------------------------------------------------------------------- sidebar
def patient_form(ds, bundle):
    """Sidebar widgets -> dict of feature values. Defaults are the training-set medians / modes."""
    cfg, X_tr = bundle["cfg"], bundle["X_tr"]
    values = {}
    for f in D.feature_cols(cfg):
        if f in CAT_OPTIONS:
            opts = CAT_OPTIONS[f]
            default = X_tr[f].mode().iloc[0]
            values[f] = st.sidebar.selectbox(label(f), list(opts),
                                             index=list(opts).index(default), format_func=opts.get)
        elif f in cfg["binary_numeric"]:
            values[f] = float(st.sidebar.checkbox("Fasting blood sugar > 120 mg/dL", value=False))
        else:
            lo, hi, step = cfg["ui"][f]
            cast = int if isinstance(step, int) else float
            default = cast(np.clip(round(float(X_tr[f].median()) / step) * step, lo, hi))
            text = label(f) + (f" ({ex.NAMES[f][1]})" if ex.NAMES[f][1] else "")
            unknown = f in cfg["unknown_ok"] and st.sidebar.checkbox(f"{text}: unknown", value=False, key=f"unk_{f}")
            if unknown:
                values[f] = np.nan
            else:
                values[f] = st.sidebar.slider(text, cast(lo), cast(hi), default, cast(step), key=f"sl_{f}")
    return values


# --------------------------------------------------------------------------- plots
def waterfall_figure(res):
    shown = [f"{ex.NAMES[f][0]} = {_short(res['row'].iloc[0][f])}" for f in res["feats"]]
    expl = shap.Explanation(values=res["attrs"]["SHAP"][0], base_values=res["attrs"]["base"],
                            data=np.array(shown, dtype=object), feature_names=shown)
    plt.figure()
    shap.plots.waterfall(expl, max_display=9, show=False)
    fig = plt.gcf()
    fig.set_size_inches(9, 5)
    return fig


def _short(v):
    if pd.isna(v):
        return "unknown"
    return f"{v:g}" if not isinstance(v, str) else v


def agreement_figure(res, top=8):
    """Normalised importance (|attribution| / sum) of the three methods for the top features."""
    imp = pd.DataFrame({m: np.abs(res["attrs"][m][0]) for m in ers.METHODS}, index=res["feats"])
    imp = imp / imp.sum()
    imp = imp.loc[imp.mean(axis=1).sort_values(ascending=False).index[:top]].iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    h = 0.26
    for k, (m, c) in enumerate(zip(ers.METHODS, ["#1f77b4", "#ff7f0e", "#2ca02c"])):
        ax.barh(np.arange(len(imp)) + (k - 1) * h, imp[m], h, label=m, color=c)
    ax.set_yticks(np.arange(len(imp)))
    ax.set_yticklabels([ex.NAMES[f][0] for f in imp.index])
    ax.set_xlabel("Share of total importance"); ax.legend(fontsize=8)
    ax.set_title("Do the three methods pick the same features?")
    plt.tight_layout()
    return fig


def signed_panels(res, top=8):
    """Compare tab: signed attribution of each method side by side (same feature order)."""
    order = np.argsort(-np.abs(res["attrs"]["SHAP"][0]))[:top][::-1]
    names = [ex.NAMES[res["feats"][i]][0] for i in order]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for a, m in zip(ax, ers.METHODS):
        v = res["attrs"][m][0][order]
        v = v / (np.abs(res["attrs"][m][0]).max() + 1e-12)          # scale each method to [-1, 1]
        a.barh(names, v, color=["#d32f2f" if x > 0 else "#1976d2" for x in v])
        a.axvline(0, color="black", lw=0.8)
        a.set_title(m); a.set_xlim(-1.05, 1.05)
        a.set_xlabel("scaled attribution (red = raises risk)")
    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------- page
st.markdown(STYLE, unsafe_allow_html=True)
st.markdown("<div class='hero'><h1>🫀 Explainable Cardiometabolic Risk</h1>"
            "<p>Predict heart-disease and diabetes risk, see <b>why</b>, and see <b>how much to trust the why</b>.</p></div>",
            unsafe_allow_html=True)
st.warning("**Educational tool, not a diagnosis.** It was built for a course project on small public "
           "datasets and must not be used to make medical decisions. Talk to a clinician about your health.")

ds_label = st.sidebar.radio("Condition", ["Heart disease", "Diabetes"])
ds = "heart" if ds_label == "Heart disease" else "diabetes"
final = final_model_name(ds)
model_name = st.sidebar.selectbox("Model", MODEL_CHOICES, index=MODEL_CHOICES.index(final) if final in MODEL_CHOICES else 0,
                                  format_func=lambda m: m + (" (best CV)" if m == final else ""))
bundle = load_bundle(ds, model_name)
art, cfg = bundle["art"], bundle["cfg"]

st.sidebar.header("Patient values")
values = patient_form(ds, bundle)
go = st.sidebar.button("Assess", type="primary")

tab_assess, tab_whatif, tab_similar, tab_compare, tab_perf = st.tabs(
    ["Assessment", "🧪 What-if lab", "👥 Similar patients", "Compare methods", "Model & experiments"])

if not go and "last" not in st.session_state:
    with tab_assess:
        st.info("Enter patient values in the sidebar and press **Assess**.")
    st.stop()
if go:
    st.session_state["last"] = (ds, model_name, tuple(values.items()))
last_ds, last_model, last_vals = st.session_state["last"]
if (last_ds, last_model) != (ds, model_name):      # user switched dataset/model: need a fresh Assess
    with tab_assess:
        st.info("Dataset or model changed - press **Assess** again.")
    st.stop()

with st.spinner("Computing SHAP, LIME, permutation importance and the reliability score..."):
    res = assess(last_ds, last_model, last_vals)
thr, risk, e = art["threshold"], res["risk"], res["ers"]

# ---------------------------------------------------------------- tab 1
with tab_assess:
    c1, c2, c3 = st.columns([1.1, 1.4, 1.5])
    with c1:
        st.markdown("##### Estimated risk")
        st.markdown(gauge_html(risk, thr), unsafe_allow_html=True)
        pct = float((train_risks(last_ds, last_model) < risk).mean())
        st.caption(f"Higher than **{pct:.0%}** of patients in the training data.")
        flagged = risk >= thr
        st.markdown(f"**Risk level: {'🔴 HIGHER' if flagged else '🟢 LOWER'}** "
                    f"(screening threshold {thr:.0%})")
        st.caption(f"Threshold chosen so that at least 90% of true cases are caught (recall >= 90%), "
                   f"because a missed patient costs more than a false alarm. Probability is "
                   f"{art['calibration_method']}-calibrated {art['model_name']}.")
    with c2:
        colour = LEVEL_COLOUR[e["level"]]
        st.markdown(f"##### Explanation reliability (ERS)")
        st.markdown(f"<div style='display:inline-block;padding:10px 18px;border-radius:10px;background:{colour};"
                    f"color:white;font-size:28px;font-weight:700'>{e['ERS']:.0f}/100 &nbsp; {e['level']}</div>",
                    unsafe_allow_html=True)
        if e["level"] == "Low":
            st.error("⚠️ Explanations disagree for this patient, interpret with caution.")
        elif e["level"] == "Medium":
            st.warning("Explanations partly agree. Treat the top factors as indicative.")
        else:
            st.success("Explanation methods agree and are stable for this patient.")
        parts = {"agreement between methods": e["agreement"], "LIME stability": e["lime_stability"],
                 "model (re-training) stability": e["model_stability"]}
        weakest = min(parts, key=parts.get)
        st.caption(f"Weakest check: {weakest} ({parts[weakest]:.2f} out of 1).")
        st.caption(f"Level is relative to typical patients of this model "
                   f"(Low < {bundle['cuts'][0]:.0f} <= Medium < {bundle['cuts'][1]:.0f} <= High).")
    with c3:
        st.markdown("##### In plain language")
        st.write(res["text"])

    st.divider()
    left, right = st.columns(2)
    with left:
        st.markdown("##### Why this risk? (SHAP waterfall)")
        st.pyplot(waterfall_figure(res), clear_figure=True)
        st.caption(f"SHAP explains the model's raw score ({res['attrs']['unit']}); the headline risk above is "
                   f"its calibrated probability. Red bars push risk up, blue bars push it down.")
    with right:
        st.markdown("##### Do the explanation methods agree?")
        st.pyplot(agreement_figure(res), clear_figure=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Agreement", f"{e['agreement']:.2f}")
        m2.metric("LIME stability", f"{e['lime_stability']:.2f}")
        m3.metric("Model stability", f"{e['model_stability']:.2f}")
        m4.metric("All 3 share top-3", f"{int(e['all3_top3_overlap'])}/3")
        st.caption("ERS = 100 x (0.5 x Agreement + 0.5 x Stability). Agreement: SHAP vs LIME vs permutation. "
                   "Stability: LIME re-runs and models re-trained on bootstrap resamples.")

    st.divider()
    st.markdown("##### What would lower this patient's risk? (counterfactuals)")
    st.caption("Only modifiable factors are changed (" +
               ", ".join(ex.NAMES[f][0] for f in cfg["modifiable"]) + "). Age, sex and other fixed factors never change.")
    if risk < thr:
        st.success(f"Risk is already below the screening threshold ({thr:.0%}); nothing to lower.")
    elif res["options"]:
        st.write(res["cf_msg"])
        rows = []
        for i, o in enumerate(res["options"], 1):
            change = "; ".join(f"{ex.NAMES[f][0]}: {_short(a)} → {_short(b)}" for f, (a, b) in o["changes"].items())
            rows.append({"Option": i, "Suggested change": change, "New estimated risk": f"{o['new_risk']:.0%}"})
        st.table(pd.DataFrame(rows).set_index("Option"))
        st.caption("Suggestions come from DiCE and are re-checked with the final model. They show what the model "
                   "associates with lower risk; they are not medical advice or proof that a change will work.")
    else:
        st.info(res["cf_msg"])

# ---------------------------------------------------------------- what-if lab
with tab_whatif:
    st.markdown("#### 🧪 What-if lab: change a modifiable factor and watch the risk move")
    st.caption("Starts from the assessed patient. Only modifiable factors are adjustable. Uses the same calibrated model.")
    base = res["row"].copy()
    new = base.copy()
    cols = st.columns(min(3, len(cfg["modifiable"])))
    for i, f in enumerate(cfg["modifiable"]):
        lo, hi, step = cfg["ui"][f] if f in cfg["ui"] else (0, 1, 1)
        cur = base.iloc[0][f]
        cur = float(bundle["X_tr"][f].median()) if pd.isna(cur) else float(cur)
        with cols[i % len(cols)]:
            if f in cfg["binary_numeric"]:
                new[f] = float(st.checkbox(label(f), value=bool(cur), key=f"wi_{f}"))
            else:
                cast = int if isinstance(step, int) else float
                new[f] = st.slider(label(f), cast(lo), cast(hi), cast(np.clip(cur, lo, hi)), cast(step), key=f"wi_{f}")
    new_risk = float(art["calibrated"].predict_proba(new)[0, 1])
    m1, m2, m3 = st.columns(3)
    m1.metric("Original risk", f"{risk:.0%}")
    m2.metric("What-if risk", f"{new_risk:.0%}", f"{(new_risk - risk) * 100:+.1f} pts", delta_color="inverse")
    m3.metric("Status", "Higher" if new_risk >= thr else "Lower", "crosses threshold" if (new_risk >= thr) != (risk >= thr) else "same side")
    st.markdown("##### Sensitivity sweep")
    feat = st.selectbox("Sweep one factor across its range", list(cfg["modifiable"]), format_func=label, key="sweep_f")
    if feat in cfg["binary_numeric"]:
        grid = np.array([0.0, 1.0])
    else:
        lo, hi, _ = cfg["ui"][feat]
        grid = np.linspace(lo, hi, 40)
    sweep = pd.concat([new] * len(grid), ignore_index=True)
    sweep[feat] = grid
    pr = art["calibrated"].predict_proba(sweep)[:, 1]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    fig.patch.set_alpha(0); ax.set_facecolor("none")
    ax.plot(grid, pr, color="#ff4b6e", lw=2.5, marker="o" if len(grid) == 2 else None)
    ax.axhline(thr, color="white", ls="--", lw=1, alpha=.6); ax.text(grid[0], thr + .015, "threshold", color="white", fontsize=8)
    ax.axvline(new.iloc[0][feat], color="#4fc3f7", lw=1.2)
    ax.set_ylim(0, 1); ax.set_xlabel(label(feat)); ax.set_ylabel("Estimated risk")
    for item in [ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()]:
        item.set_color("white")
    for sp in ax.spines.values():
        sp.set_color("#555")
    st.pyplot(fig, clear_figure=True)

# ---------------------------------------------------------------- similar patients
with tab_similar:
    st.markdown("#### 👥 Patients most similar to this one")
    Xn, yn, dist = nearest_patients(bundle, res["row"])
    obs = float(yn.mean())
    c1, c2, c3 = st.columns(3)
    c1.metric("Model's risk", f"{risk:.0%}")
    c2.metric(f"Positive in {len(yn)} nearest", f"{obs:.0%}")
    c3.metric("Gap", f"{(risk - obs) * 100:+.0f} pts",
              "model and neighbours agree" if abs(risk - obs) < .15 else "check carefully", delta_color="off")
    st.caption("A sanity check on the model: if real look-alike patients mostly had the condition, a high risk is plausible. "
               "A large gap means the patient is unusual compared with the training data.")
    show = Xn.head(8).copy()
    show.insert(0, "Condition present", np.where(yn.head(8) == 1, "Yes", "No"))
    show.insert(1, "Distance", dist.head(8).round(2))
    st.dataframe(show, use_container_width=True)
    st.download_button("⬇️ Download report (Markdown)", build_report(ds_label, art, res, thr),
                       file_name=f"{ds}_risk_report.md", mime="text/markdown")

# ---------------------------------------------------------------- tab 2
with tab_compare:
    st.markdown("#### SHAP vs LIME vs permutation importance for this patient")
    st.pyplot(signed_panels(res), clear_figure=True)
    top = pd.DataFrame({m: [ex.NAMES[res["feats"][i]][0] for i in
                            np.argsort(-np.abs(res["attrs"][m][0]))[:3]] for m in ers.METHODS},
                       index=["#1", "#2", "#3"])
    st.markdown("**Top-3 features by each method**")
    st.table(top)
    pair_rows = [{"Pair": f"{a} vs {b}", "Spearman ρ": f"{e[f'rho_{a}_{b}']:.2f}", "Top-3 overlap": f"{e[f'top3_{a}_{b}']*3:.0f}/3"}
                 for a, b in [("SHAP", "LIME"), ("SHAP", "Permutation"), ("LIME", "Permutation")]]
    st.table(pd.DataFrame(pair_rows).set_index("Pair"))
    st.caption("Permutation here is a LOCAL version: the patient's value for a feature is swapped for values from real "
               "patients and we measure how much the predicted risk moves. The three methods use different scales, so "
               "each panel is scaled to [-1, 1]; only ranking and sign are compared.")
    st.markdown("#### Global permutation importance (whole test set, for reference)")
    gi = global_importance(ds, model_name).sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh([ex.NAMES[f][0] for f in gi.index], gi.values, color="#6a1b9a")
    ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled")
    plt.tight_layout(); st.pyplot(fig, clear_figure=True)

# ---------------------------------------------------------------- tab 3
with tab_perf:
    st.markdown(f"#### Model performance - {cfg['title']}")
    st.caption("5-fold CV on the training part (mean ± std), then hold-out test. Full tables: `results/`.")
    st.table(pd.read_csv(RESULTS / f"{ds}_cv_table.csv", index_col=0))
    st.table(pd.read_csv(RESULTS / f"{ds}_test_table.csv", index_col=0))
    st.markdown("**Test performance at each model's cost-sensitive threshold (>= 90% recall on training OOF)**")
    st.table(pd.read_csv(RESULTS / f"{ds}_test_at_tuned_threshold.csv", index_col=0))
    for fname, cap in [(f"{ds}_roc_pr.png", "ROC and precision-recall curves (test)"),
                       (f"{ds}_calibration.png", "Calibration"), (f"{ds}_threshold.png", "Threshold choice"),
                       (f"{ds}_fairness_subgroups.png", "Performance by subgroup"),
                       ("heart_cross_source.png" if ds == "heart" else None, "Cross-source generalisation"),
                       (f"{ds}_{slug(final)}_ers_vs_correctness.png", "Experiment: ERS vs correctness")]:
        if fname and (RESULTS / fname).exists():
            st.image(str(RESULTS / fname), caption=cap)
