"""Explanation methods: SHAP, LIME, local permutation importance, plain-language text.

All three methods return ONE number per ORIGINAL feature (e.g. "ChestPainType", not its four
one-hot columns). They work on the transformed space (after imputation / scaling / one-hot) and
then add up the one-hot columns that belong to the same original feature. Adding is valid
because SHAP values and LIME's local-linear weights are additive.

Sign convention everywhere: positive value = pushes predicted RISK UP.
"""
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.data import SEED


# --------------------------------------------------------------------------- helpers
def _parts(pipe):
    return pipe.named_steps["pre"], pipe.named_steps["clf"]


def original_features(pipe):
    """Original feature names in the order the explanations use (numeric first, then categorical)."""
    pre, _ = _parts(pipe)
    return [c for _, _, cols in pre.transformers_ if cols is not None for c in cols
            if isinstance(cols, list)]


def grouping_matrix(pipe):
    """0/1 matrix G (n_transformed x n_original): transformed_values @ G = per-original-feature values."""
    pre, _ = _parts(pipe)
    sizes = []                                     # how many transformed columns each original feature has
    for name, trans, cols in pre.transformers_:
        if name == "num":
            sizes += [1] * len(cols)
        elif name == "cat":
            sizes += [len(c) for c in trans.named_steps["oh"].categories_]
    n_t = sum(sizes)
    G = np.zeros((n_t, len(sizes)))
    start = 0
    for j, s in enumerate(sizes):
        G[start:start + s, j] = 1
        start += s
    return G


def n_numeric(pipe):
    pre, _ = _parts(pipe)
    return sum(len(cols) for name, _, cols in pre.transformers_ if name == "num")


# --------------------------------------------------------------------------- SHAP
def shap_values(pipe, X_background, X):
    """SHAP values per original feature. Returns (values[n, p], base_value, unit_label).

    Exact explainers are used for the models we deploy: LinearExplainer (logistic regression),
    TreeExplainer (random forest, XGBoost). Anything else falls back to (slow, approximate) Kernel SHAP.
    """
    pre, clf = _parts(pipe)
    Xb, Xt = pre.transform(X_background), pre.transform(X)
    if isinstance(clf, LogisticRegression):
        ex = shap.LinearExplainer(clf, Xb)
        vals, base, unit = ex.shap_values(Xt), ex.expected_value, "log-odds"
    elif isinstance(clf, XGBClassifier):
        ex = shap.TreeExplainer(clf)
        vals, base, unit = ex.shap_values(Xt), ex.expected_value, "log-odds"
    elif isinstance(clf, RandomForestClassifier):
        ex = shap.TreeExplainer(clf)
        vals, base, unit = ex.shap_values(Xt), ex.expected_value, "probability"
        if isinstance(vals, list):                   # old shap versions: list per class
            vals, base = vals[1], base[1]
        elif vals.ndim == 3:                         # new shap: (n, features, classes)
            vals, base = vals[:, :, 1], base[1]
    else:
        f = lambda z: clf.predict_proba(z)[:, 1]
        ex = shap.KernelExplainer(f, shap.kmeans(Xb, 10))
        vals, base, unit = ex.shap_values(Xt, nsamples=200, silent=True), ex.expected_value, "probability"
    return np.asarray(vals) @ grouping_matrix(pipe), float(np.ravel(base)[0]), unit


# --------------------------------------------------------------------------- LIME
def make_lime(pipe, X_train):
    """LIME explainer built once per model. One-hot columns are declared categorical."""
    pre, _ = _parts(pipe)
    Xt = pre.transform(X_train)
    n_num = n_numeric(pipe)
    return LimeTabularExplainer(
        Xt, feature_names=list(pre.get_feature_names_out()),
        categorical_features=list(range(n_num, Xt.shape[1])),
        mode="classification", discretize_continuous=True, random_state=SEED)


def lime_values(pipe, lime_explainer, x_row, seed, num_samples=1000):
    """One LIME run for one patient (a 1-row DataFrame). Different seed -> different random perturbations."""
    pre, clf = _parts(pipe)
    lime_explainer.random_state.seed(seed)           # re-seed the shared RandomState in place
    xt = pre.transform(x_row)[0]
    exp = lime_explainer.explain_instance(xt, clf.predict_proba, labels=(1,),
                                          num_features=len(xt), num_samples=num_samples)
    w = np.zeros(len(xt))
    for idx, weight in exp.as_map()[1]:
        w[idx] = weight
    return w @ grouping_matrix(pipe)


# --------------------------------------------------------------------------- local permutation
def permutation_values(pipe, X_background, x_row, n_draws=50, seed=SEED):
    """LOCAL permutation importance for one patient.

    Classic permutation importance is global (shuffle a column over the whole test set).
    For a single patient we do the local analogue: replace feature j by values drawn from
    real training patients, and see how far the predicted risk moves:
        value_j = risk(patient) - average risk(patient with feature j replaced)
    Positive = the patient's actual value pushes risk up compared with a typical value.
    """
    base = pipe.predict_proba(x_row)[0, 1]
    bg = X_background.sample(n_draws, replace=True, random_state=seed).reset_index(drop=True)
    cols = list(x_row.columns)
    blocks = []
    for c in cols:
        block = pd.concat([x_row] * n_draws, ignore_index=True)
        block[c] = bg[c].values
        blocks.append(block)
    probs = pipe.predict_proba(pd.concat(blocks, ignore_index=True))[:, 1].reshape(len(cols), n_draws)
    return base - probs.mean(axis=1)


def global_permutation(pipe, X, y, n_repeats=10):
    """Classic (global) permutation importance on raw features, scored by ROC-AUC drop."""
    r = permutation_importance(pipe, X, y, scoring="roc_auc", n_repeats=n_repeats, random_state=SEED)
    return pd.Series(r.importances_mean, index=X.columns)


# --------------------------------------------------------------------------- plain language
# (readable name, unit) for numeric features
NAMES = {
    "Age": ("age", "years"), "RestingBP": ("resting blood pressure", "mmHg"),
    "Cholesterol": ("cholesterol", "mg/dL"), "FastingBS": ("fasting blood sugar", ""),
    "MaxHR": ("maximum heart rate", "bpm"), "Oldpeak": ("exercise ST depression", ""),
    "Sex": ("sex", ""), "ChestPainType": ("chest pain type", ""),
    "RestingECG": ("resting ECG", ""), "ExerciseAngina": ("exercise-induced angina", ""),
    "ST_Slope": ("ST slope", ""), "Pregnancies": ("number of pregnancies", ""),
    "Glucose": ("glucose", "mg/dL"), "BloodPressure": ("diastolic blood pressure", "mmHg"),
    "SkinThickness": ("skin-fold thickness", "mm"), "Insulin": ("insulin", "µU/mL"),
    "BMI": ("BMI", "kg/m²"), "DiabetesPedigreeFunction": ("family-history score", ""),
}
CATEGORY_TEXT = {
    "ChestPainType": {"ASY": "no chest-pain symptoms (asymptomatic)", "ATA": "atypical angina",
                      "NAP": "non-anginal pain", "TA": "typical angina"},
    "ExerciseAngina": {"Y": "present", "N": "absent"},
    "FastingBS": {1: "above 120 mg/dL", 0: "normal"},
    "ST_Slope": {"Up": "upsloping", "Flat": "flat", "Down": "downsloping"},
    "RestingECG": {"Normal": "normal", "ST": "ST-T abnormality", "LVH": "left ventricular hypertrophy"},
    "Sex": {"M": "male", "F": "female"},
}


def describe(feature, value):
    """'cholesterol 240 mg/dL' / 'exercise-induced angina: present' / '(not recorded)'."""
    label, unit = NAMES.get(feature, (feature, ""))
    if pd.isna(value):
        return f"{label} (not recorded)"
    if feature in CATEGORY_TEXT:
        v = int(value) if feature == "FastingBS" else value
        return f"{label}: {CATEGORY_TEXT[feature].get(v, v)}"
    v = f"{value:.1f}" if isinstance(value, (float, np.floating)) and not float(value).is_integer() else f"{int(value)}"
    return f"{label} {v} {unit}".strip()


def plain_language(shap_row, x_row, risk, threshold, features, min_share=0.05):
    """Turn one patient's SHAP values into 2-3 readable sentences.

    shap_row: 1-D array aligned with `features`; x_row: Series of raw values.
    A feature is mentioned only if it carries at least `min_share` of the total absolute SHAP mass.
    """
    shap_row = np.asarray(shap_row, dtype=float)
    total = np.abs(shap_row).sum() + 1e-12
    order = np.argsort(-np.abs(shap_row))
    up = [features[i] for i in order if shap_row[i] > 0 and abs(shap_row[i]) / total >= min_share][:3]
    down = [features[i] for i in order if shap_row[i] < 0 and abs(shap_row[i]) / total >= min_share][:2]

    flagged = risk >= threshold
    s1 = (f"The model estimates a {risk:.0%} risk, which is {'above' if flagged else 'below'} the "
          f"screening threshold of {threshold:.0%}, so this patient is flagged as "
          f"{'HIGHER' if flagged else 'LOWER'} risk.")
    sentences = [s1]
    if up:
        sentences.append("The factors raising the risk most are " +
                         _join([describe(f, x_row[f]) for f in up]) + ".")
    if down:
        sentences.append("Factors working in the patient's favour: " +
                         _join([describe(f, x_row[f]) for f in down]) + ".")
    elif not up:
        sentences.append("No single factor stands out strongly for this patient.")
    return " ".join(sentences)


def _join(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
