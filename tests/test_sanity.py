"""Sanity checks. Run from the project root:  python -m pytest -q
(needs the trained models: run `python -m src.models` first)."""
import numpy as np
import pandas as pd
import pytest

from src import counterfactuals as cf
from src import ers, explain as ex
from src.data import SEED, feature_cols, load_dataset, split_xy, train_test_split_df
from src.models import build_pipeline, choose_threshold, load_artifact, model_zoo, recall_score


@pytest.fixture(scope="module", params=["heart", "diabetes"])
def setup(request):
    ds = request.param
    df, cfg = load_dataset(ds)
    tr, te = train_test_split_df(df, cfg)
    X_tr, y_tr = split_xy(tr, cfg)
    X_te, y_te = split_xy(te, cfg)
    return ds, cfg, df, X_tr, y_tr, X_te, y_te, load_artifact(ds)


# ------------------------------------------------------------------ data
def test_hidden_zeros_become_missing():
    d = load_dataset("diabetes")[0]
    for c in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        assert (d[c] == 0).sum() == 0 and d[c].isna().sum() > 0
    h = load_dataset("heart")[0]
    assert (h["Cholesterol"] == 0).sum() == 0 and (h["RestingBP"] == 0).sum() == 0
    assert len(h) == 918 and set(h["Source"]) == {"Cleveland", "Hungarian", "Switzerland", "VA Long Beach"}


def test_split_is_disjoint_and_stratified(setup):
    ds, cfg, df, X_tr, y_tr, X_te, y_te, _ = setup
    assert set(X_tr.index).isdisjoint(X_te.index) and len(X_tr) + len(X_te) == len(df)
    assert abs(y_tr.mean() - y_te.mean()) < 0.03
    assert list(X_tr.columns) == feature_cols(cfg) and cfg["target"] not in X_tr.columns


# ------------------------------------------------------------------ leakage
def test_preprocessing_is_fitted_on_training_data_only(setup):
    ds, cfg, df, X_tr, y_tr, X_te, y_te, _ = setup
    pipe = build_pipeline(model_zoo()["Logistic Regression"][0], cfg).fit(X_tr, y_tr)
    pre = pipe.named_steps["pre"]
    scaler, imputer = pre.named_transformers_["num"]["sc"], pre.named_transformers_["num"]["imp"]
    # Pipeline statistics equal the TRAIN statistics ...
    assert np.allclose(imputer.statistics_, X_tr[cfg["numeric"]].median().values, equal_nan=True)
    imputed = imputer.transform(X_tr[cfg["numeric"]])
    assert np.allclose(scaler.mean_, imputed.mean(axis=0))
    # ... and differ from statistics that would include the test rows (so no leak)
    all_rows = imputer.transform(pd.concat([X_tr, X_te])[cfg["numeric"]])
    assert not np.allclose(scaler.mean_, all_rows.mean(axis=0))


# ------------------------------------------------------------------ threshold
def test_threshold_reaches_target_recall():
    rng = np.random.RandomState(SEED)
    y = rng.randint(0, 2, 500)
    p = np.clip(0.35 * y + rng.rand(500) * 0.65, 0, 1)
    t = choose_threshold(y, p, 0.90)
    assert recall_score(y, p >= t) >= 0.90


def test_final_model_threshold_meets_recall_on_train(setup):
    ds, cfg, df, X_tr, y_tr, X_te, y_te, art = setup
    p = art["calibrated"].predict_proba(X_tr)[:, 1]      # in-sample, so recall must be >= ~0.9
    assert recall_score(y_tr, p >= art["threshold"]) >= 0.88


# ------------------------------------------------------------------ explanations
def test_attribution_shapes_and_additivity(setup):
    ds, cfg, df, X_tr, y_tr, X_te, y_te, art = setup
    pipe = art["pipeline"]
    feats = ex.original_features(pipe)
    assert feats == feature_cols(cfg)
    bg = X_tr.sample(100, random_state=0)
    sv, base, unit = ex.shap_values(pipe, bg, X_te.iloc[:4])
    assert sv.shape == (4, len(feats))
    assert ex.lime_values(pipe, ex.make_lime(pipe, X_tr), X_te.iloc[[0]], 1).shape == (len(feats),)
    assert ex.permutation_values(pipe, bg, X_te.iloc[[0]]).shape == (len(feats),)
    if art["model_name"] == "Logistic Regression":       # SHAP is exactly additive in log-odds
        p = pipe.predict_proba(X_te.iloc[:4])[:, 1]
        assert np.allclose(base + sv.sum(1), np.log(p / (1 - p)), atol=1e-6)


def test_plain_language_is_short(setup):
    ds, cfg, df, X_tr, y_tr, X_te, y_te, art = setup
    pipe = art["pipeline"]
    sv, _, _ = ex.shap_values(pipe, X_tr.sample(100, random_state=0), X_te.iloc[:1])
    text = ex.plain_language(sv[0], X_te.iloc[0], 0.6, art["threshold"], ex.original_features(pipe))
    assert 2 <= text.count(". ") + 1 <= 4


# ------------------------------------------------------------------ ERS
def test_ers_is_between_0_and_100():
    rng = np.random.RandomState(SEED)
    for _ in range(200):
        p = rng.randint(4, 12)
        v = lambda: rng.randn(p) * rng.choice([0, 1, 5])      # includes all-zero vectors
        out = ers.compute_ers(v(), [v() for _ in range(5)], v(), [v() for _ in range(5)])
        assert 0 <= out["ERS"] <= 100 and out["level"] in ("High", "Medium", "Low")


def test_ers_is_100_for_identical_and_low_for_unrelated_attributions():
    base = np.array([5, 4, 3, 2, 1, 0.5, 0.1, 0.0])
    same = ers.compute_ers(base, [base] * 5, base, [base] * 5)
    assert same["ERS"] == pytest.approx(100.0)
    rng = np.random.RandomState(1)
    noisy = [ers.compute_ers(rng.randn(8), [rng.randn(8) for _ in range(5)], rng.randn(8),
                             [rng.randn(8) for _ in range(5)])["ERS"] for _ in range(100)]
    assert np.mean(noisy) < 50          # unrelated explanations must score clearly lower


def test_ers_is_reproducible(setup):
    ds, cfg, df, X_tr, y_tr, X_te, y_te, art = setup
    pipe = art["pipeline"]
    boot = ers.make_bootstrap_pipes(pipe, X_tr, y_tr, n=2)
    a, _ = ers.explain_and_score(pipe, boot, X_tr, X_te.iloc[:2], n_lime=2)
    b, _ = ers.explain_and_score(pipe, boot, X_tr, X_te.iloc[:2], n_lime=2)
    assert np.allclose(a.ERS, b.ERS) and ((a.ERS >= 0) & (a.ERS <= 100)).all()


# ------------------------------------------------------------------ counterfactuals
def test_counterfactuals_never_touch_immutable_features(setup):
    ds, cfg, df, X_tr, y_tr, X_te, y_te, art = setup
    model, thr = art["calibrated"], art["threshold"]
    p = model.predict_proba(X_te)[:, 1]
    flagged = X_te[p >= thr]
    checked = 0
    for i in range(min(len(flagged), 12)):
        opts, _ = cf.suggest(cfg, art, X_tr, y_tr, flagged.iloc[[i]])
        for o in opts:
            assert set(o["changes"]) <= set(cfg["modifiable"])          # nothing else (age, sex, ...) moved
            assert not {"Age", "Sex"} & set(o["changes"])
            assert o["new_risk"] < thr
            checked += 1
        if checked >= 3:
            break
    assert checked > 0
