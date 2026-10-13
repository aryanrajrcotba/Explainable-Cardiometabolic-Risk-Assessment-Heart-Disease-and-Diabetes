# Explainable Cardiometabolic Risk Assessment: Heart Disease and Diabetes Prediction

AML course project. A Streamlit app that predicts **heart-disease risk** and **diabetes risk**, explains each
prediction, and also tells the user **how much to trust that explanation** (the *Explanation Reliability Score*, ERS).

> **Educational tool, not a diagnosis.** Small public datasets, no clinical validation. Do not use it to make medical decisions.

---

## 1. Problem

Risk models for cardiometabolic disease are only useful in practice if clinicians can see *why* a patient is flagged.
But explanation methods (SHAP, LIME, permutation importance) often **disagree with each other** and can **change
between runs**. A single explanation is therefore presented with more confidence than it deserves.
This project (a) builds well-evaluated, calibrated, cost-sensitive classifiers, (b) explains them, and
(c) measures per prediction whether the explanation itself is trustworthy.

## 2. Approach (what is in the code)

| Piece | File | What it does |
|---|---|---|
| Data | `src/data.py` | Downloads UCI Heart Disease (Cleveland, Hungarian, Switzerland, VA Long Beach → 918 rows after removing 2 duplicates, **`Source` column kept**) and Pima diabetes. Impossible zeros (Pima glucose/BP/skin/insulin/BMI; heart cholesterol/BP) → `NaN`. |
| Models | `src/models.py` | Leakage-free `Pipeline` (impute → scale / one-hot → model). Logistic Regression, Random Forest, XGBoost, SVM, small MLP; `GridSearchCV` with stratified 5-fold; precision / recall / F1 / ROC-AUC / PR-AUC / Brier; calibration curves; Platt-vs-isotonic calibration; **cost-sensitive threshold**. |
| Explanations | `src/explain.py` | SHAP (exact: linear / tree explainers), LIME, **local** permutation importance, all summed back to the *original* features; plain-language text. |
| ERS + experiment | `src/ers.py` | The score (novelty A) and the research experiment (E). |
| Counterfactuals | `src/counterfactuals.py` | DiCE, restricted to modifiable features (novelty B). |
| Fairness | `src/fairness.py` | Subgroup performance and cross-source generalisation (novelty D). |
| App | `app.py` | Streamlit UI incl. the *Compare methods* tab. |
| Checks | `tests/test_sanity.py` | 18 unit-style checks (no leakage, shapes, SHAP additivity, ERS ∈ [0,100], counterfactuals never change age/sex, ...). |
| EDA | `notebooks/01_eda.ipynb` | Exploration only; nothing feeds the models. |

**Evaluation protocol.** One stratified 80/20 hold-out split (seed 42). All tuning happens by 5-fold CV on the 80 %.
The 20 % test part is only used to report. The decision threshold and the calibration method are chosen from
**out-of-fold predictions on the training part**, never from the test set.

**Cost-sensitive threshold.** A false negative (a sick patient sent home) is worse than a false positive (an extra check-up).
So we first require **recall ≥ 90 %** and then take the threshold with the **best precision** among those
(`choose_threshold` in `models.py`). `results/*_threshold.png` shows why 0.5 is the wrong operating point
(right panel: expected cost if a miss costs 5× a false alarm; the 5:1 ratio is an assumption for illustration).

**Final model.** Highest mean CV ROC-AUC. Logistic Regression wins for both datasets, but all five models are within
about one standard deviation of each other. Because the ERS is more interesting for non-linear models, the
experiment is also run for Random Forest and XGBoost, and the app lets you switch between the three.

## 3. How to run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# macOS only: XGBoost needs the OpenMP runtime ->  brew install libomp

./run_all.sh                 # reproduces everything in results/ (~10 min): models, ERS experiment, fairness, tests
streamlit run app.py         # the app (needs the trained models from run_all.sh / python -m src.models)
python -m pytest -q tests    # sanity checks only
```

Raw data is committed in `data/raw/` (and `src/data.py` re-downloads it if missing). Everything uses `SEED = 42`.
You will see many harmless numpy/Accelerate `RuntimeWarning: ... encountered in matmul` lines on macOS; they come
from numpy 2.0 + Apple's BLAS and do not affect results.

## 4. Results

Hold-out test set: 184 patients (heart), 154 patients (diabetes). CV = 5-fold on the training part, mean ± std
(the CV numbers are slightly optimistic because the same folds were used for tuning; the test columns are the unbiased ones).

### Heart disease (918 patients, 55 % positive)

5-fold CV (threshold 0.5):

| model | ROC-AUC | PR-AUC | precision | recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.867 ± 0.047 | 0.874 ± 0.052 | 0.807 ± 0.047 | 0.840 ± 0.026 | 0.822 ± 0.032 |
| Random Forest | 0.865 ± 0.046 | 0.864 ± 0.054 | 0.801 ± 0.054 | 0.852 ± 0.013 | 0.825 ± 0.031 |
| XGBoost | 0.865 ± 0.045 | 0.866 ± 0.060 | 0.805 ± 0.049 | 0.827 ± 0.033 | 0.815 ± 0.032 |
| SVM (RBF) | 0.867 ± 0.048 | 0.875 ± 0.053 | 0.798 ± 0.045 | 0.832 ± 0.019 | 0.814 ± 0.027 |
| MLP | 0.843 ± 0.062 | 0.844 ± 0.066 | 0.790 ± 0.062 | 0.813 ± 0.020 | 0.800 ± 0.040 |

Hold-out test, threshold 0.5, and then at each model's own cost-sensitive threshold (≥ 90 % recall on training OOF):

| model | ROC-AUC | PR-AUC | Brier | **@0.5** precision / recall | **@tuned thr.** threshold | precision | recall | FN | FP |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression (final) | 0.889 | 0.908 | 0.134 | 0.804 / 0.843 | 0.436 | 0.777 | 0.922 | 8 | 27 |
| Random Forest | 0.897 | 0.914 | 0.137 | 0.781 / 0.873 | 0.396 | 0.775 | 0.912 | 9 | 27 |
| XGBoost | 0.900 | 0.912 | 0.125 | 0.815 / 0.863 | 0.391 | 0.760 | 0.961 | 4 | 31 |
| SVM (RBF) | 0.890 | 0.903 | 0.132 | 0.798 / 0.853 | 0.426 | 0.770 | 0.922 | 8 | 28 |
| MLP | 0.864 | 0.887 | 0.160 | 0.782 / 0.843 | 0.409 | 0.780 | 0.941 | 6 | 27 |

For the final (calibrated) model the tuned threshold turns 15 missed patients into 8 at the price of 6 extra false alarms
(recall 0.853 → 0.922, precision 0.806 → 0.777).

### Diabetes (Pima, 768 patients, 35 % positive)

| model | CV ROC-AUC | CV PR-AUC | CV recall | test ROC-AUC | test PR-AUC | test Brier | **@tuned thr.** threshold | precision | recall | FN | FP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression (final) | 0.844 ± 0.016 | 0.745 ± 0.055 | 0.561 | 0.810 | 0.666 | 0.174 | 0.217 | 0.545 | 0.889 | 6 | 40 |
| Random Forest | 0.839 ± 0.018 | 0.731 ± 0.054 | 0.519 | 0.800 | 0.671 | 0.171 | 0.187 | 0.522 | 0.889 | 6 | 44 |
| XGBoost | 0.839 ± 0.017 | 0.741 ± 0.053 | 0.561 | 0.817 | 0.683 | 0.167 | 0.193 | 0.539 | 0.889 | 6 | 41 |
| SVM (RBF) | 0.846 ± 0.011 | 0.751 ± 0.042 | 0.533 | 0.805 | 0.676 | 0.175 | 0.204 | 0.533 | 0.889 | 6 | 42 |
| MLP | 0.831 ± 0.024 | 0.720 ± 0.054 | 0.598 | 0.826 | 0.691 | 0.164 | 0.180 | 0.544 | 0.907 | 5 | 41 |

Diabetes is a harder, noisier problem: at the default 0.5 threshold the final model finds only 52 % of true cases
(recall 0.519), the tuned threshold lifts that to 0.889 but halves precision to 0.545. Note the test recall (0.889) lands
just under the 90 % target because the threshold was set on training OOF predictions: that is the honest cost of not tuning on the test set.

**Calibration.** For Logistic Regression the OOF Brier scores before / after calibration are almost identical
(heart 0.1444 → 0.1439 isotonic; diabetes 0.1508 → 0.1508 sigmoid), i.e. logistic regression was already well
calibrated. The calibration step matters more for RF / XGBoost / SVM (see `results/*_calibration.png`).
Platt vs isotonic is chosen automatically by lowest OOF Brier.

Plots: `results/{heart,diabetes}_roc_pr.png`, `_calibration.png`, `_threshold.png`.

## 5. The Explanation Reliability Score (ERS)  - main contribution

**Idea.** An explanation deserves trust only if (1) independent methods agree on which features matter for *this*
patient and (2) the explanation would not change if we re-ran the stochastic method or re-trained the model on slightly different data.

**Per prediction we compute**
* three attribution vectors over the original features: **SHAP**, **LIME** (mean of R = 5 reseeded runs) and **local permutation importance** (replace one feature by values from 50 real training patients, measure how far the predicted risk moves; this is a local adaptation of the usual global permutation importance);
* 5 further **LIME re-runs** (different random seeds) and 5 **bootstrap-refit models** (same hyper-parameters, trained on resampled data), each with its own SHAP vector.

**Formula** (all weights are named constants at the top of `src/ers.py`; none were tuned on the data)

```
pair_score(a, b)   = 0.5 * max(0, Spearman(|a|, |b|))  +  0.5 * |top3(a) ∩ top3(b)| / 3
consistency(vecs)  = mean of pair_score over all pairs of vectors

Agreement          = consistency([SHAP, LIME, Permutation])
LIME stability     = consistency([LIME run 1 .. run 5])
Model stability    = consistency([SHAP of bootstrap model 1 .. 5])
Stability          = 0.5 * LIME stability + 0.5 * Model stability

ERS                = 100 * ( 0.5 * Agreement + 0.5 * Stability )        in [0, 100]
```

**Badge (High / Medium / Low).** Relative to the model's own typical ERS: the 33rd and 67th percentile of the ERS of 150 random
**training** patients (`results/*_ers_cutoffs.json`). Low = least reliable third. This is deliberate: with fixed cut-offs
(70 / 40) on heart data 99-100 % of patients came out "High" for LR and RF (87 % for XGBoost), so the badge carried almost no information. The 0-100 number itself stays absolute.
In the app a Low badge shows *"Explanations disagree for this patient, interpret with caution"* and names the weakest of the three checks.

### Experiment E - does the ERS flag the cases the model gets wrong? (the research finding)

On the whole hold-out test set, for LR / RF / XGBoost on both datasets (`results/*_ers_summary.json`, plots `results/*_ers_vs_correctness.png`).
"Wrong" = misclassified at the tuned threshold. AUROC is for detecting wrong predictions from *low* ERS (0.5 = useless).

**1. How often do the three methods agree?** Mostly on the broad picture, much less on the exact ranking.

| dataset / model | all 3 same #1 feature | all 3 share ≥ 2 of top-3 | mean Spearman: SHAP–LIME / SHAP–perm / LIME–perm | mean ERS |
|---|---|---|---|---|
| heart / LR | 59 % | 73 % | 0.82 / 0.96 / 0.83 | 85.6 |
| heart / RF | 45 % | 96 % | 0.87 / 0.93 / 0.82 | 87.3 |
| heart / XGBoost | 61 % | **45 %** | 0.75 / 0.90 / 0.72 | **77.2** |
| diabetes / LR | 75 % | 95 % | 0.87 / 0.96 / 0.87 | 85.7 |
| diabetes / RF | 64 % | 88 % | 0.80 / 0.90 / 0.73 | 81.1 |
| diabetes / XGBoost | 74 % | 90 % | 0.86 / 0.94 / 0.84 | 82.2 |

SHAP and permutation importance agree closely; LIME is the outlier. The boosted-tree model on heart data has clearly the
lowest agreement (only 45 % of patients have ≥ 2 shared top-3 features), so trust in an explanation depends on the model.

**2. Are low-ERS cases the ones the model gets wrong?** Largely **no**.

| dataset / model | mean ERS correct vs wrong | Mann-Whitney p (wrong lower) | AUROC of low ERS → error | AUROC of baseline "close to threshold" | error rate High / Medium / Low |
|---|---|---|---|---|---|
| heart / LR | 85.6 vs 85.5 | 0.46 | 0.51 | 0.74 | 19 % / 16 % / 21 % |
| heart / RF | 87.2 vs 87.7 | 0.77 | 0.46 | 0.72 | 18 % / 29 % / 10 % |
| heart / XGBoost | 77.1 vs 77.6 | 0.50 | 0.50 | 0.73 | 19 % / 16 % / 22 % |
| diabetes / LR | 85.6 vs 86.0 | 0.56 | 0.49 | 0.51 | 31 % / 29 % / 30 % |
| diabetes / RF | 82.6 vs 78.0 | **0.015** | **0.61** | 0.47 | 28 % / 28 % / **43 %** |
| diabetes / XGBoost | 82.8 vs 80.8 | 0.10 | 0.56 | 0.50 | 26 % / 33 % / 33 % |

**Honest reading.** In five of six settings the ERS carries essentially no information about whether the prediction is
correct (AUROC 0.46-0.56). Only diabetes / Random Forest shows a modest signal (AUROC 0.61, error rate 43 % in the Low
group vs 28 %), and with six comparisons that p-value (0.015) does not survive a Bonferroni correction (0.015 × 6 ≈ 0.09).
On heart data a simple baseline, "how close is the probability to the threshold", detects errors far better (AUROC ≈ 0.73) than the ERS does.
**Conclusion: the ERS measures the quality of the explanation, not the correctness of the prediction; the two are
largely independent in this data.** That is itself useful: a confident-and-correct prediction can still have an explanation
that should not be trusted, and the ERS is the only thing that tells you. Do not advertise the ERS as an error detector.
(Weight sensitivity of the AUROC to the 50/50 Agreement/Stability split is in each summary JSON.)

## 6. Other contributions

**B. Counterfactuals (DiCE).** Only modifiable factors may change: heart = resting BP, cholesterol, fasting-blood-sugar flag,
max heart rate, exercise ST depression; diabetes = BMI, glucose, diastolic BP. **Age, sex and every other field are frozen**
(DiCE `features_to_vary`, plus a unit test). Risk factors can only move *down* towards a healthy bound, max heart rate only up and never above 220 − age.
"Lower risk" means *below the tuned threshold*; each counterfactual is rounded and **re-scored with the real model** before it is shown.
Example (diabetes, risk 23 % vs threshold 22 %): *glucose 117 → 110* gives 19 %; *BMI 32.4 → 25.5 and glucose 117 → 99* gives 8 %.
Limit: on heart data most flagged patients are driven by clinical findings (chest-pain type, ST slope) that are not modifiable;
only 22 of 121 flagged test patients can get below the threshold even with every modifiable factor at its best allowed value,
and the app says so instead of inventing advice.

**C. Plain-language explanation.** `explain.plain_language` turns the SHAP values into 2-3 sentences (risk vs threshold, strongest risk-raising factors, protective factors).

**D. Fairness and generalisation** (`results/*_fairness_subgroups.csv/png`, `results/heart_cross_source.csv/png`).
* *Subgroups* (out-of-fold predictions for the whole dataset, tuned threshold). Heart: recall is **0.92 for men but only 0.70 for women** (false-negative rate 30 % vs 8 %; women are 21 % of the data with 26 % prevalence vs 63 % for men), and 0.80 for patients under 45 (FNR 20 %) vs 0.88-0.98 for older groups. The ≥ 90 % recall promise therefore **does not hold for women** in this model. Diabetes (Pima is female-only): recall is 0.89 for under-45s; the older groups are tiny (n = 13 for 65+) and cannot support conclusions.
* *Cross-source* (heart only; Statlog is a subset of Cleveland so it is not an independent source). Train on Cleveland, threshold re-derived on Cleveland OOF:

| test hospital | prevalence | ROC-AUC | in-source ROC-AUC | AUC drop | recall | precision |
|---|---|---|---|---|---|---|
| Hungarian | 36 % | 0.887 | 0.875 | -0.011 (none) | 0.943 | 0.505 |
| Switzerland | 94 % | 0.755 | 0.875 | **0.120** | 0.939 | 0.947 |
| VA Long Beach | 74 % | 0.741 | 0.875 | **0.134** | 1.000 | 0.755 |

  The ranking ability (AUC) drops by 0.12-0.13 on two of three new hospitals; leave-one-source-out gives the same picture
  (drops 0.13-0.14 for Switzerland and VA, none for Cleveland / Hungarian). Plain accuracy is misleading here because
  prevalence differs hugely between hospitals (Switzerland is 94 % positive), so look at AUC and precision, not accuracy.
  The model should not be assumed to transfer to a new hospital without re-validation.

## 7. App

`streamlit run app.py`. Sidebar: choose Heart or Diabetes, a model (LR / RF / XGBoost), and the patient values (ranges from
`src/data.py`; "unknown" boxes for values that are often missing). Output tabs:
* **Assessment**: calibrated risk %, risk level using the tuned threshold, SHAP waterfall, plain-language text, ERS badge with the agreement chart and its components, counterfactual suggestions, medical disclaimer.
* **Compare methods**: SHAP vs LIME vs local permutation side by side, top-3 table, pairwise Spearman / overlap, and global permutation importance.
* **Model & experiments**: all tables and plots above.

The SHAP waterfall explains the model's raw (uncalibrated) score; the headline risk is its calibrated probability. A first
assessment takes a few seconds (LIME ×5, bootstrap models, DiCE).

## 8. Limitations

* **Small data.** 918 and 768 patients, one hold-out split of 184 / 154 patients: all test numbers have wide uncertainty (e.g. ±0.05 AUC between CV folds). No external clinical validation.
* **Old, specific cohorts.** UCI hospitals from the 1980s-90s; Pima is women of one population only. "Heart disease" = any UCI `num > 0`. Heart has 33 % missing `ST_Slope` and 22 % missing cholesterol (imputed inside the pipeline).
* **Models are within noise of each other** on CV, so "the best model" is a weak statement; the explainable linear model is as good as the others.
* **ERS design choices** (top-3, 5 LIME runs, 5 bootstrap models, 50 permutation draws, equal weights, relative badge cut-offs from 150 training patients) are reasonable but not uniquely right. It needs roughly a dozen explanation runs per patient, so it is slower than a single SHAP call (a few seconds per patient in the app). The local-permutation attribution is my adaptation, not a standard method. LIME is run in the transformed (one-hot) space and summed back.
* **ERS is not an error detector** (see Experiment E), and the experiment itself is small (35-50 errors per model).
* **Counterfactuals are associational**: they show what the model links to lower risk, not what treatment would do. Which heart fields are "modifiable" (ST depression, max heart rate) is a judgement call. Many flagged heart patients have no feasible counterfactual.
* **Fairness**: heart recall for women (0.70) and under-45s (0.80) is below the 90 % target; subgroup sizes are small and no mitigation was attempted.
* **Calibration and threshold** come from the same small training set; the test recall of the diabetes model (0.889) is slightly below the 90 % target.

## 9. Future work

* Repeated / nested cross-validation and confidence intervals for every number; external validation on other cohorts (e.g. NHANES / UK Biobank for diabetes).
* Sex-aware thresholds or re-weighting to close the heart recall gap for women; test with a sex-balanced training set.
* Improve the ERS: weights learned against human-rated explanation quality, a weighted rank correlation that emphasises top features, conformal-style coverage, and a version for deep models.
* Test whether the ERS predicts *explanation faithfulness* (deletion / insertion curves), which is what it is designed to measure, instead of prediction correctness.
* Richer, clinically reviewed definition of "actionable" features with clinician input; causal rather than associational counterfactuals.



*Related ideas to read and cite properly before the viva (check the exact references yourself):* "The Disagreement Problem in Explainable Machine Learning" (Krishna et al., 2022) and Alvarez-Melis & Jaakkola, "On the Robustness of Interpretability Methods" (2018); DiCE (Mothilal et al., 2020); SHAP (Lundberg & Lee, 2017); LIME (Ribeiro et al., 2016).
