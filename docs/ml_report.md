# Machine Learning Summary

## Objective

Predict whether a customer will subscribe to a term deposit (`deposit`), using customer demographics,
account details, and campaign contact history, in order to help the bank prioritize which customers to
call.

## Feature Set

**Included:** age, job, marital, education, default, balance, housing, loan, contact, day, month,
days_since_last_contact, previous, poutcome, was_previously_contacted.

**Deliberately excluded, with reasoning:**

- `duration` — length of the current call, only known _after_ the call happens. Including it would leak
  the outcome into the input (a long call is almost always a "yes"), producing an artificially high
  accuracy that couldn't be used to decide who to call _before_ the conversation happens.
- `campaign` — number of contacts made this round. Excluded as a precaution, since the final count for a
  given customer may not be fully known until contact attempts stop (e.g. once they say yes), a milder
  version of the same leakage concern as `duration`.
- `balance_tier` — a binned version of `balance`, already represented by the raw `balance` column; keeping
  both would be redundant.

**Categorical columns** (`job`, `marital`, `education`, `contact`, `month`, `poutcome`) were one-hot encoded.

**Missing values:** `days_since_last_contact` contains `NaN` for customers with no previous contact
(`pdays = -1` originally). Two versions of the training data were prepared:

- **Imputed version** (median-filled `days_since_last_contact`) — used for Logistic Regression and Random
  Forest, since these cannot handle `NaN` natively.
- **Raw version** (`NaN`s left in place) — used for XGBoost, which handles missing values natively and can
  treat "missing" itself as a signal, appropriate here since missingness genuinely means "never contacted,"
  not a random gap.

The median for imputation was calculated from the training set only, then applied to the test set, to avoid
leaking test-set information into training.

## Train/Test Split

80/20 split, stratified on `deposit` to preserve the original class balance (52.6% / 47.4%) in both sets.
Confirmed via `Counter(y_train)` and `Counter(y_test)`, both landed within a fraction of a percent of the
original ratio.

## Models Built

| Model                   | Recall | Precision | F1   | Accuracy | ROC-AUC    |
| ----------------------- | ------ | --------- | ---- | -------- | ---------- |
| Logistic Regression     | 0.56   | 0.74      | 0.64 | 0.70     | 0.75       |
| Random Forest (default) | 0.63   | 0.74      | 0.68 | 0.72     | 0.76       |
| Random Forest (tuned)   | 0.60   | 0.76      | 0.67 | 0.72     | 0.78       |
| XGBoost (default)       | 0.59   | 0.77      | 0.67 | 0.72     | **0.7853** |
| XGBoost (tuned)         | 0.61   | 0.74      | 0.67 | 0.71     | 0.77       |

Hyperparameter tuning (`RandomizedSearchCV`, `scoring='roc_auc'`, 5-fold cross-validation) was applied to
both Random Forest and XGBoost, tuned only against the training set, never the test set.

## Selected Model: XGBoost (default)

Chosen for the best ROC-AUC of all five models (0.7853), narrowly ahead of Random Forest (tuned) at 0.7819.
The **untuned, default** XGBoost model outperforms every tuned model in the
comparison. This reinforces a finding; for this dataset,
algorithm choice mattered more than hyperparameter tuning, and tuning produced small, inconsistent gains
rather than a clear, reliable improvement.

All five models converged within a fairly tight performance band (accuracy 0.70-0.72, ROC-AUC 0.75-0.79).
This suggests the available features carry a real but limited amount of predictive signal, consistent with
the strongest EDA finding (`poutcome_success`) already capturing much of the available signal on its own.

## Key Predictive Drivers

From Logistic Regression's odds ratios and XGBoost's SHAP values:

- **`poutcome_success`** — the single strongest predictor across models. A customer who subscribed in a
  previous campaign is substantially more likely to subscribe again (86% increase in odds, per the logistic
  regression coefficient).
- **Timing** (`month_mar`, `month_dec`, `month_oct`, `month_sep`) — certain months show meaningfully higher
  odds of conversion, consistent with the seasonal pattern found during EDA.
- **`education_tertiary`**, **`balance`**, **`job_retired`**, **`job_student`** — moderate positive
  contributors.

## Limitations

- `duration` and `campaign` were excluded to avoid leakage; a model including them would likely show
  higher accuracy, but would not be usable for real-world prioritization before a call takes place.
- No customer ID exists in this dataset, so the model cannot account for a customer's full multi-campaign
  history beyond what `previous`, `poutcome`, and `was_previously_contacted` capture.
- Performance across all models sits in a similar range; further gains would likely require additional
  features not present in this dataset (e.g. more granular financial history) rather than further tuning.
