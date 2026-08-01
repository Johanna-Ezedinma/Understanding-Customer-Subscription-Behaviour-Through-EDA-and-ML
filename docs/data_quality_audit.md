# Bank Marketing Analysis

## Dataset Description

**Source:** [Bank Marketing Dataset – Kaggle](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset)

This dataset tracks a direct marketing campaign run by a Portuguese retail bank, where customers were
contacted by phone and asked to subscribe to a **term deposit** (a savings product where funds are locked
in for a fixed period in exchange for a set interest rate). Each row represents one customer contact event,
and whether that contact ultimately resulted in a subscription (`deposit`).

**Shape:** 11,162 rows × 17 columns. No true missing (`NaN`) values, no duplicate rows.

**Note:** there is no unique customer ID and no year column (only day-of-month and month name), so this
dataset can't distinguish repeat contacts with the same customer, and can only show within-year seasonal
patterns, not a true multi-year trend.

### Column Reference

| Column      | Description                                                                                                                                          | Type                  |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- |
| `age`       | Customer's age, in years                                                                                                                             | Numeric               |
| `job`       | Occupation category (12 categories, e.g. management, technician, student, retired)                                                                   | Categorical           |
| `marital`   | Marital status (married, single, divorced)                                                                                                           | Categorical           |
| `education` | Highest education level (primary, secondary, tertiary, unknown)                                                                                      | Categorical           |
| `default`   | Whether the customer currently has credit in default (yes/no)                                                                                        | Boolean               |
| `balance`   | Average yearly account balance (can be negative, i.e. overdrawn)                                                                                     | Numeric               |
| `housing`   | Whether the customer has a housing loan (yes/no)                                                                                                     | Boolean               |
| `loan`      | Whether the customer has a personal loan (yes/no)                                                                                                    | Boolean               |
| `contact`   | Contact communication method (cellular, telephone, **or `unknown`**)                                                                                 | Categorical           |
| `day`       | Day of the month of the last contact                                                                                                                 | Numeric               |
| `month`     | Month of the last contact (jan–dec)                                                                                                                  | Categorical (ordered) |
| `duration`  | Length of the last contact call, in seconds. **Only known after the call happens, treat as a leakage risk for any predictive use, descriptive only** | Numeric               |
| `campaign`  | Number of contacts made to this customer during this campaign                                                                                        | Numeric (count)       |
| `pdays`     | Days since last contact from a previous campaign; **`-1` is a sentinel meaning "never contacted before," not a real value**                          | Numeric               |
| `previous`  | Number of contacts made to this customer before this campaign                                                                                        | Numeric (count)       |
| `poutcome`  | Outcome of the previous campaign (success, failure, other, **or `unknown`**)                                                                         | Categorical           |
| `deposit`   | **Target variable** — did the customer subscribe to a term deposit? (yes/no)                                                                         | Boolean               |

---

# Data Quality Notes

- No Missing values
- No duplicates
- `contact` and `poutcome` both contain the text `"unknown"` which were confirmed to mean "not applicable" and will be left as is.
- `pdays` has `-1` values meaning "never contacted before," not a real day count.

> **Decision:** `pdays` was split into two clean fields rather than left as a single column mixing a
> sentinel value with real numbers: `was_previously_contacted` (a boolean flag, `True`/`False`), and
> `days_since_last_contact` (the real day count, kept only for customers who were genuinely contacted
> before; left as missing for everyone else, rather than filled with a misleading placeholder like 0).

- `balance` and `campaign` have some extreme values

> **Decision:** Extreme values in balance and campaign will be left as is since they represent real customers not errors
