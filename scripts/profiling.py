"""
profiling.py

A reusable toolkit for data profiling — understanding your dataset
before cleaning or modelling begins.

Import into any project with:

    from profiling import *

Suggested order of use:
1. summarize                                              -> SHAPE & TYPES
2. check_missing_values, check_duplicates,
   check_constant_columns, check_unique_values,
   check_dtypes                                           -> DATA QUALITY
3. check_skewness, check_outliers_iqr, detect_outliers   -> DISTRIBUTIONS & OUTLIERS
4. apply_log_transform                                    -> TRANSFORM
5. plot_distribution, plot_before_after_log,
   plot_density                                           -> VISUALIZE DISTRIBUTIONS
5. plot_correlation, plot_heatmap                         -> RELATIONSHIPS

This file is for profiling only — cleaning functions live in cleaning_utils.py,
EDA and chart functions live in eda_utils.py.

Default color  : #2E86AB (steel blue)
Default theme  : theme_minimal()
All charts support:
    figure_size=(width, height)  -> resize any chart
    title=, subtitle=            -> override auto-generated labels
    show_values=True             -> display value labels where relevant

Dependencies:
    pip install plotnine seaborn matplotlib pandas numpy scipy
"""

import pandas as pd
from IPython.display import display as ipy_display
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import (
    ggplot, aes,
    geom_histogram, geom_density,
    geom_line, geom_tile, geom_text,
    geom_vline,
    facet_wrap,
    scale_fill_gradient2,
    scale_fill_manual,
    theme_minimal, theme, labs,
    element_text,
)

DEFAULT_COLOR  = '#2E86AB'
DEFAULT_FIGURE = (10, 6)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _apply_facet(plot, facet_by):
    return plot + facet_wrap(facet_by) if facet_by else plot

def _labs(title, subtitle, x, y):
    return labs(title=title, subtitle=subtitle, x=x, y=y)

def _figsize(figure_size):
    return theme(figure_size=figure_size)


# ─────────────────────────────────────────────
# SHAPE & TYPES
# ─────────────────────────────────────────────

def summarize(df):
    """
    Print dataset shape, column data types, and descriptive statistics.
    The starting point for any profiling session.

    Example:
        summarize(df)
    """
    print("-" * 50)
    print(f"DATASET SHAPE")
    print(f"{df.shape[0]:,} rows x {df.shape[1]} columns")
    print("-" * 50)
    print("\n--- DATA TYPES ---")
    print(df.dtypes)
    print("\n--- DESCRIPTIVE STATISTICS ---")
    print(df.describe(include='all').round(2))


def check_dtypes(df):
    """
    Print column data types.
    Useful for spotting numbers stored as text, dates stored as strings, etc.

    Example:
        check_dtypes(df)
    """
    print(df.dtypes)


# ─────────────────────────────────────────────
# DATA QUALITY
# ─────────────────────────────────────────────

def check_missing_values(df, threshold=5.0):
    """
    Print null count and percentage per column.
    Flags columns above the threshold as needing attention.

    threshold: percentage above which a column is flagged (default 5.0)

    Examples:
        check_missing_values(df)
        check_missing_values(df, threshold=10.0)
    """
    null_count = df.isnull().sum()
    null_pct   = (null_count / len(df)) * 100
    null_df    = pd.DataFrame({
        'null_count': null_count,
        'null_pct':   null_pct.round(2)
    }).sort_values('null_pct', ascending=False)

    null_df = null_df[null_df['null_count'] > 0]

    if null_df.empty:
        print("No missing values found. Dataset is complete.")
        return null_df

    print(f"{'Column':<30} {'Null Count':>12} {'Null %':>10}  {'Flag'}")
    print("-" * 65)
    for col, row in null_df.iterrows():
        flag = "⚠ ACTION NEEDED" if row['null_pct'] > threshold else ""
        print(f"{col:<30} {int(row['null_count']):>12,} {row['null_pct']:>9.2f}%  {flag}")

    print(f"\n{len(null_df)} column(s) have missing values.")
    flagged = null_df[null_df['null_pct'] > threshold]
    if not flagged.empty:
        print(f"{len(flagged)} column(s) exceed the {threshold}% threshold and need attention.")
    return null_df


def check_duplicates(df, show_rows=False):
    """
    Print duplicate row count and percentage.
    show_rows=True displays the actual duplicate rows for quick inspection.
    For full side-by-side review before deleting, use show_duplicates(df)
    from cleaning_utils.

    Examples:
        check_duplicates(df)
        check_duplicates(df, show_rows=True)
    """
    dup_count = df.duplicated().sum()
    dup_pct   = round((dup_count / len(df)) * 100, 2)

    print(f"Total rows      : {len(df):,}")
    print(f"Duplicate rows  : {dup_count:,} ({dup_pct}%)")

    if dup_count == 0:
        print("No duplicates found.")
    else:
        print("Review with show_duplicates(df) from cleaning_utils before removing.")
        if show_rows:
            print("\n--- Duplicate Rows ---")
            print(df[df.duplicated(keep=False)])
    return dup_count


def check_unique_values(df, column):
    """
    Print value counts for a column.
    Good for spotting typos, inconsistent labels, or unexpected categories.

    Example:
        check_unique_values(df, 'country')
    """
    counts = df[column].value_counts()
    print(f"'{column}' — {counts.shape[0]} unique values:\n")
    print(counts)


def check_constant_columns(df):
    """
    Identify columns where every value is the same.
    These carry no analytical value and can usually be dropped.

    Example:
        check_constant_columns(df)
    """
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
    if constant_cols:
        print(f"Constant columns found: {constant_cols}")
    else:
        print("No constant columns found.")
    return constant_cols


# ─────────────────────────────────────────────
# DISTRIBUTIONS & OUTLIERS
# ─────────────────────────────────────────────

def check_skewness(df, plot=False, bins=30, fill=DEFAULT_COLOR, color='black',
                   figure_size=DEFAULT_FIGURE):
    """
    Print skewness for all numeric columns with direction and plain-English implication.
    plot=True also shows a histogram per column so you can see the shape.
    bins: number of histogram bars — your manual call based on what looks right.

    Skewness interpretation:
        > 1          : Highly right-skewed. Most values low, few very high outliers.
                       Common in price, income, sales. Consider log transformation.
        0.5 to 1     : Moderately right-skewed. Worth noting before modelling.
       -0.5 to 0.5   : Roughly symmetric. No transformation needed.
       -1 to -0.5    : Moderately left-skewed. Worth noting before modelling.
        < -1         : Highly left-skewed. Most values high, few very low outliers.

    Examples:
        check_skewness(df)
        check_skewness(df, plot=True, bins=20)
        check_skewness(df, plot=True, bins=50, fill='#E63946', figure_size=(12, 5))
    """
    def is_binary(col):
        """Returns True if column contains only two unique values (e.g. 0 and 1)."""
        return df[col].dropna().nunique() == 2

    def interpret(val):
        if val > 1:
            return "Highly right-skewed -> most values low with high outliers. Consider log transformation."
        elif val > 0.5:
            return "Moderately right-skewed -> slight positive tail. Worth noting before modelling."
        elif val >= -0.5:
            return "Roughly symmetric -> close to normal. No transformation needed."
        elif val >= -1:
            return "Moderately left-skewed -> slight negative tail. Worth noting before modelling."
        else:
            return "Highly left-skewed -> most values high with low outliers. Consider transformation."

    numeric_cols = df.select_dtypes(include='number').columns
    skew = df[numeric_cols].skew().sort_values(ascending=False)

    print("--- Skewness (numeric columns) ---\n")
    for col, val in skew.items():
        if is_binary(col):
            unique_vals = sorted(df[col].dropna().unique().tolist())
            print(f"{col}")
            print(f"  Skewness : {val:.4f}")
            print(f"  Type     : Binary column {unique_vals} — skewness reflects class imbalance,")
            print(f"             not distribution shape. Do not log-transform.")
            print(f"             Imbalance: {df[col].value_counts().to_dict()}\n")
        else:
            direction = 'Right (positive)' if val > 0 else 'Left (negative)' if val < 0 else 'Symmetric'
            print(f"{col}")
            print(f"  Skewness : {val:.4f}")
            print(f"  Direction: {direction}")
            print(f"  Meaning  : {interpret(val)}\n")

    if plot:
        for col in numeric_cols:
            p = (
                ggplot(df, aes(x=col))
                + geom_histogram(bins=bins, fill=fill, color=color)
                + labs(
                    title=f'{col} — Skewness: {skew[col]:.4f}',
                    subtitle=f'Mean: {df[col].mean():.2f} | Median: {df[col].median():.2f}',
                    x=col, y='Count'
                )
                + theme_minimal()
                + _figsize(figure_size)
            )
            ipy_display(p.draw())
    plt.close()
    return skew


def check_outliers_iqr(df, column):
    """
    Flag and return rows outside 1.5x IQR for a numeric column.
    Use detect_outliers() for a more detailed report with method options.
    Binary columns (only two unique values) are skipped automatically
    since IQR outlier detection is not meaningful for them.

    Example:
        outliers = check_outliers_iqr(df, 'unit_price')
    """
    unique_vals = df[column].dropna().nunique()
    if unique_vals == 2:
        vals = sorted(df[column].dropna().unique().tolist())
        print(f"{column}")
        print(f"  Type     : Binary column {vals} — IQR outlier detection does not apply.")
        print(f"             Every minority class value would be flagged as an outlier,")
        print(f"             which is not meaningful. Class distribution: {df[column].value_counts().to_dict()}")
        print(f"             If class imbalance is a concern, address it during modelling")
        print(f"             (e.g. SMOTE, class_weight='balanced', stratified sampling).")
        return df[df[column].isna()]  # return empty df of same structure

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower) | (df[column] > upper)]
    print(f"{column}: {len(outliers)} potential outliers found "
          f"(fence: [{lower:.2f}, {upper:.2f}])")
    return outliers


def detect_outliers(df, column, method='iqr', threshold=3.0):
    """
    Detailed outlier report for a numeric column.
    Binary columns (only two unique values) are skipped automatically
    since outlier detection is not meaningful for them.

    method    : 'iqr'    — interquartile range fence (default, robust to skew)
                'zscore' — standard deviations from the mean (assumes normality)
    threshold : for zscore only — number of std deviations to use (default 3.0)

    Returns a dataframe of the outlier rows for your inspection.

    Examples:
        detect_outliers(df, 'unit_price')
        detect_outliers(df, 'sales', method='zscore', threshold=2.5)
    """
    unique_vals = df[column].dropna().nunique()
    if unique_vals == 2:
        vals = sorted(df[column].dropna().unique().tolist())
        print(f"Column   : {column}")
        print(f"Type     : Binary column {vals} — outlier detection does not apply.")
        print(f"           Class distribution: {df[column].value_counts().to_dict()}")
        print(f"           If class imbalance is a concern, address during modelling")
        print(f"           (e.g. SMOTE, class_weight='balanced', stratified sampling).")
        return df[df[column].isna()]  # return empty df of same structure

    if method == 'iqr':
        q1    = df[column].quantile(0.25)
        q3    = df[column].quantile(0.75)
        iqr   = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers    = df[(df[column] < lower) | (df[column] > upper)]
        method_desc = f'IQR fence: [{lower:.2f}, {upper:.2f}]'

    elif method == 'zscore':
        mean     = df[column].mean()
        std      = df[column].std()
        z        = (df[column] - mean) / std
        outliers = df[z.abs() > threshold]
        method_desc = f'Z-score threshold: ±{threshold}'

    else:
        raise ValueError("method must be 'iqr' or 'zscore'")

    pct = round(len(outliers) / len(df) * 100, 2)
    print(f"Column   : {column}")
    print(f"Method   : {method_desc}")
    print(f"Outliers : {len(outliers):,} rows ({pct}% of data)")
    if len(outliers) > 0:
        print(f"Min value: {outliers[column].min():.2f}")
        print(f"Max value: {outliers[column].max():.2f}")
    return outliers


# ─────────────────────────────────────────────
# VISUALIZE DISTRIBUTIONS
# ─────────────────────────────────────────────

def plot_distribution(df, column, fill=DEFAULT_COLOR, color='black',
                      bins=30, title=None, subtitle=None, facet_by=None,
                      show_mean=True, show_median=True, show_bell=False,
                      figure_size=DEFAULT_FIGURE):
    """
    Histogram with optional mean line (red), median line (green), and bell curve.

    show_mean  : red dashed vertical line at the mean (default True)
    show_median: green dashed vertical line at the median (default True)
    show_bell  : overlays a normal distribution curve to compare your data
                 against what a perfect normal distribution would look like (default False)
    figure_size: (width, height) e.g. figure_size=(14, 6)

    Examples:
        plot_distribution(df, 'unit_price')
        plot_distribution(df, 'unit_price', show_bell=True)
        plot_distribution(df, 'unit_price', bins=50, figure_size=(14, 5))
        plot_distribution(df, 'unit_price', facet_by='country')
    """
    from scipy.stats import norm

    mean_val   = df[column].mean()
    median_val = df[column].median()
    skew_val   = df[column].skew()

    # Binary column guard
    if df[column].dropna().nunique() == 2:
        vals = sorted(df[column].dropna().unique().tolist())
        print(f"Warning: '{column}' is a binary column {vals}.")
        print(f"A histogram is not meaningful here — use plot_bar(df, '{column}') instead,")
        print(f"which shows the class distribution as a proper bar chart.")
        print(f"Class distribution: {df[column].value_counts().to_dict()}")
        return None

    title    = title or f'Distribution of {column}'
    subtitle = subtitle or (
        f"Mean: {mean_val:.2f} | "
        f"Median: {median_val:.2f} | "
        f"Skewness: {skew_val:.2f}"
    )

    plot = (
        ggplot(df, aes(x=column))
        + geom_histogram(bins=bins, fill=fill, color=color, alpha=0.8)
        + _labs(title, subtitle, column, 'Count')
        + theme_minimal()
        + _figsize(figure_size)
    )

    if show_mean:
        plot = plot + geom_vline(
            aes(xintercept=mean_val),
            color='red', linetype='dashed', size=1
        )

    if show_median:
        plot = plot + geom_vline(
            aes(xintercept=median_val),
            color='green', linetype='dashed', size=1
        )

    if show_bell:
        col_data  = df[column].dropna()
        x_range   = np.linspace(col_data.min(), col_data.max(), 200)
        bin_width = (col_data.max() - col_data.min()) / bins
        y_bell    = norm.pdf(x_range, mean_val, col_data.std()) * len(col_data) * bin_width
        bell_df   = pd.DataFrame({column: x_range, 'y': y_bell})
        plot = plot + geom_line(
            data=bell_df,
            mapping=aes(x=column, y='y'),
            color='#E63946', size=1
        )

    plot = _apply_facet(plot, facet_by)
    ipy_display(plot.draw())
    plt.close()
    return None


def plot_density(df, column, fill=DEFAULT_COLOR, color='black', alpha=0.4,
                 title=None, subtitle=None, facet_by=None,
                 figure_size=DEFAULT_FIGURE):
    """
    Smooth density (KDE) curve for a numeric column.
    Shows the shape of the distribution without the bin-size sensitivity of a histogram.

    Examples:
        plot_density(df, 'unit_price')
        plot_density(df, 'unit_price', fill='#E63946', figure_size=(12, 5))
    """
    # Binary column guard
    if df[column].dropna().nunique() == 2:
        vals = sorted(df[column].dropna().unique().tolist())
        print(f"Warning: '{column}' is a binary column {vals}.")
        print(f"A density curve is not meaningful here — use plot_bar(df, '{column}') instead.")
        print(f"Class distribution: {df[column].value_counts().to_dict()}")
        return None

    title    = title or f'Density of {column}'
    subtitle = subtitle or (
        f"Mean: {df[column].mean():.2f} | Std: {df[column].std():.2f}"
    )

    plot = (
        ggplot(df, aes(x=column))
        + geom_density(fill=fill, color=color, alpha=alpha)
        + _labs(title, subtitle, column, 'Density')
        + theme_minimal()
        + _figsize(figure_size)
    )
    plot = _apply_facet(plot, facet_by)
    ipy_display(plot.draw())
    plt.close()
    return None



def apply_log_transform(df, columns, handle_zeros='shift', drop_original=False):
    """
    Apply log transformation to one or more numeric columns flagged as
    highly skewed by check_skewness().

    Only use on continuous numeric columns with skewness > 1 or < -1.
    Never use on binary columns (0/1) or columns with negative values
    without first understanding what those negatives mean.

    columns      : a single column name or list of column names to transform.
    handle_zeros : what to do if a column contains zeros, since log(0) = -inf.
                   'shift'  — adds 1 before transforming: log(x + 1). Safe default.
                              preserves zero as log(1) = 0 and keeps all rows.
                   'drop'   — removes rows where the column is zero before transforming.
                              use only if zeros are data errors, not real values.
                   'ignore' — applies log directly. zeros become -inf. use only
                              if you are certain the column has no zeros.
    drop_original: if True, removes the original column after creating the log version.
                   default False keeps both so you can compare.

    Creates a new column named log_{column} for each column transformed.
    Run plot_before_after_log(df, column) after this to visually confirm
    the transformation improved the distribution shape.

    Examples:
        df = apply_log_transform(df, 'sales')
        df = apply_log_transform(df, ['sales', 'profit'], handle_zeros='shift')
        df = apply_log_transform(df, 'bmi', drop_original=True)
    """
    df = df.copy()

    if isinstance(columns, str):
        columns = [columns]

    for col in columns:
        # Binary column guard
        if df[col].dropna().nunique() == 2:
            vals = sorted(df[col].dropna().unique().tolist())
            print(f"Skipping '{col}': binary column {vals} — log transformation does not apply.")
            continue

        # Negative value warning
        if (df[col] < 0).any():
            neg_count = (df[col] < 0).sum()
            print(f"Warning: '{col}' contains {neg_count} negative values.")
            print(f"  Log transformation is not defined for negative numbers.")
            print(f"  If negatives represent losses (e.g. profit), log transformation")
            print(f"  is not appropriate — consider leaving this column untransformed")
            print(f"  or using a different technique (e.g. Yeo-Johnson via PowerTransformer).")
            continue

        zero_count = (df[col] == 0).sum()
        log_col = f'log_{col}'

        if zero_count > 0:
            if handle_zeros == 'shift':
                df[log_col] = np.log1p(df[col])  # log(x + 1)
                print(f"'{col}' -> '{log_col}': log(x+1) applied "
                      f"({zero_count} zeros shifted safely).")
            elif handle_zeros == 'drop':
                df = df[df[col] > 0].copy()
                df[log_col] = np.log(df[col])
                print(f"'{col}' -> '{log_col}': log(x) applied after dropping "
                      f"{zero_count} zero rows.")
            elif handle_zeros == 'ignore':
                df[log_col] = np.log(df[col].replace(0, np.nan))
                print(f"'{col}' -> '{log_col}': log(x) applied. "
                      f"{zero_count} zeros became NaN.")
        else:
            df[log_col] = np.log(df[col])
            print(f"'{col}' -> '{log_col}': log(x) applied (no zeros found).")

        skew_before = df[col].skew()
        skew_after  = df[log_col].skew()
        improvement = abs(skew_before) - abs(skew_after)
        print(f"  Skewness before: {skew_before:.4f} -> after: {skew_after:.4f} "
              f"(improvement: {improvement:.4f})")

        if abs(skew_after) > abs(skew_before):
            print(f"  Note: log transformation made skewness worse for '{col}'.")
            print(f"  Consider leaving it untransformed or trying a different approach.")

        if drop_original:
            df = df.drop(columns=[col])
            print(f"  Original '{col}' column removed.")

        print()

    return df

def plot_before_after_log(df, column, bins=40, fill=DEFAULT_COLOR,
                           log_fill='#928e85', figure_size=(12, 4),
                           show_bell=False):
    """
    Side-by-side histogram comparing original vs log-transformed distribution
    using facet_wrap so both panels sit on the same row for direct comparison.

    Use when check_skewness() flags a column as highly right-skewed to confirm
    whether log transformation brings it closer to normal before modelling.

    show_bell  : overlay a normal distribution bell curve on both panels (default False).
                 Useful for visually confirming how close each distribution is to normal.
    figure_size: default (12, 4) — wide enough for two side-by-side panels.

    Examples:
        plot_before_after_log(df, 'sales')
        plot_before_after_log(df, 'bmi', bins=50)
        plot_before_after_log(df, 'bmi', show_bell=True)
        plot_before_after_log(df, 'unit_price', figure_size=(14, 5))
    """
    from scipy.stats import norm

    # Binary column guard
    if df[column].dropna().nunique() == 2:
        vals = sorted(df[column].dropna().unique().tolist())
        print(f"Warning: '{column}' is a binary column {vals}.")
        print(f"Log transformation and this plot do not apply to binary columns.")
        return None

    # Negative value guard
    if (df[column] < 0).any():
        print(f"Warning: '{column}' contains negative values.")
        print(f"Log transformation is not defined for negative numbers.")
        print(f"Consider leaving this column untransformed or using PowerTransformer.")
        return None

    skew_orig = df[column].skew()

    # Build log column using log1p to handle zeros safely
    log_col = f'log_{column}'
    plot_df = df[[column]].copy().dropna()
    plot_df[log_col] = np.log1p(plot_df[column])
    skew_log = plot_df[log_col].skew()

    orig_label = f'Original\nSkewness: {skew_orig:.4f}'
    log_label  = f'Log Transformed\nSkewness: {skew_log:.4f}'

    # Reshape to long format for facet_wrap
    original_df = pd.DataFrame({'value': plot_df[column],   'version': orig_label})
    log_df_long = pd.DataFrame({'value': plot_df[log_col],  'version': log_label})
    combined = pd.concat([original_df, log_df_long], ignore_index=True)

    # Lock panel order so Original always appears on the left
    combined['version'] = pd.Categorical(
        combined['version'],
        categories=[orig_label, log_label],
        ordered=True
    )

    improvement = abs(skew_orig) - abs(skew_log)
    verdict = "Transformation helped." if improvement > 0 else "Transformation did not improve skewness."

    p = (
        ggplot(combined, aes(x='value', fill='version'))
        + geom_histogram(bins=bins, color='black', show_legend=False, alpha=0.8)
        + facet_wrap('version', scales='free')
        + scale_fill_manual(values={orig_label: fill, log_label: log_fill})
        + labs(
            title=f'{column} — Before vs After Log Transformation',
            subtitle=f'Skewness: {skew_orig:.4f} → {skew_log:.4f} | {verdict}',
            x='Value', y='Count'
        )
        + theme_minimal()
        + _figsize(figure_size)
    )

    if show_bell:
        # Build bell curve for each panel scaled to histogram height
        bell_frames = []
        for vals, label, col_name in [
            (plot_df[column].values,  orig_label, column),
            (plot_df[log_col].values, log_label,  log_col)
        ]:
            x_range  = np.linspace(vals.min(), vals.max(), 200)
            bw       = (vals.max() - vals.min()) / bins
            y_bell   = norm.pdf(x_range, vals.mean(), vals.std()) * len(vals) * bw
            bell_frames.append(pd.DataFrame({
                'value'  : x_range,
                'y'      : y_bell,
                'version': label
            }))

        bell_combined = pd.concat(bell_frames, ignore_index=True)
        bell_combined['version'] = pd.Categorical(
            bell_combined['version'],
            categories=[orig_label, log_label],
            ordered=True
        )

        p = p + geom_line(
            data=bell_combined,
            mapping=aes(x='value', y='y', group='version'),
            color='#E63946', size=1, inherit_aes=False
        )

    ipy_display(p.draw())
    plt.close()
    return None    # prevents Jupyter auto-display from trying to render p a second time


# ─────────────────────────────────────────────
# RELATIONSHIPS
# ─────────────────────────────────────────────

def plot_correlation(df, title='Correlation Heatmap', subtitle=None,
                     figsize=(12, 8), cmap='coolwarm'):
    """
    Heatmap of correlations across all numeric columns.
    Values near  1 = strong positive relationship.
    Values near -1 = strong negative relationship.
    Values near  0 = little to no relationship.

    figsize: (width, height) — matplotlib tuple since seaborn handles this chart.
    cmap   : color scheme (default 'coolwarm': red = positive, blue = negative)

    Examples:
        plot_correlation(df)
        plot_correlation(df, figsize=(10, 6), cmap='viridis')
    """
    numeric_df = df.select_dtypes(include='number')
    corr       = numeric_df.corr()
    subtitle   = subtitle or f'{len(numeric_df.columns)} numeric columns analysed'

    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap=cmap,
                square=True, linewidths=0.5)
    plt.title(f'{title}\n{subtitle}', fontsize=13)
    plt.tight_layout()
    plt.show()
    return corr


def plot_heatmap(df, x_col, y_col, value_col, midpoint=0,
                 low_color='#E63946', mid_color='#f5f5f5', high_color='#62866c',
                 show_values=True, value_size=7,
                 title=None, subtitle=None, figure_size=(10, 8)):
    """
    Tile heatmap for any two categorical columns crossed against a numeric value.
    Red = low/negative, white = midpoint, green = high/positive.

    x_col      : column for x axis (e.g. 'region')
    y_col      : column for y axis (e.g. 'sub_category')
    value_col  : numeric column to encode as color (e.g. 'profit')
    midpoint   : value where the color transitions (default 0)
    show_values: display rounded numeric values inside each tile (default True)
    value_size : font size of tile labels (default 7)

    Examples:
        plot_heatmap(df, 'region', 'sub_category', 'profit')
        plot_heatmap(df, 'month', 'product', 'revenue', midpoint=5000)
        plot_heatmap(df, 'region', 'category', 'sales', show_values=False)
    """
    heatmap_df = (
        df.groupby([x_col, y_col])[value_col]
        .sum()
        .reset_index()
    )

    # Pre-round values into a label column so geom_text can reference it correctly
    label_col = f'{value_col}_label'
    heatmap_df[label_col] = heatmap_df[value_col].round(0).astype(int).astype(str)

    title    = title or f'{value_col} by {x_col} and {y_col}'
    subtitle = subtitle or (
        f'Red = low/negative | White = midpoint ({midpoint}) | Green = high/positive'
    )

    plot = (
        ggplot(heatmap_df, aes(x=x_col, y=y_col, fill=value_col))
        + geom_tile(color='white')
        + scale_fill_gradient2(
            low=low_color, mid=mid_color, high=high_color,
            midpoint=midpoint
        )
        + _labs(title, subtitle, x_col, y_col)
        + theme_minimal()
        + theme(
            axis_text_x=element_text(angle=45, hjust=1),
            axis_text_y=element_text(size=8)
        )
        + _figsize(figure_size)
    )

    if show_values:
        plot = plot + geom_text(aes(label=label_col), size=value_size)

    ipy_display(plot.draw())
    plt.close()
    return None