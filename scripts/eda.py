"""
eda_utils.py

A reusable toolkit of Exploratory Data Analysis (EDA) functions.
Import into any project with:

    from eda_utils import *

Suggested order of use:
1. summarize, check_skewness                              -> OVERVIEW
2. plot_distribution, plot_before_after_log,
   plot_density, plot_boxplot, plot_violin,
   plot_bar, plot_highlight_bar, plot_ranked_bar,
   plot_bar_stacked, plot_bar_grouped,
   plot_faceted_bar, plot_lollipop                        -> SINGLE VARIABLE / CATEGORICAL
3. plot_line, plot_line_grouped, plot_line_bar,
   plot_area, plot_area_stacked                           -> TIME SERIES
4. plot_scatter, plot_scatter_trend,
   plot_scatter_grouped, plot_bubble,
   plot_linearity_check                                   -> RELATIONSHIPS
5. plot_heatmap, plot_correlation,
   plot_return_rate, plot_donut                           -> COMPOSITION & OPERATIONAL
6. summarize_groups, detect_outliers                      -> GROUPED ANALYSIS
7. plot_actual_vs_predicted,
   plot_feature_importance                                -> MODEL EVALUATION
8. auto_plot                                              -> SMART SELECTOR

Default color  : #2E86AB (steel blue)
Default theme  : theme_minimal()
All charts support:
    figure_size=(width, height)  -> resize any chart
    title=, subtitle=            -> override auto-generated labels
    flip=                        -> horizontal layout where relevant
    facet_by=                    -> wrap into panels by category where relevant
    y_col=                       -> pass pre-aggregated data to bar/lollipop/donut
    show_points=True             -> add data points on line charts (optional)
    line_size=                   -> control line thickness on line/area charts
    show_values=True             -> add data value labels on bar/lollipop/line charts

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
    geom_bar, geom_col,
    geom_boxplot, geom_violin,
    geom_point, geom_line, geom_area,
    geom_segment, geom_tile, geom_text,
    geom_hline, geom_abline, geom_smooth,
    geom_vline,
    facet_wrap,
    scale_fill_gradient2, scale_fill_manual,
    scale_x_continuous,
    theme_minimal, theme, labs,
    coord_flip, element_text, element_blank, element_line,  # add element_line here
    position_stack, position_dodge,
)

DEFAULT_COLOR  = '#62866c'
DEFAULT_FIGURE = (6, 4)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _apply_flip(plot, flip):
    return plot + coord_flip() if flip else plot

def _apply_facet(plot, facet_by):
    return plot + facet_wrap(facet_by) if facet_by else plot

def _labs(title, subtitle, x, y, x_label=None, y_label=None):
    return labs(
        title=title,
        subtitle=subtitle,
        x=x_label if x_label is not None else x,
        y=y_label if y_label is not None else y
    )

def _figsize(figure_size):
    return theme(
        figure_size=figure_size,
        # Left-align title and subtitle like Power BI
        plot_title=element_text(ha='left', size=12, weight='bold'),
        plot_subtitle=element_text(ha='left', size=9, margin={'t': 6, 'b': 10}),
        # Clean subtle dashed gridlines
        panel_grid_major=element_line(color='#e0e0e0', size=0.4, linetype='dashed'),
        panel_grid_minor=element_blank(),
    )


# ─────────────────────────────────────────────
# OVERVIEW
# ─────────────────────────────────────────────

def summarize(df):
    """
    Print shape, data types, and descriptive statistics.
    Run this first before any plotting.
    """
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")
    print("--- Data Types ---")
    print(df.dtypes)
    print("\n--- Descriptive Statistics ---")
    print(df.describe(include='all'


# ─────────────────────────────────────────────
# DISTRIBUTION
# ─────────────────────────────────────────────


def plot_boxplot(df, numeric_col, category_col=None, fill=DEFAULT_COLOR,
                 color='black', title=None, subtitle=None, flip=True,
                 facet_by=None, x_label=None, y_label=None,
                      figure_size=DEFAULT_FIGURE):
    """
    Boxplot for a numeric column, optionally grouped by a category.
    flip=True (default) shows horizontal layout.

    Examples:
        plot_boxplot(df, 'unit_price')
        plot_boxplot(df, 'unit_price', 'country')
        plot_boxplot(df, 'unit_price', 'country', figure_size=(12, 8))
    """
    # Binary column guard on numeric_col
    if df[numeric_col].dropna().nunique() == 2:
        vals = sorted(df[numeric_col].dropna().unique().tolist())
        print(f"Warning: '{numeric_col}' is a binary column {vals}.")
        print(f"A boxplot collapses to a line for binary columns — use plot_bar(df, '{numeric_col}') instead.")
        print(f"Class distribution: {df[numeric_col].value_counts().to_dict()}")
        return None

    if category_col:
        title    = title    or f'{numeric_col} by {category_col}'
        subtitle = subtitle or f'Median overall: {df[numeric_col].median():.2f}'
        plot = (
            ggplot(df, aes(x=category_col, y=numeric_col))
            + geom_boxplot(fill=fill, color=color)
            + _labs(title, subtitle, category_col, numeric_col, x_label=x_label, y_label=y_label)
            + theme_minimal()
            + theme(axis_text_y=element_text(size=8))
            + _figsize(figure_size)
        )
    else:
        title    = title    or f'Boxplot of {numeric_col}'
        subtitle = subtitle or (
            f"Median: {df[numeric_col].median():.2f} | "
            f"IQR: {df[numeric_col].quantile(0.75) - df[numeric_col].quantile(0.25):.2f}"
        )
        plot = (
            ggplot(df, aes(x=f'"{numeric_col}"', y=numeric_col))
            + geom_boxplot(fill=fill, color=color)
            + _labs(title, subtitle, '', numeric_col, x_label=x_label, y_label=y_label)
            + theme_minimal()
            + _figsize(figure_size)
        )
    plot = _apply_flip(plot, flip)
    plot = _apply_facet(plot, facet_by)
    ipy_display(plot.draw())
    plt.close()
    return None

# ─────────────────────────────────────────────
# CATEGORICAL
# ─────────────────────────────────────────────

def plot_bar(df, column, y_col=None, fill=DEFAULT_COLOR, color='black',
             title=None, subtitle=None, top_n=None, flip=True,
             bar_width=0.9, facet_by=None, show_values=False, value_size=7,
             x_label=None, y_label=None,
                      figure_size=DEFAULT_FIGURE):
    """
    Bar chart showing frequency of each category.

    y_col       : pass the count/value column name if your data is already aggregated.
                  If not provided, function counts raw rows automatically.
    show_values : set True to display value labels on each bar (default False)
    value_size  : font size of value labels (default 7)

    Examples:
        plot_bar(netflix, 'country', top_n=10)
        plot_bar(netflix, 'country', top_n=10, show_values=True)
        plot_bar(top_countries, 'country', y_col='count')
        plot_bar(df, 'country', figure_size=(14, 7))
    """
    if y_col:
        # Keep facet_by column if provided so plotnine can find it
        cols = [column, y_col] + ([facet_by] if facet_by and facet_by in df.columns else [])
        plot_df = df[cols].copy()
        plot_df = plot_df.rename(columns={y_col: 'count'})
    else:
        plot_df = df[column].value_counts().reset_index()
        plot_df.columns = [column, 'count']
        # Re-attach facet column if needed
        if facet_by and facet_by in df.columns:
            plot_df = plot_df.merge(
                df[[column, facet_by]].drop_duplicates(),
                on=column, how='left'
            )

    if top_n:
        plot_df = plot_df.sort_values('count', ascending=False).head(top_n)

    # Always derive top category from the highest value, not row position
    # since input data may not be sorted
    top_row  = plot_df.loc[plot_df['count'].idxmax()]
    top_cat  = top_row[column]
    top_pct  = top_row['count'] / plot_df['count'].sum() * 100
    title    = title    or f'Frequency of {column}'
    subtitle = subtitle or f'Top category: {top_cat} ({top_pct:.1f}% of total)'

    # reorder() conflicts with facet panels when each panel has different
    # value scales — use plain column name when faceting
    x_aes = f'reorder({column}, count)' if not facet_by else column

    plot = (
        ggplot(plot_df, aes(x=x_aes, y='count'))
        + geom_col(fill=fill, color=color, width=bar_width)
        + _labs(title, subtitle, column, y_col if y_col else 'Count', x_label=x_label, y_label=y_label)
        + theme_minimal()
        + theme(axis_text_y=element_text(size=8))
        + _figsize(figure_size)
    )

    if show_values:
        plot = plot + geom_text(
            aes(label='count'),
            position=position_stack(vjust=1.05),
            size=value_size
        )

    plot = _apply_flip(plot, flip)
    # Use free scales when faceting so each panel has its own axis range
    if facet_by:
        plot = plot + facet_wrap(facet_by, scales='free')
    ipy_display(plot.draw())
    plt.close()
    return None


def plot_bar_stacked(df, x_col, fill_col, y_col=None, title=None,
                     subtitle=None, flip=False, bar_width=0.9,
                     x_label_angle=0, x_label=None, y_label=None,
                      figure_size=DEFAULT_FIGURE):
    """
    Stacked bar chart showing composition of categories.

    y_col: pass value column if data is pre-aggregated.
           If not provided, counts rows automatically.

    Examples:
        plot_bar_stacked(netflix, 'country', 'type')
        plot_bar_stacked(summary_df, 'country', 'type', y_col='count')
        plot_bar_stacked(df, 'country', 'type', figure_size=(14, 7))
    """
    title    = title    or f'{x_col} by {fill_col} (stacked)'
    subtitle = subtitle or f'Each bar shows the composition of {fill_col} within {x_col}'

    if y_col:
        plot = (
            ggplot(df, aes(x=x_col, y=y_col, fill=fill_col))
            + geom_col(position=position_stack(), color='white', width=bar_width)
            + _labs(title, subtitle, x_col, y_col, x_label=x_label, y_label=y_label)
            + theme_minimal()
            + theme(axis_text_x=element_text(rotation=x_label_angle, hjust=1 if x_label_angle > 0 else 0.5),
            plot_margin=0.05)
            + _figsize(figure_size)
        )
    else:
        plot = (
            ggplot(df, aes(x=x_col, fill=fill_col))
            + geom_bar(position=position_stack(), color='white', width=bar_width)
            + _labs(title, subtitle, x_col, 'Count', x_label=x_label, y_label=y_label)
            + theme_minimal()
            + theme(axis_text_x=element_text(rotation=x_label_angle, hjust=1 if x_label_angle > 0 else 0.5),
            plot_margin=0.05)
            + _figsize(figure_size)
        )
    plot = _apply_flip(plot, flip)
    ipy_display(plot.draw())
    plt.close()
    return None


def plot_bar_grouped(df, x_col, fill_col, y_col=None, title=None,
                     subtitle=None, flip=False, bar_width=0.9,
                     x_label_angle=0, x_label=None, y_label=None,
                      figure_size=DEFAULT_FIGURE):
    """
    Grouped (side-by-side) bar chart.

    y_col: pass value column if data is pre-aggregated.
           If not provided, counts rows automatically.

    Examples:
        plot_bar_grouped(netflix, 'country', 'type')
        plot_bar_grouped(summary_df, 'country', 'type', y_col='count')
        plot_bar_grouped(df, 'country', 'type', figure_size=(14, 6))
    """
    title    = title    or f'{x_col} by {fill_col} (grouped)'
    subtitle = subtitle or f'Side-by-side comparison of {fill_col} within {x_col}'

    if y_col:
        plot = (
            ggplot(df, aes(x=x_col, y=y_col, fill=fill_col))
            + geom_col(position=position_dodge(), color='white', width=bar_width)
            + _labs(title, subtitle, x_col, y_col, x_label=x_label, y_label=y_label)
            + theme_minimal()
            + theme(axis_text_x=element_text(rotation=x_label_angle, hjust=1 if x_label_angle > 0 else 0.5),
            plot_margin=0.05)
            + _figsize(figure_size)
        )
    else:
        plot = (
            ggplot(df, aes(x=x_col, fill=fill_col))
            + geom_bar(position=position_dodge(), color='white', width=bar_width)
            + _labs(title, subtitle, x_col, 'Count', x_label=x_label, y_label=y_label)
            + theme_minimal()
            + theme(axis_text_x=element_text(rotation=x_label_angle, hjust=1 if x_label_angle > 0 else 0.5),
            plot_margin=0.05)
            + _figsize(figure_size)
        )
    plot = _apply_flip(plot, flip)
    ipy_display(plot.draw())
    plt.close()
    return None


def plot_highlight_bar(df, col, value_col, highlight='both', n=1,
                       top_color='#62866c', bottom_color='#E63946',
                       normal_color='#c9c9c9', title=None, subtitle=None,
                       flip=True, show_values=False, value_size=7,
                       figure_size=DEFAULT_FIGURE):
    """
    Bar chart with top and/or bottom n bars highlighted in different colors.
    Everything else stays grey.

    highlight    : 'top', 'bottom', or 'both' (default 'both')
    n            : number of bars to highlight at each end (default 1)
    top_color    : color for highest bar(s) (default green)
    bottom_color : color for lowest bar(s) (default red)
    normal_color : color for all other bars (default grey)
    show_values  : set True to display value labels on bars (default False)
    value_size   : font size of value labels (default 7)

    Examples:
        plot_highlight_bar(df, 'sub_category', 'profit')
        plot_highlight_bar(df, 'sub_category', 'profit', highlight='top', n=3)
        plot_highlight_bar(df, 'region', 'sales', highlight='bottom', show_values=True)
    """
    plot_df = df[[col, value_col]].copy().sort_values(value_col)
    plot_df['_highlight'] = 'normal'

    if highlight in ('top', 'both'):
        top_idx = plot_df.nlargest(n, value_col).index
        plot_df.loc[top_idx, '_highlight'] = 'top'

    if highlight in ('bottom', 'both'):
        bot_idx = plot_df.nsmallest(n, value_col).index
        plot_df.loc[bot_idx, '_highlight'] = 'bottom'

    color_map = {'top': top_color, 'bottom': bottom_color, 'normal': normal_color}

    title    = title    or f'{value_col} by {col}'
    subtitle = subtitle or f'Top and bottom {n} highlighted'

    plot = (
        ggplot(plot_df, aes(f'reorder({col}, {value_col})', value_col, fill='_highlight'))
        + geom_col(show_legend=False)
        + scale_fill_manual(values=color_map)
        + _labs(title, subtitle, col, value_col, x_label=x_label, y_label=y_label)
        + theme_minimal()
        + _figsize(figure_size)
    )

    if show_values:
        plot = plot + geom_text(
            aes(label=value_col),
            position=position_stack(vjust=1.05),
            size=value_size
        )

    plot = _apply_flip(plot, flip)
    ipy_display(plot.draw())
    plt.close()
    return None




def plot_lollipop(df, column, y_col=None, title=None, subtitle=None,
                  top_n=None, color=DEFAULT_COLOR, flip=True,
                  line_size=0.8, point_size=3,
                  show_values=False, value_size=7,
                  x_label=None, y_label=None,
                      figure_size=DEFAULT_FIGURE):
    """
    Lollipop chart: cleaner alternative to bar charts for ranked data.

    y_col       : pass value column if data is pre-aggregated.
                  If not provided, counts rows automatically.
    show_values : set True to display value labels on points (default False)
    value_size  : font size of value labels (default 7)

    Examples:
        plot_lollipop(netflix, 'country', top_n=15)
        plot_lollipop(netflix, 'country', top_n=15, show_values=True)
        plot_lollipop(top_countries, 'country', y_col='count')
        plot_lollipop(df, 'country', figure_size=(12, 8))
    """
    if y_col:
        plot_df = df[[column, y_col]].copy()
        plot_df.columns = [column, 'count']
    else:
        plot_df = df[column].value_counts().reset_index()
        plot_df.columns = [column, 'count']

    if top_n:
        plot_df = plot_df.sort_values('count', ascending=False).head(top_n)

    title    = title    or f'Ranked frequency of {column}'
    subtitle = subtitle or f'Top {len(plot_df)} categories shown'

    plot = (
        ggplot(plot_df, aes(x=f'reorder({column}, count)', y='count'))
        + geom_segment(aes(xend=column, y=0, yend='count'), color=color, size=line_size)
        + geom_point(color=color, size=point_size)
        + _labs(title, subtitle, column, y_col if y_col else 'Count', x_label=x_label, y_label=y_label)
        + theme_minimal()
        + theme(axis_text_y=element_text(size=8))
        + _figsize(figure_size)
    )

    if show_values:
        plot = plot + geom_text(
            aes(label='count'),
            nudge_y=plot_df['count'].max() * 0.02,
            size=value_size
        )

    plot = _apply_flip(plot, flip)
    ipy_display(plot.draw())
    plt.close()
    return None


# ─────────────────────────────────────────────
# TIME SERIES
# ─────────────────────────────────────────────

def plot_line(df, x_col, y_col, color=DEFAULT_COLOR, title=None,
              subtitle=None, facet_by=None, line_size=1,
              show_points=False, point_color=None,
              agg_func=None, x_breaks=None, x_label_angle=0,
                      x_label=None, y_label=None,
                      figure_size=DEFAULT_FIGURE):
    """
    Line chart for trends over time or any ordered x axis.
    Auto-subtitle shows date range if x_col is datetime.

    line_size   : thickness of the line (default 1)
    show_points : set True to add data points on the line (default False)
    point_color : color of points if show_points=True (defaults to line color)
    agg_func    : optional aggregation if passing raw data.
                  e.g. agg_func='sum' groups by x_col and sums y_col first.
                  Options: 'sum', 'mean', 'count', 'median'

    Examples:
        plot_line(content_by_year, 'year_added', 'count')
        plot_line(df, 'invoice_date', 'amount', agg_func='sum')
        plot_line(df, 'invoice_date', 'amount', show_points=True, line_size=2)
        plot_line(df, 'year', 'sales', figure_size=(14, 5))
    """
    if agg_func:
        df = df.groupby(x_col)[y_col].agg(agg_func).reset_index()

    title = title or f'{y_col} over {x_col}'
    if pd.api.types.is_datetime64_any_dtype(df[x_col]):
        subtitle = subtitle or f"Period: {df[x_col].min().date()} to {df[x_col].max().date()}"
    else:
        subtitle = subtitle or f'Min: {df[y_col].min():,.0f} | Max: {df[y_col].max():,.0f}'

    plot = (
        ggplot(df, aes(x=x_col, y=y_col, group=1))
        + geom_line(color=color, size=line_size)
        + _labs(title, subtitle, x_col, y_col, x_label=x_label, y_label=y_label)
        + theme_minimal()
        + theme(axis_text_x=element_text(rotation=x_label_angle, hjust=1 if x_label_angle > 0 else 0.5),
            plot_margin=0.05)
        + _figsize(figure_size)
    )

    if show_points:
        pt_color = point_color or color
        plot = plot + geom_point(color=pt_color, size=2)

    if x_breaks is not None and pd.api.types.is_numeric_dtype(df[x_col]):
        mn = int(df[x_col].min())
        mx = int(df[x_col].max())
        breaks = list(range(mn, mx + 1, x_breaks))
        from plotnine import scale_x_continuous
        plot = plot + scale_x_continuous(breaks=breaks)

    plot = _apply_facet(plot, facet_by)
    ipy_display(plot.draw())
    plt.close()
    return None



def plot_area(df, x_col, y_col, fill=DEFAULT_COLOR, color=DEFAULT_COLOR,
              alpha=0.4, title=None, subtitle=None, line_size=1,
              show_points=False, agg_func=None, x_breaks=None,
              x_label_angle=0, x_label=None, y_label=None,
                      figure_size=DEFAULT_FIGURE):
    """
    Filled area chart. Good for showing volume trends over time.

    line_size   : thickness of the line border on the area (default 1)
    show_points : set True to add data points (default False)
    agg_func    : optional aggregation e.g. 'sum', 'mean'

    Examples:
        plot_area(monthly_df, 'month', 'sales')
        plot_area(df, 'date', 'revenue', agg_func='sum', show_points=True)
        plot_area(df, 'year', 'count', figure_size=(14, 5))
    """
    if agg_func:
        df = df.groupby(x_col)[y_col].agg(agg_func).reset_index()

    title    = title    or f'{y_col} over {x_col} (area)'
    subtitle = subtitle or (
        f"Total: {df[y_col].sum():,.0f} | Peak: {df[y_col].max():,.0f}"
    )

    plot = (
        ggplot(df, aes(x=x_col, y=y_col))
        + geom_area(fill=fill, color=color, alpha=alpha, size=line_size)
        + _labs(title, subtitle, x_col, y_col, x_label=x_label, y_label=y_label)
        + theme_minimal()
        + theme(axis_text_x=element_text(rotation=x_label_angle, hjust=1 if x_label_angle > 0 else 0.5),
            plot_margin=0.05)
        + _figsize(figure_size)
    )

    if show_points:
        plot = plot + geom_point(color=color, size=2)

    if x_breaks is not None and pd.api.types.is_numeric_dtype(df[x_col]):
        mn = int(df[x_col].min())
        mx = int(df[x_col].max())
        breaks = list(range(mn, mx + 1, x_breaks))
        from plotnine import scale_x_continuous
        plot = plot + scale_x_continuous(breaks=breaks)

    ipy_display(plot.draw())
    plt.close()
    return None




# ─────────────────────────────────────────────
# COMPOSITION & OPERATIONAL
# ─────────────────────────────────────────────

def plot_correlation(df, title='Correlation Heatmap', subtitle=None,
                     figsize=(12, 8), cmap='coolwarm'):
    """
    Heatmap of correlations across all numeric columns (seaborn).
    figsize: (width, height) — uses matplotlib figsize since seaborn handles this chart.

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





def plot_donut(df, column, y_col=None, title=None, subtitle=None,
               top_n=None, colors=None, figsize=(7, 7)):
    """
    Donut chart for proportions of a categorical column.
    Uses matplotlib since plotnine does not support pie/donut natively.

    y_col  : pass value column name if data is pre-aggregated.
              If not provided, counts rows automatically.
    top_n  : limit to top N categories, grouping the rest as 'Other'.
    figsize: (width, height) e.g. figsize=(9, 9)

    Examples:
        plot_donut(netflix, 'country')
        plot_donut(netflix, 'country', top_n=6)
        plot_donut(summary_df, 'country', y_col='count', top_n=6)
        plot_donut(df, 'type', colors=['#2E86AB', '#E63946'], figsize=(8, 8))
    """
    if y_col:
        counts = df.set_index(column)[y_col]
    else:
        counts = df[column].value_counts()

    if top_n and len(counts) > top_n:
        top   = counts.head(top_n)
        other = counts.iloc[top_n:].sum()
        counts = pd.concat([top, pd.Series({'Other': other})])

    top_cat  = counts.index[0]
    top_pct  = counts.iloc[0] / counts.sum() * 100
    title    = title    or f'Composition of {column}'
    subtitle = subtitle or f'Largest share: {top_cat} ({top_pct:.1f}%)'

    if colors is None:
        palette = ['#2E86AB', '#E63946', '#F4A261', '#2A9D8F',
                   '#E9C46A', '#A8DADC', '#457B9D', '#1D3557']
        colors = palette[:len(counts)]

    fig, ax = plt.subplots(figsize=figsize)
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=counts.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        wedgeprops=dict(width=0.5)
    )
    for text in autotexts:
        text.set_fontsize(9)

    ax.set_title(f'{title}\n{subtitle}', fontsize=13, pad=20)
    plt.tight_layout()
    plt.show()



# ─────────────────────────────────────────────
# MODEL EVALUATION
# ─────────────────────────────────────────────

def plot_actual_vs_predicted(y_test, y_pred, point_color='#62866c',
                              line_color='red', alpha=0.3,
                              title='Actual vs Predicted',
                              subtitle='Red line = perfect prediction',
                              figure_size=(7, 5)):
    """
    Scatter plot of actual vs predicted values with a perfect prediction line.
    Points close to the red line = good predictions.
    Reusable across any regression model.

    y_test : actual target values (array, Series, or list)
    y_pred : predicted values from model.predict()

    Examples:
        plot_actual_vs_predicted(y_test, profit_pred)
        plot_actual_vs_predicted(y_test, y_pred, point_color='#2E86AB')
    """
    import pandas as pd
    from sklearn.metrics import r2_score, mean_absolute_error

    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    subtitle = subtitle or f'R² = {r2:.4f} | MAE = {mae:.2f}'

    results_df = pd.DataFrame({'actual': y_test, 'predicted': y_pred})

    plot = (
        ggplot(results_df, aes(x='actual', y='predicted'))
        + geom_point(color=point_color, alpha=alpha, size=1.5)
        + geom_abline(intercept=0, slope=1, color=line_color, linetype='dashed')
        + _labs(title, subtitle, 'Actual', 'Predicted', x_label=x_label, y_label=y_label)
        + theme_minimal()
        + _figsize(figure_size)
    )
    ipy_display(plot.draw())
    plt.close()
    return None


def plot_feature_importance(model, feature_names, model_type='tree',
                             top_n=None, pos_color='#62866c', neg_color='#E63946',
                             title='Feature Importance',
                             subtitle=None, flip=True, figure_size=(8, 6)):
    """
    Plot feature importance for any sklearn or XGBoost model.
    Automatically handles both tree-based models and linear models.

    model        : fitted model object
    feature_names: list of feature column names (X.columns)
    model_type   : 'tree' for Random Forest/XGBoost (uses feature_importances_)
                   'linear' for Linear/Ridge/Lasso (uses coef_, direction matters)
    top_n        : show only top N features (default None = show all)
    pos_color    : color for positive coefficients/high importance (default green)
    neg_color    : color for negative coefficients (default red, linear only)

    Examples:
        plot_feature_importance(model_xgb, X.columns)
        plot_feature_importance(model_rf, X.columns, top_n=10)
        plot_feature_importance(model_lr, X.columns, model_type='linear')
    """
    if model_type == 'tree':
        importance_df = pd.DataFrame({
            'feature':    list(feature_names),
            'importance': model.feature_importances_
        }).sort_values('importance')

        if top_n:
            importance_df = importance_df.tail(top_n)

        subtitle = subtitle or 'Higher score = stronger influence on prediction'

        plot = (
            ggplot(importance_df, aes('reorder(feature, importance)', 'importance'))
            + geom_col(fill=pos_color, show_legend=False)
            + _labs(title, subtitle, 'Feature', 'Importance Score', x_label=x_label, y_label=y_label)
            + theme_minimal()
            + _figsize(figure_size)
        )

    elif model_type == 'linear':
        importance_df = pd.DataFrame({
            'feature':     list(feature_names),
            'coefficient': model.coef_
        }).sort_values('coefficient')

        if top_n:
            n = top_n // 2
            importance_df = pd.concat([
                importance_df.head(n),
                importance_df.tail(n)
            ])

        importance_df['direction'] = importance_df['coefficient'].apply(
            lambda x: 'Positive' if x > 0 else 'Negative'
        )
        subtitle = subtitle or 'Green = increases profit | Red = decreases profit'

        plot = (
            ggplot(importance_df, aes('reorder(feature, coefficient)', 'coefficient', fill='direction'))
            + geom_col(show_legend=False)
            + geom_hline(yintercept=0, linetype='dashed', color='black', size=0.4)
            + scale_fill_manual(values={'Positive': pos_color, 'Negative': neg_color})
            + _labs(title, subtitle, 'Feature', 'Coefficient', x_label=x_label, y_label=y_label)
            + theme_minimal()
            + _figsize(figure_size)
        )
    else:
        raise ValueError("model_type must be 'tree' or 'linear'")

    plot = _apply_flip(plot, flip)
    ipy_display(plot.draw())
    plt.close()
    return None


