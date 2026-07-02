import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for file output
import matplotlib.pyplot as plt

sold     = pd.read_csv('combined_sold.csv')
listings = pd.read_csv('combined_listings.csv')

keep_cols = [
    'ClosePrice', 'ListPrice', 'OriginalListPrice', 'LivingArea',
    'LotSizeAcres', 'BedroomsTotal', 'BathroomsTotalInteger',
    'DaysOnMarket', 'YearBuilt'
]

PLOTS_DIR = 'full_distribution_plots'
os.makedirs(PLOTS_DIR, exist_ok=True)


def dataset_understanding(): #
    print(f"Columns: {sold.columns.tolist()}")
    print(f"Head:\n{sold.head()}")
    print(f"Types of Listings: {sold['PropertyType'].unique()}")  # should be 1


def null_count_summary(df, name="DataFrame"):
    total = len(df)
    null_counts = df.isnull().sum()
    null_pct    = df.isnull().mean() * 100
    summary = pd.DataFrame({
        'null_count': null_counts,
        'null_pct':   null_pct.round(2)
    }).sort_values('null_count', ascending=False)

    print(f"\n{'='*50}")
    print(f"  Null Count Summary — {name}  (n={total:,} rows, {len(df.columns)} cols)")
    print(f"{'='*50}")
    print(f"  {'Column':<35} {'Nulls':>8} {'%':>8}")
    print(f"  {'-'*53}")
    for col, row in summary.iterrows():
        if row['null_count'] > 0:
            print(f"  {col:<35} {int(row['null_count']):>8,} {row['null_pct']:>7.2f}%")
    no_nulls = (summary['null_count'] == 0).sum()
    print(f"  {'-'*53}")
    print(f"  {no_nulls} column(s) have no nulls.")
    print(f"{'='*50}\n")


def missing_value_analysis(cols):
    limit = 0.9
    columns_to_drop = sold.columns[sold.isnull().mean() > limit].tolist()
    print(f"Columns to flag (>{int(limit*100)}% missing): {columns_to_drop}")
    drop_targets = [c for c in columns_to_drop if c not in cols]
    if drop_targets:
        sold.drop(columns=drop_targets, inplace=True)
        print(f"Dropped: {drop_targets}")


def distribution_analysis(field):
    try:
        series = sold[field].dropna()
        if series.empty:
            print(f"[{field}] No non-null values — skipping.")
            return

        # ── Percentile summary ────────────────────────────────────────────────
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        pct_values  = np.percentile(series, percentiles)
        print(f"\n{'='*55}")
        print(f"  Distribution analysis: {field}  (n={len(series):,})")
        print(f"{'='*55}")
        print(f"  {'Statistic':<12}  {'Value':>15}")
        print(f"  {'-'*30}")
        print(f"  {'Mean':<12}  {series.mean():>15,.2f}")
        print(f"  {'Std':<12}  {series.std():>15,.2f}")
        for p, v in zip(percentiles, pct_values):
            print(f"  {f'p{p}':<12}  {v:>15,.2f}")
        print(f"  {'Min':<12}  {series.min():>15,.2f}")
        print(f"  {'Max':<12}  {series.max():>15,.2f}")

        # ── IQR-based extreme outlier detection ───────────────────────────────
        q1, q3  = series.quantile(0.25), series.quantile(0.75)
        iqr     = q3 - q1
        lo_mild, hi_mild     = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        lo_extreme, hi_extreme = q1 - 3.0 * iqr, q3 + 3.0 * iqr

        mild_mask    = (series < lo_mild)    | (series > hi_mild)
        extreme_mask = (series < lo_extreme) | (series > hi_extreme)

        print(f"\n  Outlier fences (IQR={iqr:,.2f})")
        print(f"    Mild    (1.5×IQR): [{lo_mild:,.2f}, {hi_mild:,.2f}]  — {mild_mask.sum():,} rows")
        print(f"    Extreme (3.0×IQR): [{lo_extreme:,.2f}, {hi_extreme:,.2f}]  — {extreme_mask.sum():,} rows")

        if extreme_mask.sum():
            extreme_vals = series[extreme_mask].sort_values()
            print(f"\n  Extreme outlier values (sample, up to 20):")
            print(f"  {extreme_vals.head(10).tolist()}  …  {extreme_vals.tail(10).tolist()}")

        # ── Histogram ─────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(series, bins=60, color='steelblue', edgecolor='white', linewidth=0.4)
        ax.axvline(series.mean(),   color='red',    linestyle='--', linewidth=1.2, label=f'Mean {series.mean():,.0f}')
        ax.axvline(series.median(), color='orange', linestyle='--', linewidth=1.2, label=f'Median {series.median():,.0f}')
        ax.set_title(f'{field} — Histogram', fontsize=13)
        ax.set_xlabel(field)
        ax.set_yscale('log')
        ax.set_ylabel('Count')
        ax.legend()
        plt.tight_layout()
        hist_path = os.path.join(PLOTS_DIR, f'{field}_histogram.png')
        fig.savefig(hist_path, dpi=150)
        plt.close(fig)
        print(f"\n  Histogram saved → {hist_path}")

        # ── Boxplot ───────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.boxplot(series, vert=False, patch_artist=True,
                   boxprops=dict(facecolor='steelblue', color='navy'),
                   medianprops=dict(color='red', linewidth=2),
                   flierprops=dict(marker='o', markerfacecolor='salmon',
                                   markersize=3, alpha=0.5))
        ax.set_title(f'{field} — Boxplot', fontsize=13)
        ax.set_xlabel(field)
        plt.tight_layout()
        box_path = os.path.join(PLOTS_DIR, f'{field}_boxplot.png')
        fig.savefig(box_path, dpi=150)
        plt.close(fig)
        print(f"  Boxplot   saved → {box_path}")

    except KeyError:
        print(f"[{field}] Column not found in dataset.")
    except Exception as e:
        print(f"[{field}] Error during distribution analysis: {e}")


if __name__ == "__main__":
    print(f"Sold Before: {sold.shape}") # 414197, 69
    print(f"Listings Before: {listings.shape}") # 567138, 84
    dataset_understanding()
    print(sold.shape) # 414197, 69
    print(listings.shape) # 567138, 84
    null_count_summary(sold, "sold_residential")
    null_count_summary(listings, "listings_residential")
    missing_value_analysis(keep_cols)
    print(sold.shape) # 414197, 69
    print(listings.shape) # 567138, 84
    distribution_analysis("ClosePrice")
    distribution_analysis("LivingArea")
    distribution_analysis("DaysOnMarket")
    sold.to_csv('analysis/analyzed_sold_residential.csv', index=False)
    listings.to_csv('analysis/analyzed_listings_residential.csv', index=False)