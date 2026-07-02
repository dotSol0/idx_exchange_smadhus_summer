#!/usr/bin/env python3
"""
questions.csv — exploratory data analysis for analyzed residential MLS files

This script reads:
  - analysis/analyzed_listings_residential.csv
  - analysis/analyzed_sold_residential.csv

It answers:
  - Residential vs other property type share
  - median and average close prices
  - Days on Market distribution
  - percentage of homes sold above vs below list price
  - apparent date consistency issues
  - counties with the highest median prices

Run with:
  python3 questions.csv
"""

import os
import pandas as pd

INPUT_LISTINGS = os.path.join('combined_listings.csv')
INPUT_SOLD = os.path.join('combined_sold.csv')


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f'Missing input file: {path}')
    return pd.read_csv(path)


def property_type_share(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if 'PropertyType' not in df.columns:
        return pd.DataFrame()

    counts = df['PropertyType'].fillna('Unknown').value_counts(dropna=False)
    share = counts / counts.sum() * 100
    result = pd.DataFrame({'count': counts, 'share_pct': share.round(2)})
    result.index.name = source_name
    return result


def price_statistics(df: pd.DataFrame) -> pd.Series:
    if 'ClosePrice' not in df.columns:
        return pd.Series(dtype=float)
    close_prices = df['ClosePrice'].dropna().astype(float)
    return pd.Series({
        'median_close_price': close_prices.median(),
        'average_close_price': close_prices.mean(),
        'n_sold': len(close_prices),
    })


def dom_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if 'DaysOnMarket' not in df.columns:
        return pd.DataFrame()
    dom = df['DaysOnMarket'].dropna().astype(float)
    summary = dom.describe()
    bins = [0, 7, 15, 30, 60, 90, 180, 365, 9999]
    bin_counts = pd.cut(dom, bins=bins, right=False).value_counts().sort_index()
    return pd.DataFrame({
        'statistic': summary,
        'bin_count': bin_counts,
    })


def sale_price_vs_list(df: pd.DataFrame) -> pd.DataFrame:
    if not {'ClosePrice', 'ListPrice'}.issubset(df.columns):
        return pd.DataFrame()

    sold = df[['ClosePrice', 'ListPrice']].dropna()
    sold = sold.astype(float)
    conditions = [
        sold['ClosePrice'] > sold['ListPrice'],
        sold['ClosePrice'] < sold['ListPrice'],
        sold['ClosePrice'] == sold['ListPrice'],
    ]
    labels = ['above_list_price', 'below_list_price', 'equal_list_price']
    counts = pd.Series(pd.cut(sold['ClosePrice'] - sold['ListPrice'], bins=[-float('inf'), -0.0001, 0.0001, float('inf')], labels=labels, include_lowest=True)).value_counts()
    total = counts.sum()
    return pd.DataFrame({
        'count': counts,
        'percent': (counts / total * 100).round(2),
    })


def date_consistency_issues(df: pd.DataFrame) -> pd.Series:
    issues = {}
    for col in ['ListingContractDate', 'PurchaseContractDate', 'CloseDate']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    if {'ListingContractDate', 'CloseDate'}.issubset(df.columns):
        issues['listing_after_close'] = int(((df['ListingContractDate'].notna()) &
                                            (df['CloseDate'].notna()) &
                                            (df['ListingContractDate'] > df['CloseDate'])).sum())
    if {'PurchaseContractDate', 'CloseDate'}.issubset(df.columns):
        issues['purchase_after_close'] = int(((df['PurchaseContractDate'].notna()) &
                                             (df['CloseDate'].notna()) &
                                             (df['PurchaseContractDate'] > df['CloseDate'])).sum())
    if {'ListingContractDate', 'PurchaseContractDate'}.issubset(df.columns):
        issues['listing_after_purchase'] = int(((df['ListingContractDate'].notna()) &
                                               (df['PurchaseContractDate'].notna()) &
                                               (df['ListingContractDate'] > df['PurchaseContractDate'])).sum())
    return pd.Series(issues)


def counties_highest_median_prices(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if 'CountyOrParish' not in df.columns or 'ClosePrice' not in df.columns:
        return pd.DataFrame()
    grouped = df.dropna(subset=['CountyOrParish', 'ClosePrice']).copy()
    grouped['ClosePrice'] = grouped['ClosePrice'].astype(float)
    median_prices = grouped.groupby('CountyOrParish')['ClosePrice'].median().sort_values(ascending=False)
    return median_prices.head(top_n).reset_index().rename(columns={'ClosePrice': 'median_close_price'})


def main() -> None:
    listings = load_csv(INPUT_LISTINGS)
    sold = load_csv(INPUT_SOLD)

    print('\n=== Property Type Share: Listings ===')
    print(property_type_share(listings, 'PropertyType'))

    print('\n=== Property Type Share: Sold ===')
    print(property_type_share(sold, 'PropertyType'))

    print('\n=== Close Price Statistics ===')
    print(price_statistics(sold).round(2).to_string())

    print('\n=== Days on Market Distribution ===')
    dom_dist = dom_distribution(sold)
    print(dom_dist.to_string())

    print('\n=== Sold Price vs List Price ===')
    print(sale_price_vs_list(sold).to_string())

    print('\n=== Date Consistency Issues ===')
    print(date_consistency_issues(sold).to_string())

    print('\n=== Counties with Highest Median Close Prices ===')
    print(counties_highest_median_prices(sold, top_n=15).to_string(index=False))


if __name__ == '__main__':
    main()
