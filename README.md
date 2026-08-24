# California AVM — Automated Valuation Model

An Automated Valuation Model (AVM) that predicts `ClosePrice` for residential properties in
California, built end-to-end from raw MLS exports through a deployable Streamlit pricing tool.
Modeling decisions in this repo follow IDX Exchange's internal `AVM_Data_Science_Best_Practices
v.1.pdf`, condensed in [`CLAUDE.md`](CLAUDE.md) — chronological (never random) train/test
splits, leakage-checked features, and a multi-metric evaluation broken out by price band and
geography, not just a single top-line number.

## Dataset

Raw data is a set of per-file MLS "Listing" and "Sold" CSV exports (California residential
sales), not committed to this repo (see `.gitignore` — `data/`, `school_districts/`, and `*.csv`
are all local-only). `concatenation_wout_filter.py` concatenates every CSV in a local `data/`
folder into `combined_listings.csv` / `combined_sold.csv`, then restricts to rows tagged
`PropertyType == 'Residential'` (dropping `ResidentialLease`, `Land`, `CommercialSale`, etc.,
which share the `ClosePrice` column but aren't comparable sale prices).

A second reference dataset, [CA School District Areas 2024-25](https://data.ca.gov/dataset/california-school-district-areas-2024-25/resource/7dfaf005-58eb-45db-93b1-7aff091b2172)
(boundary polygons), is spatially joined against each property's lat/long in
`05_feature_engineering.ipynb` to build a `SchoolDistrict` feature — also not committed (see
[Reproducing the pipeline](#reproducing-the-pipeline) for where to get it).

Every cleaning step logs its row-count impact; the totals through preprocessing:

| Stage | Rows remaining | Note |
|---|---|---|
| Raw `combined_sold.csv`, all property types | 414,197 | |
| Restrict to `PropertyType == 'Residential'` | 370,878 | |
| Drop missing `ClosePrice` | 370,876 | 2 rows |
| Drop implausible `ClosePrice`/price-per-sqft (<$1,000 or >$5,000/sqft) | 370,759 | 117 rows |
| Hard sanity checks (`CloseDate<ListDate`, non-positive sqft, out-of-CA lat/long, etc.) | 370,565 | 194 rows |
| Statistical outlier fences (0.5th/99.5th pct of `ClosePrice`, `OriginalListPrice`, `LivingArea`, beds/baths — **fit on pre-test rows only**) | 355,924 | 14,641 rows |
| Drop columns with ≥50% nulls (25 columns, mostly co-listing-agent/tax/school fields) | 355,924 rows × 57 cols | |

Chronological split (train = 2023-06 through 2025-05, test = 2025-06, touched once):
**336,198 train rows / 10,581 test rows.** Earlier months (2022 and early 2023) have too few
monthly sales to be reliable training signal and are excluded from both splits (see the
`X_months`-vs-`val_mae` sweep in `02_preprocessing.ipynb`, cell 13).

## Repository layout

```
02_preprocessing.ipynb       Cleaning, outlier fences, chronological split, imputation, encoding
03_regression.ipynb          Week 3 checkpoint (folded into 02_preprocessing.ipynb)
04_testing_models.ipynb      Baseline models: Decision Tree, Random Forest (+ log-target variants)
05_feature_engineering.ipynb BedBathRatio, PropertyAge, SchoolDistrict spatial join,
                              month sin/cos, distance-to-nearest-CBD; re-trains baseline models
05_advanced_models.ipynb     Full R²/MAE/MAPE/MdAPE recap across feature sets; XGBoost/LightGBM/
                              tuned Decision Tree & Random Forest; baseline-vs-advanced comparison
06_evaluation.ipynb          Refits the winning model; breaks error out by price band and city;
                              exports metrics_summary.csv
06_additional_models.ipynb   Empty — earmarked for CatBoost/HistGBM/stacking (see Next steps)
exploration.ipynb            Early EDA, superseded by 02_preprocessing.ipynb
cleaning.py                  Standalone EDA helpers (null summaries, distribution plots)
concatenation_wout_filter.py Concatenates raw per-file MLS CSV exports into combined_*.csv
train_artifacts.py           Refits the winning model and exports every app.py artifact
app.py                       Streamlit production-check app
model_artifacts/             Committed: model.joblib, feature_cols.json, school district
                              encoding, BedBathRatio fallback median, form defaults, test metrics
AUDIT.md                     Open best-practices gaps, tracked as a checklist
```

## Preprocessing (`02_preprocessing.ipynb`)

- **Chronological split determined before any stats are computed.** `pre_test` (everything
  before the 2025-06 test month) is defined first; every outlier fence, group-median imputation,
  and target encoding below is fit on `pre_test` only, then applied unchanged to test — never
  recomputed on test.
- **Outlier removal, two passes:**
  1. Hard domain sanity checks — logically impossible values (non-positive `LivingArea`, absurd
     `YearBuilt`, out-of-California lat/long) — dropped unconditionally (194 rows).
  2. Statistical fences at the 0.5th/99.5th percentile of `ClosePrice`, `OriginalListPrice`,
     `LivingArea`, `BedroomsTotal`, `BathroomsTotalInteger` — computed on `pre_test` rows only,
     frozen, then applied to both train and test (14,641 rows). `GarageSpaces` was tried and
     dropped from this list: its IQR fence collapsed to `[2, 2]` (most rows already equal 2),
     which would have silently dropped 36% of the dataset for not being exactly 2.
- **Missingness:** columns ≥90% null are dropped early in `cleaning.py`'s EDA pass; columns
  ≥50% null (25 columns — mostly co-listing-agent, tax, and school-layer fields) are dropped in
  the notebook. Remaining missing values are imputed group-wise (e.g. by county) with medians,
  fit on train only.
- **Encoding:** `CountyOrParish` and `SchoolDistrict` are target-encoded (mean `ClosePrice` per
  group, fit on train only, unseen categories fall back to the train global mean). Categorical
  `Levels` field is one-hot encoded.
- **Leakage exclusions:** `ListPrice`, `OriginalListPrice` (kept only for a sanity check, then
  dropped from `feature_cols`), `DaysOnMarket`, and other post-sale-process fields are excluded
  from the modeling feature set — these aren't available for an off-market property, which the
  model must also be able to value.

## Feature engineering (`05_feature_engineering.ipynb`)

Three families of engineered features, added on top of the ~16-feature preprocessing baseline
(`old_feature_cols`):

| Feature | Family | Notes |
|---|---|---|
| `BedBathRatio` | Intrinsic | Bedrooms ÷ bathrooms, a layout/crowding proxy. Rows with `BathroomsTotalInteger == 0` are treated as missing (would otherwise divide by zero) and filled with the train-only median ratio. |
| `PropertyAge` | Temporal | `year(CloseDate) − YearBuilt`, clipped at 0 for same-year new construction. |
| `SchoolDistrict` → `SchoolDistrict_MeanPrice` | Locational | Spatial join of each property's lat/long against CA School District Areas 2024-25 polygons (624 districts), replacing the raw `HighSchoolDistrict` column (27% missing, high-school layer only). Unified districts win outright; overlapping Elementary+High districts are combined into one label; unmatched points fall back to nearest-polygon lookup. Target-encoded the same way as `CountyOrParish` — fit on train only. |
| `CloseMonth_sin` / `CloseMonth_cos` | Temporal | Seasonality as sine/cosine, not a raw month ordinal (so December and January are adjacent, not maximally far apart). |
| `DistanceToNearestCBD_km` | Locational | Haversine distance from each property to the nearest of 6 major CA metro centroids (LA, SF, San Diego, Sacramento, San Jose, Fresno). |

Three feature-set/file pairs came out of this notebook, all sharing the same chronological split:

- `basic_train/test_set.csv` — `old_feature_cols` (16 features, pre-Week-6 baseline)
- `engineered_train/test_set.csv` — `new_feature_cols` (+ BedBathRatio, PropertyAge, SchoolDistrict_MeanPrice = 19 features)
- `engineered_train/test_set_v2.csv` — `newest_feature_cols` (+ CloseMonth sin/cos, DistanceToNearestCBD_km = 21 features) — **the winning model's feature set**

The V2 set was chosen for gradient boosting by highest **average test R² across all 6 baseline
models** (Linear Regression / Decision Tree / Random Forest, each raw + log-target) — not by
feature count.

## Models tested and results

Baseline is Linear Regression, per the best-practices doc; every model below is compared against
it on the same V2 feature set and the same held-out test month (2025-06, touched once). Ranked
by test MdAPE, the headline metric (real-estate errors are right-skewed and multi-scale, so a
median is more representative than a mean):

| Model | R² | MAE | MAPE | MdAPE |
|---|---|---|---|---|
| **XGBoost** (`max_depth=7, n_estimators=400, lr=0.1`) | **0.910** | **$115,315** | **11.7%** | **8.12%** |
| Random Forest (tuned: `n_estimators=200, max_depth=15, min_samples_leaf=1`) | 0.897 | $121,490 | 12.1% | 8.41% |
| Random Forest (log target) | 0.887 | $126,040 | 12.1% | 8.78% |
| LightGBM (`max_depth=7, n_estimators=400, lr=0.1`) | 0.898 | $123,095 | 12.6% | 8.84% |
| Decision Tree (tuned: `max_depth=20, min_samples_leaf=10`) | 0.865 | $136,596 | 13.3% | 8.94% |
| Decision Tree | 0.822 | $163,556 | 16.4% | 11.91% |
| Decision Tree (log target) | 0.798 | $169,213 | 16.3% | 12.04% |
| Linear Regression (log target) | 0.566 | $247,391 | 23.97% | 18.24% |
| **Linear Regression (baseline)** | 0.678 | $236,885 | 25.87% | **19.24%** |

**Winner: XGBoost on the V2 (21-feature) set** — beats the Linear Regression baseline's MdAPE by
more than 2.3×, and beats the next-best tuned Random Forest by ~0.3pp MdAPE. Hyperparameters were
tuned via a small manual grid over `max_depth`/`learning_rate`/`n_estimators` (XGBoost/LightGBM)
or `max_depth`/`min_samples_leaf`(/`n_estimators`) (Decision Tree/Random Forest), scored on a
validation month carved out of train (2025-05) — test was touched exactly once, for the final
numbers above.

Full train-vs-test metrics for every model × feature-set combination (old/new/V2) are in
`05_advanced_models.ipynb`; large train/test gaps on non-log Decision Tree/Random Forest are the
clearest overfitting signal in the baseline lineup.

### Error breakdown (`06_evaluation.ipynb` → `metrics_summary.csv`)

**By price band (quintiles of test-set `ClosePrice`):**

| Band | Range | MdAPE |
|---|---|---|
| Q1 | $29,900 – $560,000 | 8.7% |
| Q2 | $561,000 – $759,000 | 6.9% |
| Q3 | $759,900 – $990,000 | 7.1% |
| Q4 | $992,500 – $1,450,000 | 9.0% |
| Q5 | $1,450,700 – $3,450,000 | 9.7% |

The model is best in the $560K–$990K middle bands and worst at both tails — comps are thinner
and homes are more heterogeneous at the cheap end (condos/entry-level/manufactured) and the
luxury end. (R² is not a useful metric within a narrow price band — it goes negative in the
middle quintiles even though absolute error is low there, because within-band price variance is
too small a denominator; MAPE/MdAPE are what to trust at this granularity.)

**By geography (cities with ≥30 test-set sales):** inland/suburban tract-housing markets
(Winchester, Beaumont, Hesperia, Corona, Apple Valley — Inland Empire) run 3–5% MdAPE; coastal,
luxury, or architecturally varied markets (San Clemente, Palm Springs, Laguna Woods, Oakland,
Walnut Creek) run 12–17% MdAPE.

**Production implication:** treat the model's point estimate as most reliable for $560K–$990K
tract-style suburban homes, and flag predictions for manual review below ~$560K, above ~$1.45M,
or in high-MdAPE cities.

## Known limitations / next steps

Tracked in [`AUDIT.md`](AUDIT.md) against the full best-practices checklist. Open items:

- **No rolling-origin backtest yet.** Only the single 2025-06 cutoff has been evaluated; the
  price-band/city rankings above should be re-checked at 2–3 additional historical cutoffs
  before being treated as stable rather than a one-month artifact.
- **Leakage prevention is discipline-based, not structural** in `02_preprocessing.ipynb` — no
  `Pipeline`/`ColumnTransformer` wraps imputation/encoding/outlier logic; `ListPrice` etc. stay
  in the exported CSVs and are excluded only by omission from `feature_cols`.
- **No significance check on the baseline-vs-advanced gap** — the table above is compared by eye,
  not a noise/CI check.
- **No data snapshot/version recorded** for `combined_sold.csv`.
- `06_additional_models.ipynb` is a stub — CatBoost, `HistGradientBoostingRegressor`,
  stacking/blending, and quantile regression (for calibrated price ranges instead of a point
  estimate) are the candidates noted in `05_advanced_models.ipynb`.

## Reproducing the pipeline

### 1. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the versions this project was built and evaluated with
(pandas, numpy, scikit-learn, xgboost, lightgbm, geopandas, streamlit, etc.). Random seeds
(`random_state=42`) are fixed everywhere a model or split uses one.

### 2. Data

None of the raw data is committed (`.gitignore` excludes `data/`, `school_districts/`, `*.csv`).
To reproduce from scratch:

1. Place per-file MLS "Listing"/"Sold" CSV exports in a local `data/` folder (filenames must
   contain "Listing" or "Sold").
2. Run `python concatenation_wout_filter.py` → produces `combined_listings.csv` /
   `combined_sold.csv`.
3. Download the [CA School District Areas 2024-25](https://data.ca.gov/dataset/california-school-district-areas-2024-25/resource/7dfaf005-58eb-45db-93b1-7aff091b2172)
   shapefile into a local `school_districts/` folder (used by `05_feature_engineering.ipynb`'s
   spatial join).

### 3. Run the notebooks, in order

```
02_preprocessing.ipynb       →  basic_train_set.csv, basic_test_set.csv, cleaned_sold.csv
05_feature_engineering.ipynb →  engineered_train/test_set.csv (+ _v2.csv)
05_advanced_models.ipynb     →  full model comparison, hyperparameter tuning, picks the winner
06_evaluation.ipynb          →  metrics_summary.csv (overall + price-band + city breakdowns)
```

(`04_testing_models.ipynb` is the earlier baseline pass, superseded by `05_advanced_models.ipynb`
but kept for history; `03_regression.ipynb` and `exploration.ipynb` are early-week checkpoints,
folded into `02_preprocessing.ipynb`.)

### 4. Regenerate the deployable model artifacts

```bash
python train_artifacts.py
```

Refits the winning XGBoost/V2 configuration on `engineered_train_set_v2.csv` /
`engineered_test_set_v2.csv` and writes everything `app.py` needs into `model_artifacts/`:
`model.joblib`, `feature_cols.json`, `school_district_encoding.json`,
`bedbath_ratio_median.json`, `form_defaults.json`, `model_metrics.json`. The script asserts the
refit model's feature order matches `FEATURE_COLS` before saving — a schema mismatch fails loudly
here rather than silently in the app. `model_artifacts/` is committed to this repo, so this step
is only required if you've changed the data or the model.

### 5. Launch the app

```bash
streamlit run app.py
```

`app.py` loads every artifact in `model_artifacts/`, re-asserts the model's feature schema
matches `feature_cols.json`, and lets you price a property by entering living area, bedrooms,
bathrooms, and lot size (every other input is held at its saved training-set median/mode — see
`form_defaults.json`). It also needs `metrics_summary.csv` on disk (from step 3, `06_evaluation.
ipynb`) to render the price-band metrics table and the per-prediction error caveat — regenerate
it if you don't already have a local copy, since it's gitignored along with the rest of the CSVs.
