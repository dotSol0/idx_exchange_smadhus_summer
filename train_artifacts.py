"""
Refit the winning model (XGBoost, V2/21-feature set — see 06_evaluation.ipynb) and export
every artifact the Streamlit app (app.py) needs to reproduce its predictions:

  model_artifacts/
    model.joblib              fitted XGBRegressor
    feature_cols.json         exact feature order the model expects
    school_district_encoding.json   {district_name: mean_train_close_price}, plus global fallback
    bedbath_ratio_median.json train-only median BedBathRatio, for the same 0-bath fallback
                               used in 05_feature_engineering.ipynb
    form_defaults.json        train-median (or mode) value per raw input field, so the app
                               form can pre-fill something reasonable

Same RANDOM_STATE, hyperparameters, and train/test files as 06_evaluation.ipynb, so this is a
reproduction, not a re-tune — tuning already happened in 05_advanced_models.ipynb against a
validation month carved out of train.
"""
import json

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

RANDOM_STATE = 42
ARTIFACT_DIR = "model_artifacts"


def median_absolute_percentage_error(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.median(np.abs((y_true - y_pred) / y_true)))


def main():
    train = pd.read_csv("engineered_train_set_v2.csv", low_memory=False)
    test = pd.read_csv("engineered_test_set_v2.csv", low_memory=False)

    old_feature_cols = ['Latitude', 'Longitude', 'LivingArea', 'ParkingTotal', 'LotSizeAcres', 'YearBuilt',
                         'BathroomsTotalInteger', 'BedroomsTotal', 'Stories', 'LotSizeArea', 'MainLevelBedrooms',
                         'AssociationFee', 'LotSizeSquareFeet', 'Levels_MultiSplit', 'NumLevels']
    new_feature_cols = old_feature_cols + ['BedBathRatio', 'PropertyAge', 'SchoolDistrict_MeanPrice']
    FEATURE_COLS = new_feature_cols + ['CloseMonth_sin', 'CloseMonth_cos', 'DistanceToNearestCBD_km']

    X_train, y_train = train[FEATURE_COLS], train['ClosePrice']
    X_test, y_test = test[FEATURE_COLS], test['ClosePrice']

    model = XGBRegressor(max_depth=7, n_estimators=400, learning_rate=0.1, n_jobs=8,
                          objective='reg:squarederror', random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    # Sanity-check against the numbers reported in 06_evaluation.ipynb / metrics_summary.csv
    preds = model.predict(X_test)
    metrics = {
        'R2': r2_score(y_test, preds),
        'MAE': mean_absolute_error(y_test, preds),
        'MAPE': mean_absolute_percentage_error(y_test, preds),
        'MdAPE': median_absolute_percentage_error(y_test, preds),
    }
    print("Refit test metrics:", metrics)

    # --- Structural schema check before anything gets saved ---------------
    booster_features = list(model.get_booster().feature_names)
    assert booster_features == FEATURE_COLS, (
        f"Model feature order does not match FEATURE_COLS.\n"
        f"model: {booster_features}\nexpected: {FEATURE_COLS}"
    )

    # --- School district target-encoding (fit on train only) --------------
    district_means = train.groupby('SchoolDistrict')['ClosePrice'].mean()
    global_mean = float(train['ClosePrice'].mean())
    district_encoding = {
        'district_means': {str(k): float(v) for k, v in district_means.items()},
        'global_mean': global_mean,
    }

    # --- BedBathRatio fallback median (0-bath rows treated as missing, --
    # --- same as 05_feature_engineering.ipynb, computed on train only) ---
    bath = train['BathroomsTotalInteger'].replace(0, np.nan)
    ratio = train['BedroomsTotal'] / bath
    bedbath_ratio_median = float(ratio.median())

    # --- Form defaults: train median per raw numeric input, mode for the --
    # --- two categorical-ish Levels_* fields ------------------------------
    raw_input_cols = old_feature_cols  # everything the user fills in directly
    form_defaults = {}
    for col in raw_input_cols:
        if col in ('Levels_MultiSplit', 'NumLevels'):
            mode = train[col].mode(dropna=True)
            form_defaults[col] = float(mode.iloc[0]) if len(mode) else 0.0
        else:
            form_defaults[col] = float(train[col].median())

    import os
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(model, f"{ARTIFACT_DIR}/model.joblib")
    with open(f"{ARTIFACT_DIR}/feature_cols.json", "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)
    with open(f"{ARTIFACT_DIR}/school_district_encoding.json", "w") as f:
        json.dump(district_encoding, f, indent=2)
    with open(f"{ARTIFACT_DIR}/bedbath_ratio_median.json", "w") as f:
        json.dump({'bedbath_ratio_median': bedbath_ratio_median}, f, indent=2)
    with open(f"{ARTIFACT_DIR}/form_defaults.json", "w") as f:
        json.dump(form_defaults, f, indent=2)
    with open(f"{ARTIFACT_DIR}/model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved artifacts to {ARTIFACT_DIR}/")
    print(f"  model.joblib ({len(FEATURE_COLS)} features)")
    print(f"  school district encodings: {len(district_encoding['district_means'])} districts")
    print(f"  bedbath_ratio_median: {bedbath_ratio_median:.4f}")
    print(f"  form_defaults: {form_defaults}")


if __name__ == "__main__":
    main()
