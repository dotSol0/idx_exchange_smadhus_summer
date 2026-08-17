"""
Streamlit production check for the AVM winning model — XGBoost on the V2 (21-feature) set
(see 06_evaluation.ipynb / model_artifacts/, built by train_artifacts.py).

Loads the saved model + every preprocessing artifact (school-district target-encoding,
BedBathRatio fallback median, form defaults), asserts the loaded feature list matches the
model's expected schema, then lets a user price a property through the same feature
pipeline used in training (05_feature_engineering.ipynb).
"""
import datetime as dt
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ARTIFACT_DIR = "model_artifacts"

CBD_COORDS = {
    'Los Angeles': (34.0407, -118.2468),
    'San Francisco': (37.7749, -122.4194),
    'San Diego': (32.7157, -117.1611),
    'Sacramento': (38.5816, -121.4944),
    'San Jose': (37.3382, -121.8863),
    'Fresno': (36.7378, -119.7871),
}


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def distance_to_nearest_cbd_km(lat, lon):
    return min(haversine_km(lat, lon, clat, clon) for clat, clon in CBD_COORDS.values())


@st.cache_resource
def load_artifacts():
    model = joblib.load(f"{ARTIFACT_DIR}/model.joblib")
    with open(f"{ARTIFACT_DIR}/feature_cols.json") as f:
        feature_cols = json.load(f)
    with open(f"{ARTIFACT_DIR}/school_district_encoding.json") as f:
        district_encoding = json.load(f)
    with open(f"{ARTIFACT_DIR}/bedbath_ratio_median.json") as f:
        bedbath_ratio_median = json.load(f)['bedbath_ratio_median']
    with open(f"{ARTIFACT_DIR}/form_defaults.json") as f:
        form_defaults = json.load(f)
    with open(f"{ARTIFACT_DIR}/model_metrics.json") as f:
        model_metrics = json.load(f)

    # Structural schema check, per the production-readiness rule: never predict with a model
    # whose expected feature list doesn't match what we loaded.
    booster_features = list(model.get_booster().feature_names)
    if booster_features != feature_cols:
        raise RuntimeError(
            "Loaded feature list does not match the model's expected schema.\n"
            f"model expects: {booster_features}\nfeature_cols.json: {feature_cols}"
        )

    metrics_summary = pd.read_csv("metrics_summary.csv")
    return model, feature_cols, district_encoding, bedbath_ratio_median, form_defaults, model_metrics, metrics_summary


model, FEATURE_COLS, district_encoding, bedbath_ratio_median, form_defaults, model_metrics, metrics_summary = load_artifacts()

st.set_page_config(page_title="AVM — Property Valuation", layout="centered")
st.title("California AVM — Property Valuation")
st.caption(
    "XGBoost model (max_depth=7, n_estimators=400, lr=0.1) trained on 2023-06 to 2025-05 sold "
    "listings, tested on 2025-06 (touched once). See 06_evaluation.ipynb for the full writeup."
)

with st.expander("Model metrics (test month, 2025-06)", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²", f"{model_metrics['R2']:.3f}")
    c2.metric("MAE", f"${model_metrics['MAE']:,.0f}")
    c3.metric("MAPE", f"{model_metrics['MAPE']:.1%}")
    c4.metric("MdAPE", f"{model_metrics['MdAPE']:.1%}")
    st.caption(
        "MdAPE is the headline number (right-skewed, multi-scale real-estate errors); MAE/MAPE "
        "give a dollar-denominated and tail-risk read. Error varies by price band and city — see "
        "the caveat under the prediction below."
    )
    price_bands = metrics_summary[metrics_summary['Segment Type'] == 'Price Band']
    st.dataframe(
        price_bands[['Segment', 'N', 'R2', 'MAE', 'MAPE', 'MdAPE']].reset_index(drop=True),
        use_container_width=True,
    )

st.header("Property details")
st.caption(
    "Only the four features below are user-editable; every other model input is held at its "
    "saved training-set default (see form_defaults.json) or, for valuation date, today's date."
)

col1, col2 = st.columns(2)
with col1:
    living_area = st.number_input("Living area (sqft)", min_value=1.0,
                                   value=form_defaults['LivingArea'], step=50.0)
    bedrooms = st.number_input("Bedrooms", min_value=0.0,
                                value=form_defaults['BedroomsTotal'], step=1.0)
with col2:
    bathrooms = st.number_input("Bathrooms", min_value=0.0,
                                 value=form_defaults['BathroomsTotalInteger'], step=1.0)
    lot_size_sqft = st.number_input("Lot size (sqft)", min_value=0.0,
                                     value=form_defaults['LotSizeSquareFeet'], step=100.0)

if st.button("Estimate value", type="primary"):
    # Fields the user no longer sets directly — held at saved training-set defaults.
    main_level_bedrooms = form_defaults['MainLevelBedrooms']
    year_built = int(form_defaults['YearBuilt'])
    stories = form_defaults['Stories']
    multi_split = bool(form_defaults['Levels_MultiSplit'])
    num_levels = 1
    parking_total = form_defaults['ParkingTotal']
    association_fee = form_defaults['AssociationFee']
    latitude = form_defaults['Latitude']
    longitude = form_defaults['Longitude']
    lot_size_acres = lot_size_sqft / 43560.0
    lot_size_area = lot_size_sqft
    valuation_date = dt.date.today()

    bath_for_ratio = bathrooms if bathrooms != 0 else np.nan
    bedbath_ratio = bedrooms / bath_for_ratio if not np.isnan(bath_for_ratio) else np.nan
    if np.isnan(bedbath_ratio):
        bedbath_ratio = bedbath_ratio_median

    property_age = max(0, valuation_date.year - year_built)

    close_month = valuation_date.month
    close_month_sin = np.sin(2 * np.pi * close_month / 12)
    close_month_cos = np.cos(2 * np.pi * close_month / 12)

    school_district_mean_price = district_encoding['global_mean']

    distance_to_cbd = distance_to_nearest_cbd_km(latitude, longitude)

    row = {
        'Latitude': latitude,
        'Longitude': longitude,
        'LivingArea': living_area,
        'ParkingTotal': parking_total,
        'LotSizeAcres': lot_size_acres,
        'YearBuilt': year_built,
        'BathroomsTotalInteger': bathrooms,
        'BedroomsTotal': bedrooms,
        'Stories': stories,
        'LotSizeArea': lot_size_area,
        'MainLevelBedrooms': main_level_bedrooms,
        'AssociationFee': association_fee,
        'LotSizeSquareFeet': lot_size_sqft,
        'Levels_MultiSplit': int(multi_split),
        'NumLevels': num_levels,
        'BedBathRatio': bedbath_ratio,
        'PropertyAge': property_age,
        'SchoolDistrict_MeanPrice': school_district_mean_price,
        'CloseMonth_sin': close_month_sin,
        'CloseMonth_cos': close_month_cos,
        'DistanceToNearestCBD_km': distance_to_cbd,
    }
    X = pd.DataFrame([row])[FEATURE_COLS]  # enforce exact training column order

    prediction = float(model.predict(X)[0])

    st.subheader(f"Estimated value: ${prediction:,.0f}")

    price_bands = metrics_summary[metrics_summary['Segment Type'] == 'Price Band'].copy()

    def band_bounds(segment):
        lo, hi = segment.split("(")[1].rstrip(")").split("–")
        return float(lo.replace("$", "").replace(",", "")), float(hi.replace("$", "").replace(",", ""))

    matched_band = None
    for _, band_row in price_bands.iterrows():
        lo, hi = band_bounds(band_row['Segment'])
        if lo <= prediction <= hi:
            matched_band = band_row
            break

    if matched_band is not None:
        mdape = matched_band['MdAPE']
        st.caption(
            f"Properties in this price range (test month) had a median absolute error of "
            f"{mdape:.1%} — roughly **${prediction * (1 - mdape):,.0f} – ${prediction * (1 + mdape):,.0f}** "
            f"as a rough interval, not a formal confidence interval."
        )

    st.caption(
        "Per 06_evaluation.ipynb: the model is most reliable for \\$560K–\\$990K tract-style "
        "suburban homes, and less reliable at the price extremes or in coastal/luxury markets "
        "(e.g. San Clemente, Palm Springs, Oakland) — treat this estimate as a starting point, "
        "not a substitute for comps."
    )

    with st.expander("Model input row"):
        st.dataframe(X.T.rename(columns={0: "value"}))
