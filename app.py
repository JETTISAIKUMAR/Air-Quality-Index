import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ----------------------------------------
# 1. Page Configuration
# ----------------------------------------
st.set_page_config(
    page_title="AQI Prediction",
    page_icon="🌿",
    layout="wide"
)

# ----------------------------------------
# 2. Load AQI Model
# ----------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "aqi_random_forest.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.title("🌿 Air Quality Index Prediction Application")
st.write("Predict Air Quality Index using environmental data.")

if not MODEL_PATH.exists():
    st.error(
        "AQI model file not found. "
        "Please place a trained aqi_random_forest.pkl file "
        "in the same folder as app.py."
    )
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ----------------------------------------
# 3. Sidebar Environmental Information
# ----------------------------------------
st.sidebar.header("Environmental Information")

pm25 = st.sidebar.number_input(
    "PM2.5",
    min_value=0.0,
    max_value=1000.0,
    value=50.0,
    key="aqi_pm25"
)

pm10 = st.sidebar.number_input(
    "PM10",
    min_value=0.0,
    max_value=1000.0,
    value=80.0,
    key="aqi_pm10"
)

no2 = st.sidebar.number_input(
    "NO2",
    min_value=0.0,
    max_value=1000.0,
    value=20.0,
    key="aqi_no2"
)

nh3 = st.sidebar.number_input(
    "NH3",
    min_value=0.0,
    max_value=1000.0,
    value=10.0,
    key="aqi_nh3"
)

co = st.sidebar.number_input(
    "CO",
    min_value=0.0,
    max_value=100.0,
    value=1.0,
    key="aqi_co"
)

so2 = st.sidebar.number_input(
    "SO2",
    min_value=0.0,
    max_value=1000.0,
    value=10.0,
    key="aqi_so2"
)

o3 = st.sidebar.number_input(
    "O3",
    min_value=0.0,
    max_value=1000.0,
    value=30.0,
    key="aqi_o3"
)

# ----------------------------------------
# 4. Prepare Input Data
# ----------------------------------------
feature_names = [
    "PM2.5",
    "PM10",
    "NO2",
    "NH3",
    "CO",
    "SO2",
    "O3"
]

input_data = pd.DataFrame([[
    pm25,
    pm10,
    no2,
    nh3,
    co,
    so2,
    o3
]], columns=feature_names)

# ----------------------------------------
# 5. Display Environmental Details
# ----------------------------------------
st.subheader("Environmental Details")

st.dataframe(
    input_data,
    width="stretch"
)

# ----------------------------------------
# 6. Predict AQI
# ----------------------------------------
if st.button("Predict AQI", type="primary"):
    try:
        prediction = model.predict(input_data)[0]
        prediction = float(prediction)

        st.subheader("AQI Prediction Result")

        st.metric(
            label="Predicted Air Quality Index",
            value=f"{prediction:.2f}"
        )

        # Simplified AQI categories
        if prediction <= 50:
            st.success("Air Quality: Good")

        elif prediction <= 100:
            st.info("Air Quality: Satisfactory")

        elif prediction <= 200:
            st.warning("Air Quality: Moderate")

        elif prediction <= 300:
            st.warning("Air Quality: Poor")

        elif prediction <= 400:
            st.error("Air Quality: Very Poor")

        else:
            st.error("Air Quality: Severe")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ----------------------------------------
# 7. About Project
# ----------------------------------------
with st.expander("About This Project"):
    st.write(
        "The Air Quality Index Prediction system uses "
        "environmental pollutant measurements and a trained "
        "machine learning model to estimate AQI."
    )

    st.write("Pollutants: PM2.5, PM10, NO2, NH3, CO, SO2 and O3.")