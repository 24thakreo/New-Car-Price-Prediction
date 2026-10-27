import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from pathlib import Path

# Get project folder
BASE_DIR = Path(__file__).resolve().parent

# Page configuration
st.set_page_config(
    page_title="New Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# Custom design
st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
h1 {
    color: #172554;
}
.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    height: 45px;
    width: 100%;
    font-size: 17px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("🚗 New Car Price Prediction System")
st.write("Enter the car specifications to estimate its Ex-Showroom Price.")

st.divider()

# File paths
MODEL_FILE = BASE_DIR / "new_car_price_model.pkl"
FEATURE_FILE = BASE_DIR / "car_model_features.json"
DATA_FILE = BASE_DIR / "cars_cleaned.csv"
MODEL_NAME_FILE = BASE_DIR / "best_car_model_name.txt"
COMPARISON_FILE = BASE_DIR / "car_model_comparison.csv"

# Check required files
required_files = [
    MODEL_FILE,
    FEATURE_FILE,
    DATA_FILE
]

missing_files = [
    file.name for file in required_files
    if not file.exists()
]

if missing_files:
    st.error("Missing files: " + ", ".join(missing_files))
    st.info("Place the downloaded Colab files in the same folder as app.py.")
    st.stop()

# Load model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

# Load features
@st.cache_data
def load_features():
    with open(FEATURE_FILE, "r") as file:
        return json.load(file)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

# Load project files
try:
    model = load_model()
    features = load_features()
    data = load_data()

    # Support a list of feature names or a dictionary containing them
    if isinstance(features, dict):
        features = features.get("features", list(features.keys()))

    if not isinstance(features, list):
        st.error("Invalid car_model_features.json format.")
        st.stop()

except Exception as e:
    st.error(f"Unable to load project files: {e}")
    st.stop()

# Input section
st.subheader("Enter Car Specifications")
st.write("Select or enter the details of the car below.")

input_values = {}

# Use a form so inputs are submitted together
with st.form("car_prediction_form"):

    col1, col2 = st.columns(2)

    for i, column in enumerate(features):

        current_col = col1 if i % 2 == 0 else col2

        with current_col:

            if column not in data.columns:
                st.warning(f"Feature not found in dataset: {column}")
                continue

            series = data[column]

            if pd.api.types.is_numeric_dtype(series):

                values = pd.to_numeric(
                    series, errors="coerce"
                ).dropna()

                if len(values) == 0:
                    input_values[column] = st.number_input(
                        column, value=0.0
                    )
                else:
                    minimum = float(values.min())
                    maximum = float(values.max())
                    default = float(values.median())

                    if minimum == maximum:
                        input_values[column] = st.number_input(
                            column,
                            value=default
                        )
                    else:
                        input_values[column] = st.number_input(
                            column,
                            min_value=minimum,
                            max_value=maximum,
                            value=default,
                            step=1.0
                        )

            else:

                options = sorted(
                    series.dropna().astype(str).unique().tolist()
                )

                if options:
                    input_values[column] = st.selectbox(
                        column,
                        options
                    )
                else:
                    input_values[column] = st.text_input(
                        column,
                        value="Unknown"
                    )

    submitted = st.form_submit_button(
        "🚘 Predict New Car Price"
    )

# Prediction
if submitted:

    if len(input_values) != len(features):
        st.error(
            "Some features are missing from the dataset. "
            "Please check your feature names and cleaned CSV."
        )

    else:
        try:
            input_df = pd.DataFrame(
                [input_values],
                columns=features
            )

            prediction = model.predict(input_df)
            price = float(prediction[0])

            st.success("Prediction completed successfully!")

            st.markdown(
                f"""
                <div style="
                    background-color:#e0f2fe;
                    padding:25px;
                    border-radius:12px;
                    text-align:center;
                    margin-top:15px;
                ">
                    <h3 style="color:#075985;">
                        Estimated Ex-Showroom Price
                    </h3>
                    <h1 style="color:#0c4a6e;">
                        ₹ {price:,.0f}
                    </h1>
                </div>
                """,
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"Prediction error: {e}")

# Model information
st.divider()
st.subheader("Model Information")

if MODEL_NAME_FILE.exists():

    with open(MODEL_NAME_FILE, "r") as file:
        model_name = file.read().strip()

    st.write("Selected Best Model:", model_name)

else:
    st.write("Model: Trained Machine Learning Pipeline")

# Model comparison
if COMPARISON_FILE.exists():

    results = pd.read_csv(COMPARISON_FILE)

    st.subheader("Model Performance Comparison")

    st.dataframe(
        results,
        use_container_width=True
    )

    if "Model" in results.columns and "R2 Score" in results.columns:

        st.bar_chart(
            results.set_index("Model")[["R2 Score"]]
        )

# Footer
st.divider()

st.caption(
    "New Car Price Prediction System | Machine Learning Project"
)
