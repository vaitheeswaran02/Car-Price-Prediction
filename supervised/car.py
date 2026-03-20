
import streamlit as st
import pandas as pd
import base64
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")

# ---------- BACKGROUND FUNCTION ----------
def set_bg(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}

        .block-container {{
            background-color: rgba(0, 0, 0, 0.65);
            padding: 25px;
            border-radius: 15px;
        }}

        h1, h2, h3, h4, h5, h6, label {{
            color: white !important;
        }}

        /* Button Style */
        div.stButton > button {{
            background-color: #6a0dad;
            color: white;
            border-radius: 10px;
            height: 50px;
            font-size: 18px;
        }}
        div.stButton > button:hover {{
            background-color: #9b59b6;
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ---------- APPLY BACKGROUND ----------
set_bg("car_bg.jpg")

# ---------- TITLE ----------
st.markdown("<h1 style='text-align:center;'>🚗 Car Price Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

# ---------- LOAD DATA ----------
data = pd.read_csv("car_price_prediction.csv")
data.columns = data.columns.str.strip()

if "Car ID" in data.columns:
    data = data.drop("Car ID", axis=1)

# Convert categorical
data = pd.get_dummies(data, drop_first=True)

X = data.drop("Price", axis=1)
y = data["Price"]

# ---------- TRAIN MODELS ----------
lr_model = LinearRegression().fit(X, y)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)

# ---------- LAYOUT ----------
col1, col2 = st.columns([1, 1])

# ---------- INPUT SECTION ----------
with col1:
    st.subheader("📋 Enter Car Details")

    year = st.number_input("Year", 1990, 2025)
    engine = st.number_input("Engine Size (CC)", 500, 5000)
    mileage = st.number_input("Mileage (KM)", 0, 200000)

    fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    condition = st.selectbox("Condition", ["New", "Used"])

# ---------- PREPARE INPUT ----------
input_dict = {
    "Year": year,
    "Engine Size": engine,
    "Mileage": mileage,
}

for col in X.columns:
    if col not in input_dict:
        input_dict[col] = 0

if "Fuel Type_Diesel" in input_dict and fuel == "Diesel":
    input_dict["Fuel Type_Diesel"] = 1

if "Transmission_Manual" in input_dict and transmission == "Manual":
    input_dict["Transmission_Manual"] = 1

if "Condition_Used" in input_dict and condition == "Used":
    input_dict["Condition_Used"] = 1

input_df = pd.DataFrame([input_dict])

# ---------- CENTER BUTTON ----------
st.markdown("<br>", unsafe_allow_html=True)
btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])

with btn_col2:
    predict_clicked = st.button("🚀 Predict Price", use_container_width=True)

# ---------- OUTPUT SECTION ----------
with col2:
    st.subheader("📊 Prediction Results")

    if predict_clicked:

        lr_pred = lr_model.predict(input_df)[0]
        rf_pred = rf_model.predict(input_df)[0]

        st.markdown("### 💰 Estimated Prices")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown(
                f"""
                <div style="padding:20px;border-radius:15px;background-color:#6a0dad;color:white;text-align:center;">
                <h3>Linear Regression</h3>
                <h2>₹ {int(lr_pred)}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:
            st.markdown(
                f"""
                <div style="padding:20px;border-radius:15px;background-color:#9b59b6;color:white;text-align:center;">
                <h3>Random Forest</h3>
                <h2>₹ {int(rf_pred)}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
