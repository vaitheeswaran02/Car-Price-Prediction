import streamlit as st
import pandas as pd
import numpy as np
import base64
import random

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Car Price Predictor (RL)", page_icon="🚗", layout="wide")

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
        }}

        .block-container {{
            background-color: rgba(0, 0, 0, 0.65);
            padding: 25px;
            border-radius: 15px;
        }}

        h1, h2, h3, label {{
            color: white !important;
        }}

        div.stButton > button {{
            background-color: #6a0dad;
            color: white;
            border-radius: 10px;
            height: 50px;
            font-size: 18px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg("car_bg.jpg")

# ---------- TITLE ----------
st.markdown("<h1 style='text-align:center;'>🚗 RL Car Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("---")

# ---------- LOAD DATA ----------
data = pd.read_csv("car_price_prediction.csv")

# ---------- RL SETUP ----------
price_bins = [200000, 400000, 600000, 800000, 1000000]
actions = [-1, 0, 1]  # decrease, stay, increase
Q = np.zeros((len(price_bins), len(actions)))

alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 500

def get_state(price):
    for i, p in enumerate(price_bins):
        if price <= p:
            return i
    return len(price_bins) - 1

# ---------- TRAIN Q-LEARNING ----------
for episode in range(episodes):
    row = data.sample(1).iloc[0]

    actual_price = row["Price"]
    pred_price = random.choice(price_bins)

    for step in range(10):
        state = get_state(pred_price)

        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, 2)
        else:
            action = np.argmax(Q[state])

        pred_price += actions[action] * 50000

        error = abs(actual_price - pred_price)
        reward = -error / 100000

        new_state = get_state(pred_price)

        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[new_state]) - Q[state, action]
        )

# ---------- LAYOUT ----------
col1, col2 = st.columns([1, 1])

# ---------- INPUT ----------
with col1:
    st.subheader("📋 Enter Car Details")

    year = st.number_input("Year", 1990, 2025)
    engine = st.number_input("Engine Size (CC)", 500, 5000)
    mileage = st.number_input("Mileage (KM)", 0, 200000)

# ---------- PREDICTION FUNCTION ----------
def rl_predict():
    pred_price = random.choice(price_bins)

    for _ in range(10):
        state = get_state(pred_price)
        action = np.argmax(Q[state])
        pred_price += actions[action] * 50000

    return pred_price

# ---------- BUTTON ----------
st.markdown("<br>", unsafe_allow_html=True)
btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])

with btn_col2:
    predict_clicked = st.button("🚀 Predict Price", use_container_width=True)

# ---------- OUTPUT ----------
with col2:
    st.subheader("📊 Prediction Result")

    if predict_clicked:
        rl_price = rl_predict()

        st.markdown(
            f"""
            <div style="padding:25px;border-radius:15px;background-color:#6a0dad;color:white;text-align:center;">
            <h3>Q-Learning Prediction</h3>
            <h1>₹ {int(rl_price)}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )
      
