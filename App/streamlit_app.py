import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# Load Saved Models

xgb_model = joblib.load("xgb_model.pkl")

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Initialize and load weights
input_size = 18  # number of features you used for training
model = LSTMModel(input_size)
model.load_state_dict(torch.load("lstm_model.pth", map_location=torch.device('cpu')))
model.eval()


# Streamlit App UI

st.set_page_config(page_title="PredictWise - AI Pricing Dashboard", layout="wide")
st.title("PredictWise – LSTM vs XGBoost Sales Forecast")

st.sidebar.header(" Upload CSV for Prediction")
uploaded_file = st.sidebar.file_uploader("Upload a CSV (with 18 feature columns used during training)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

    # XGBoost predictions
    xgb_pred = xgb_model.predict(df)

    # LSTM predictions
    X_tensor = torch.tensor(df.values, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        lstm_pred = model(X_tensor).cpu().numpy().flatten()

    df["Pred_XGBoost"] = xgb_pred
    df["Pred_LSTM"] = lstm_pred

    st.subheader("📈 Predictions (First 10 Rows)")
    st.dataframe(df.head(10))

    # Visualization
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df["Pred_XGBoost"][:200], label="XGBoost Predictions", color="#4CAF50")
    ax.plot(df["Pred_LSTM"][:200], label="LSTM Predictions", color="#FF9800", linestyle="--")
    ax.set_title("Predicted Sales Comparison (First 200 Points)", fontsize=13, weight="bold")
    ax.legend()
    st.pyplot(fig)

    # Download predictions
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Predictions as CSV", data=csv, file_name="Predictions_Output.csv")
else:
    st.info("Upload your feature CSV in the sidebar to generate predictions.")
