#  PredictWise: AI Sales Forecasting

**PredictWise-AI-SalesForecast** is an AI-driven project that predicts daily store sales using a combination of **Machine Learning (XGBoost)** and **Deep Learning (LSTM)** models.  
It leverages historical sales, promotions, and seasonality patterns to generate accurate forecasts for retail planning.



## Overview

This project focuses on solving a real-world retail forecasting problem by combining data preprocessing, feature engineering, and model training into a unified workflow.  
Both traditional ML and neural network-based approaches were explored to evaluate performance and capture temporal patterns in sales data.



## Key Highlights
- Forecasts daily store sales based on past trends and promotions  
- Compares multiple models (XGBoost, LSTM, Linear Regression)  
- Provides performance evaluation using **RMSE** and **R² metrics**  
- Includes an interactive **Streamlit dashboard** to visualize results and test predictions  
- Allows users to upload custom CSV files for prediction  



## Results Summary

| Model | RMSE | Key Insight |
|:------|:----:|:-------------|
| **XGBoost** | ~156 | Best overall generalization |
| **LSTM** | ~1000 | Captures sequential sales patterns |
| **ARIMA / Linear Regression** | ~2500+ | Used as baselines |



## How It Works
1️⃣ Clean and prepare sales data using PySpark  
2️⃣ Engineer time-based and store-specific features  
3️⃣ Train XGBoost and LSTM models  
4️⃣ Evaluate and compare model performance  
5️⃣ Visualize predictions with Streamlit dashboard  

## How to Run

1️⃣ **Clone the repository**
git clone https://github.com/aditirpatil11/PredictWise-AI-SalesForecast.git
cd PredictWise-AI-SalesForecast

pip install pandas numpy pyspark torch xgboost scikit-learn streamlit matplotlib joblib

streamlit run app/streamlit_app.py

Test Data: https://drive.google.com/file/d/1bO6P6osjURbb8C8IuLukvbINo_3FYckg/view?usp=sharing

## Tech Stack: 
Python,cPandas, NumPy, Scikit-learn, PyTorch, XGBoost, Matplotlib, PySpark, Streamlit, Google Colab, Jupyter Notebook, GitHub
