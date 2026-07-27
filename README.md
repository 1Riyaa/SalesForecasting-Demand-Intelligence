# Sales Forecasting & Demand Intelligence System

## Live Demo

Streamlit App:
https://salesforecasting-demand-intelligence-zkpscvtdfkzyqjkuo2tvgy.streamlit.app/

## Overview

This project presents an end-to-end sales forecasting and demand intelligence solution developed using the Superstore dataset. It analyzes historical sales trends, compares multiple forecasting models, detects anomalies, segments products based on demand patterns, and provides an interactive dashboard for business decision-making.

# 📸 Screenshots

### 🏠 Home Page

<img width="1917" height="866" alt="image" src="https://github.com/user-attachments/assets/f3d5f4e9-a0c8-4782-8b8c-cc7c9ce25680" />
Overview of the Sales Forecasting & Demand Intelligence dashboard.

---

### 📊 Exploratory Business Analysis

<img width="1917" height="907" alt="image" src="https://github.com/user-attachments/assets/95c5a525-eca8-4e23-bffc-5f3c2d50aefb" />
Interactive dashboard displaying monthly sales trends, product category performance, regional sales, and customer segment analysis.

---

### 📈 Forecast Explorer

<img width="1917" height="832" alt="image" src="https://github.com/user-attachments/assets/f3893db3-6d88-4ccb-83f3-d7d4aa8e5611" />
Forecasts generated using SARIMA, Prophet, and XGBoost models.

---

### 🚨 Anomaly Report

<img width="1888" height="835" alt="image" src="https://github.com/user-attachments/assets/345cd920-c9b7-4252-9c40-7b746d56a8a7" />
Detects unusual sales patterns using Isolation Forest and Z-Score anomaly detection techniques, helping identify sales spikes, drops, and potential business outliers through interactive visualizations.

---

## Features

- Sales trend analysis
- Exploratory Data Analysis (EDA)
- Time Series Forecasting
- SARIMA Forecasting
- Prophet Forecasting
- XGBoost Forecasting
- Model Performance Comparison
- Category & Region-wise Forecasting
- Anomaly Detection
- Product Demand Segmentation
- Interactive Streamlit Dashboard

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Statsmodels
- Prophet
- XGBoost

## Dataset

- Sample Superstore Dataset
- Historical sales transactions from 2015–2018
- Used for demand forecasting, anomaly detection, and product segmentation.

## Project Structure

```
├── app.py
├── requirements.txt
├── README.md
├── data/
├── charts/
├── notebooks/
└── report/
```

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run app.py
```

## Models Evaluated

- SARIMA
- Prophet
- XGBoost

SARIMA achieved the best forecasting performance based on MAE, RMSE, and MAPE and was selected as the final deployment model.

## Dashboard

The Streamlit dashboard provides:

- Sales Overview
- Forecast Explorer
- Anomaly Report
- Product Demand Segments
- Business Insights

## Author

**Riya Dobhal**

B.Tech CSE (AI & ML)

Academic Machine Learning Project


