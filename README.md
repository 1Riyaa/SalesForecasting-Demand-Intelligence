# Sales Forecasting & Demand Intelligence System

## Live Demo

Streamlit App:
https://salesforecasting-demand-intelligence-zkpscvtdfkzyqjkuo2tvgy.streamlit.app/

## Overview

This project presents an end-to-end sales forecasting and demand intelligence solution developed using the Superstore dataset. It analyzes historical sales trends, compares multiple forecasting models, detects anomalies, segments products based on demand patterns, and provides an interactive dashboard for business decision-making.

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


