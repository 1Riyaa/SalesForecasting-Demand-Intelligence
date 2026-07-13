import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="Sales Forecasting & Demand Intelligence",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("data/train.csv")

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        format="mixed",
        dayfirst=True
    )

    df["Ship Date"] = pd.to_datetime(
        df["Ship Date"],
        format="mixed",
        dayfirst=True
    )

    return df

@st.cache_data
def load_forecast(segment):
    path = f"data/segment_forecasts/{segment}.csv"
    return pd.read_csv(path, parse_dates=["Date"])

df = load_data()

# Sidebar

st.sidebar.title("📊 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "📊 Sales Overview",
        "📈 Exploratory Analysis",
        "📈 Forecast Explorer",
        "🚨 Anomaly Report",
        "📦 Product Segments",
        "ℹ️ About"
    ]
)

# HOME PAGE

if page == "🏠 Home":

    st.title("📈 Sales Forecasting & Demand Intelligence System")

    st.markdown("""
    ## Welcome

    This dashboard presents an end-to-end Sales Forecasting project developed using multiple forecasting techniques.

    ### Features

    - Exploratory Data Analysis
    - Sales Trend Visualization
    - SARIMA Forecasting
    - Prophet Forecasting
    - XGBoost Forecasting
    - Model Performance Comparison
    - Business Insights

    ---
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Dataset", "Superstore")
    col2.metric("Forecast Models", "3")
    col3.metric("Forecast Horizon", "12 Months")

# PLACEHOLDERS

elif page == "📊 Sales Overview":

    st.title("📊 Sales Overview Dashboard")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Total Sales by Year")

    yearly_sales = (
        df.groupby(df["Order Date"].dt.year)["Sales"].sum()
        )

    st.bar_chart(yearly_sales)

    st.subheader("Monthly Sales Trend")

    monthly_sales = (
        df.groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"].sum()
        )

    st.line_chart(monthly_sales)

    st.subheader("Sales by Region") 

    region = st.selectbox(
        "Select Region",sorted(df["Region"].unique())
        )

    region_df = df[df["Region"] == region]

    region_sales = (
        region_df.groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"].sum()
        )

    st.line_chart(region_sales)

    st.subheader("Sales by Category")

    category = st.selectbox(
        "Select Category",sorted(df["Category"].unique())
        )

    cat_df = df[df["Category"] == category]

    cat_sales = (
        cat_df.groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"].sum()
    )

    st.line_chart(cat_sales)

elif page == "📈 Exploratory Analysis":

    st.title("📈 Exploratory Business Analysis")

    st.markdown("""
    Explore historical sales performance through interactive visualizations,
    including trends, regional performance, customer segments, and product analysis.
    """)

    chart_folder = "charts"

    def show_chart(title, filename):
        st.subheader(title)
        path = os.path.join(chart_folder, filename)

        if os.path.exists(path):
            st.image(path, use_container_width=True)
        else:
            st.warning(f"{filename} not found.")

    col1, col2 = st.columns(2)

    with col1:
        show_chart("Monthly Sales Trend", "monthly_sales_trend.png")

    with col2:
        show_chart("Sales by Product Category", "sales_by_category.png")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        show_chart("Sales by Region", "sales_by_region.png")

    with col2:
        show_chart("Sales by Customer Segment", "sales_by_segment.png")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        show_chart("Monthly Sales by Year", "monthly_sales_by_year.png")

    with col2:
        show_chart("Top 10 Products", "top10_products.png")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        show_chart("Monthly Sales Time Series", "monthly_time_series.png")

    with col2:
        show_chart("Time Series Decomposition", "time_series_decomposition.png")

elif page == "📈 Forecast Explorer":

    st.title("📈 Forecast Explorer")

    st.markdown("""
    Compare forecasting results generated using three different forecasting models.
    Each model includes forecast visualization, evaluation results, and business interpretation.
    """)


    chart_folder = "charts"

    sarima_tab, prophet_tab, xgb_tab = st.tabs(
        ["📈 SARIMA", "🔮 Prophet", "⚡ XGBoost"]
    )

    # ==========================
    # SARIMA
    # ==========================

    with sarima_tab:

        st.header("SARIMA Forecast")

        forecast_horizon = st.selectbox(
            "Forecast Horizon",
            ["1 Month", "2 Months", "3 Months"]
        )

        segment = st.selectbox(
            "Select Forecast Series",
            [
                "Furniture",
                "Office Supplies",
                "Technology",
                "East Region",
                "West Region"
            ]
        )

        segment_files = {
            "Furniture": "furniture",
            "Office Supplies": "office_supplies",
            "Technology": "technology",
            "East Region": "east_region",
            "West Region": "west_region"
        }

        forecast_df = load_forecast(segment_files[segment])

        months = {
            "1 Month": 1,
            "2 Months": 2,
            "3 Months": 3
        }

        display_df = forecast_df.head(months[forecast_horizon])

        st.subheader("Selected Forecast")

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        st.line_chart(
            display_df.set_index("Date")["Forecast"]
        )

        if os.path.exists(f"{chart_folder}/sarima_forecast.png"):
            st.image(f"{chart_folder}/sarima_forecast.png", use_container_width=True)

        st.subheader("Model Evaluation")

        if os.path.exists(f"{chart_folder}/sarima_evaluation.png"):
            st.image(f"{chart_folder}/sarima_evaluation.png", use_container_width=True)

        st.success("""
Best Model from Time-Series Analysis

MAE : 14860.76

RMSE : 17719.90

MAPE : 26.33%
""")

        st.info("""
Business Insight

SARIMA achieved the lowest forecasting error among all evaluated models,
making it the most suitable forecasting approach for this dataset.
""")

    # ==========================
    # PROPHET
    # ==========================

    with prophet_tab:

        st.header("Prophet Forecast")

        st.info(
            "SARIMA was selected as the final production model because it achieved the lowest "
            "forecasting error (MAE, RMSE, and MAPE). Prophet is presented here as an "
            "evaluated baseline model for comparison."
        )

        if os.path.exists(f"{chart_folder}/prophet_forecast.png"):
            st.image(f"{chart_folder}/prophet_forecast.png", use_container_width=True)

        st.subheader("Trend & Seasonality")

        if os.path.exists(f"{chart_folder}/prophet_components.png"):
            st.image(f"{chart_folder}/prophet_components.png", use_container_width=True)

        st.subheader("Model Evaluation")

        if os.path.exists(f"{chart_folder}/prophet_evaluation.png"):
            st.image(f"{chart_folder}/prophet_evaluation.png", use_container_width=True)

        st.success("""
Performance

MAE : 17360.04

RMSE : 21571.07

MAPE : 28.91%
""")

        st.info("""
Business Insight

Prophet effectively captures long-term trends and yearly seasonality.
Its forecasting accuracy is slightly lower than SARIMA but remains reliable.
""")

    # ==========================
    # XGBOOST
    # ==========================

    with xgb_tab:

        st.header("XGBoost Forecast")

        st.info(
            "XGBoost was evaluated as an alternative forecasting model. Although it captured "
            "nonlinear relationships, its forecasting accuracy was lower than SARIMA, so it "
            "was not selected for deployment."
        )

        if os.path.exists(f"{chart_folder}/xgboost_evaluation.png"):
            st.image(f"{chart_folder}/xgboost_evaluation.png", use_container_width=True)

        st.subheader("Feature Importance")

        if os.path.exists(f"{chart_folder}/xgboost_feature_importance.png"):
            st.image(f"{chart_folder}/xgboost_feature_importance.png", use_container_width=True)

        st.success("""
Performance

MAE : 23598.36

RMSE : 26657.31

MAPE : 34.68%
""")

        st.info("""
Business Insight

XGBoost captured nonlinear relationships using lag features,
but produced the highest forecasting error among the three models.
""")

elif page == "🚨 Anomaly Report":

    st.title("🚨 Anomaly Report")

    st.markdown("""
    Explore anomalous sales periods detected using Isolation Forest
    and Z-Score based anomaly detection methods.
    """)

    chart_folder = "charts"

    tab1, tab2 = st.tabs(
        ["🌲 Isolation Forest", "📈 Z-Score"]
    )

    with tab1:

        st.subheader("Isolation Forest Anomalies")

        if os.path.exists(f"{chart_folder}/anomaly_isolation_forest.png"):
            st.image(
                f"{chart_folder}/anomaly_isolation_forest.png",
                use_container_width=True
            )
        else:
            st.warning("Isolation Forest chart not found.")

        st.info("""
Isolation Forest detects unusual weekly sales patterns using an unsupervised
machine learning approach. These anomalies may correspond to exceptional demand,
seasonal peaks, or unexpected market events.
""")

    with tab2:

        st.subheader("Z-Score Anomalies")

        if os.path.exists(f"{chart_folder}/anomaly_zscore.png"):
            st.image(
                f"{chart_folder}/anomaly_zscore.png",
                use_container_width=True
            )
        else:
            st.warning("Z-Score chart not found.")

        st.info("""
The Z-Score method identifies weeks where sales deviate significantly from
their rolling average. These anomalies provide early indicators of unusual
business activity.
""")

    st.divider()

    st.success("""
Business Recommendation

Monitoring sales anomalies enables organizations to identify abnormal demand
patterns early, optimize inventory decisions, and improve future forecasting
accuracy.
""")
    

elif page == "📦 Product Segments":

    st.title("📦 Product Demand Segments")

    st.markdown("""
    Explore product demand segments identified using K-Means clustering.
    Products are grouped according to their sales characteristics to support
    inventory planning and business decision-making.
    """)

    chart_folder = "charts"

    st.subheader("PCA Cluster Visualization")

    if os.path.exists(f"{chart_folder}/cluster_pca_scatter.png"):
        st.image(
            f"{chart_folder}/cluster_pca_scatter.png",
            use_container_width=True
        )
    else:
        st.warning("PCA visualization not found.")

    st.divider()

    st.subheader("Inventory Strategy")

    strategy = pd.DataFrame({
        "Cluster": [
            "High Volume, Stable Demand",
            "Low Volume, Stable Demand",
            "High Value, High Volatility Demand",
            "Emerging/Growing Niche Demand"
        ],
        "Recommended Strategy": [
            "Maintain regular replenishment with moderate safety stock.",
            "Keep lean inventory and reorder only as required.",
            "Monitor closely and maintain higher safety stock due to volatility.",
            "Track demand carefully and increase inventory gradually if growth continues."
        ]
    })

    st.dataframe(
        strategy,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.success("""
Business Recommendation

Product segmentation enables inventory managers to tailor stocking
strategies according to demand behaviour, reducing inventory costs
while improving product availability.
""")

elif page == "ℹ️ About":

    st.title("ℹ️ About the Project")

    st.markdown("""
    ## Sales Forecasting & Demand Intelligence System

    This project demonstrates an end-to-end data analytics and forecasting workflow
    developed using historical sales data from the Superstore dataset.

    The objective is to analyze historical sales patterns and forecast future demand
    using statistical and machine learning models.
    """)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📂 Dataset")

        st.write("""
- Superstore Sales Dataset
- Historical Orders
- Monthly Sales Aggregation
- Demand Forecasting
""")

        st.subheader("🛠 Technologies")

        st.write("""
- Python
- Pandas
- NumPy
- Matplotlib
- Streamlit
- Statsmodels
- Prophet
- XGBoost
""")

    with col2:

        st.subheader("📈 Forecasting Models")

        st.write("""
- SARIMA
- Prophet
- XGBoost Regression
""")

        st.subheader("📊 Evaluation Metrics")

        st.write("""
- MAE
- RMSE
- MAPE
""")

    st.divider()

    st.subheader("✨ Dashboard Features")

    st.write("""
✔ Dataset Overview

✔ Exploratory Business Analysis

✔ Sales Trend Visualization

✔ SARIMA Forecasting

✔ Prophet Forecasting

✔ XGBoost Forecasting

✔ Model Performance Comparison

✔ Business Insights
""")

    st.divider()

    st.subheader("🎯 Project Outcome")

    st.success("""
The comparative evaluation showed that SARIMA achieved the best forecasting
performance for the Superstore monthly sales dataset based on MAE, RMSE,
and MAPE. The dashboard provides an interactive platform for exploring
historical sales patterns and forecasting future demand.
""")

    st.divider()

    st.caption("Developed as an academic Data Science & Machine Learning project using Streamlit.")

    st.write("""
This project was developed to forecast future sales using:

- SARIMA
- Prophet
- XGBoost

The dashboard was created using Streamlit to provide interactive business insights.
""")