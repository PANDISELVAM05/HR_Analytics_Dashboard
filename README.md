# HR_Analytics_Dashboard
# Identify Trends, Patterns & Extract Insights
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("data/processed/products_model_ready.csv")

1) Correlations & feature importance proxy
corr = df.corr(numeric_only=True)["price_log"].sort_values(ascending=False)
print("Top drivers of price_log:\\n", corr.head())

2) Simple baseline prediction
X = df.drop(columns=["price_log"])
y = df["price_log"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, pred))
print("R2:", r2_score(y_test, pred))


# End-to-End Data Analytics Pipeline + EDA (Exploratory Data Analysis) 🚀

In today’s world, raw data is everywhere — but turning it into actionable insights requires a structured pipeline. Recently, I worked on a project that combines data collection, cleaning, transformation, visualization, EDA, and insights extraction into one workflow. Here’s what I built:

# 1. Data Collection

Web Scraping with BeautifulSoup & Requests to extract product/HR data

APIs to pull structured JSON datasets with pagination and authentication
📌 Ensures both real-time and historical data availability.

# 2. Data Cleaning (Pandas & NumPy)

Removed duplicates & fixed inconsistent formats

Treated missing values and outliers

Standardized column names, types, and units
📌 Clean data is the foundation of trustworthy analysis.

# 3. Data Transformation

Feature engineering (log-transforms, premium flag, new KPIs)

Scaling (StandardScaler) & categorical encoding (One-Hot)

Applied business rules to create meaningful metrics
📌 Prepares the dataset for both visualization and modeling.

#  4. Exploratory Data Analysis (EDA)
Before jumping into predictions, I explored the data deeply to understand relationships and detect patterns:

Univariate Analysis → Distributions of numerical features (histograms, boxplots)

Bivariate Analysis → Scatterplots & bar charts (e.g., price vs rating, department vs attrition)

Multivariate Analysis → Correlation heatmaps, pairplots, cluster tendencies

Outlier & Anomaly Detection → Identified unusual behaviors impacting averages

Trend/Seasonality Checks (if time-series data available) → Observed demand spikes or attrition cycles
📌 EDA helps to ask the right questions before building predictive models.

#  5. Data Visualization

Matplotlib & Seaborn → Histograms, scatterplots, heatmaps, pairplots

Power BI & Tableau → Dashboards for business stakeholders (HR.pbix example)
📌 Visualization makes patterns and anomalies clear at a glance.

#  6. Insights & Future Predictions
Through EDA + visualization, I extracted key insights:

Correlation: Strong/weak drivers behind KPIs (e.g., employee tenure vs attrition, discount depth vs sales)

Patterns: Seasonal demand peaks, department-specific attrition trends

Forecasting: Used Linear Regression & ARIMA as baselines for prediction
📌 Actionable insights drive decisions such as pricing strategy, workforce planning, and promotion optimization.

#  Key Learnings

EDA is the bridge between raw data and predictive modeling

A structured pipeline ensures clean → transformed → insightful data

Combining Python (EDA/ML) + BI tools (dashboards) gives both analytical depth and stakeholder accessibility
