# HR_Analytics_Dashboard
# Identify Trends, Patterns & Extract Insights
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("data/processed/products_model_ready.csv")

# 1) Correlations & feature importance proxy
corr = df.corr(numeric_only=True)["price_log"].sort_values(ascending=False)
print("Top drivers of price_log:\\n", corr.head())

# 2) Simple baseline prediction
X = df.drop(columns=["price_log"])
y = df["price_log"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, pred))
print("R2:", r2_score(y_test, pred))
