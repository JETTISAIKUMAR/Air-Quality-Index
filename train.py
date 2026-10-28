
# Step 1: Import necessary packages

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib

# Step 2: Read CSV file

data = Path("dataset/city_day.csv")
df = pd.read_csv(data)

# Step 3: Perform EDA

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

# Step 4: Select input features and target

features = [
    "PM2.5", "PM10", "NO2", "NH3",
    "CO", "SO2", "O3"
]

target = "AQI"

# Keep only the required columns
df = df[features + [target]].copy()

# Convert columns to numeric and handle missing values
for col in features + [target]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()

print("\nDataset Shape After Cleaning:", df.shape)

# Step 5: Separate input and output

X = df[features]
y = df[target]

# Step 6: Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 7: Train Random Forest Regressor

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Step 8: Predict AQI

y_pred = model.predict(X_test)

# Step 9: Evaluate the model

print("\nModel Evaluation")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

# Step 10: Save the trained model

joblib.dump(model, "aqi_random_forest.pkl")
print("\nModel saved as aqi_random_forest.pkl")

# Step 11: Predict AQI for a sample input

sample = pd.DataFrame([{
    "PM2.5": 35.0,
    "PM10": 60.0,
    "NO2": 15.0,
    "NH3": 10.0,
    "CO": 0.5,
    "SO2": 8.0,
    "O3": 25.0
}])

prediction = model.predict(sample)[0]
print("\nPredicted AQI:", round(prediction, 2))