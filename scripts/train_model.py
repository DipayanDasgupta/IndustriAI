import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

# Define paths
data_path = "../data/collected_data.csv"
model_path = "../models/esg_model.pkl"
feature_columns_path = "../models/feature_columns.pkl"

# Create necessary directories if not exist
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# Load the dataset
print("Loading data...")
esg_data = pd.read_csv(data_path)
print("Data loaded successfully.")

# Data Cleaning
if "Value" not in esg_data.columns:
    raise KeyError("The dataset must contain a 'Value' column for the ESG score.")
esg_data = esg_data.dropna()

# Normalize Year
esg_data["Year"] = esg_data["Year"] - esg_data["Year"].min()

# Encode categorical features
print("Encoding features...")
esg_data_encoded = pd.get_dummies(esg_data[["Country Code", "Indicator", "Year"]], drop_first=True)

# Save feature columns for future predictions
feature_columns = esg_data_encoded.columns.tolist()
with open(feature_columns_path, "wb") as f:
    pickle.dump(feature_columns, f)
print(f"Feature columns saved to {feature_columns_path}")

# Scale features and target
scaler = MinMaxScaler()
X = scaler.fit_transform(esg_data_encoded)
y = scaler.fit_transform(esg_data["Value"].values.reshape(-1, 1)).ravel()

# Train-test split
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
print("Training model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
print("Evaluating model...")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")

# Save the model
print("Saving the model...")
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"Model saved to {model_path}")
