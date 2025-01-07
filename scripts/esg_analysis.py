import pandas as pd
import pickle
import os

# Define paths
model_path = "../models/esg_model.pkl"
feature_columns_path = "../models/feature_columns.pkl"
scaler_path = "../models/scaler.pkl"  # Assuming MinMaxScaler is saved during training
data_path = "../data/collected_data.csv"

# Load Model and Features
if not os.path.exists(model_path) or not os.path.exists(feature_columns_path):
    raise FileNotFoundError("Required files (model or feature columns) are missing. Train the model first.")
print("Loading model and feature columns...")
with open(model_path, "rb") as f:
    esg_model = pickle.load(f)

with open(feature_columns_path, "rb") as f:
    feature_columns = pickle.load(f)

if os.path.exists(scaler_path):
    print("Loading scaler...")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
else:
    scaler = None

print("Files loaded successfully.")

# ESG Scoring Function
def predict_esg(country_code, indicator, year):
    # Prepare input data
    input_data = pd.DataFrame({
        "Country Code": [country_code],
        "Indicator": [indicator],
        "Year": [year]
    })

    # Normalize Year (as done in training)
    input_data["Year"] = input_data["Year"] - pd.read_csv(data_path)["Year"].min()

    # Encode and align with training features
    encoded_input = pd.get_dummies(input_data, drop_first=True).reindex(columns=feature_columns, fill_value=0)

    # Scale input data if scaler exists
    if scaler:
        encoded_input = scaler.transform(encoded_input)

    # Predict ESG Score
    esg_score = esg_model.predict(encoded_input)
    return esg_score[0]

# Example Usage
if __name__ == "__main__":
    country_code = "IND"
    indicator = "SP.POP.TOTL"
    year = 2022

    try:
        print(f"Predicting ESG score for {country_code}, {indicator}, {year}...")
        score = predict_esg(country_code, indicator, year)
        print(f"Predicted ESG Score: {score}")
    except Exception as e:
        print(f"Error during prediction: {e}")
