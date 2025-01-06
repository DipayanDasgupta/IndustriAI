import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

# Train ESG Scoring Model
esg_data = pd.read_csv("../data/collected_data.csv")
X_esg = esg_data[['Year']].astype(float)  # Example feature
y_esg = esg_data['Value']

X_train_esg, X_test_esg, y_train_esg, y_test_esg = train_test_split(X_esg, y_esg, test_size=0.2, random_state=42)
esg_model = RandomForestRegressor(n_estimators=100, random_state=42)
esg_model.fit(X_train_esg, y_train_esg)

with open("../models/esg_model.pkl", "wb") as f:
    pickle.dump(esg_model, f)
print("ESG model trained and saved.")

# Train Optimization Model (example)
project_data = pd.read_csv("../data/projects.csv")
X_proj = project_data[['Budget', 'Risk Score']]
y_proj = project_data['Budget']  # Placeholder for a target variable

X_train_proj, X_test_proj, y_train_proj, y_test_proj = train_test_split(X_proj, y_proj, test_size=0.2, random_state=42)
project_model = RandomForestRegressor(n_estimators=100, random_state=42)
project_model.fit(X_train_proj, y_train_proj)

with open("../models/optimization_model.pkl", "wb") as f:
    pickle.dump(project_model, f)
print("Optimization model trained and saved.")
