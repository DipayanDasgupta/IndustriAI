import pandas as pd
from scipy.optimize import linprog
import pickle
import os
from scripts.esg_analysis import predict_esg  # Import the ESG prediction function

# File paths
project_data_path = "../data/projects.csv"
optimization_model_path = "../models/optimization_model.pkl"

# Load Data
if not os.path.exists(project_data_path):
    raise FileNotFoundError("The project data file is missing.")
print("Loading project data...")
project_data = pd.read_csv(project_data_path)
print("Project data loaded successfully.")

# Preprocessing
if "Budget" not in project_data.columns or "Risk Score" not in project_data.columns:
    raise KeyError("Project data must contain 'Budget' and 'Risk Score' columns.")

# Predict ESG Scores (Optional Integration)
if "Country Code" in project_data.columns and "Indicator" in project_data.columns and "Year" in project_data.columns:
    print("Predicting ESG Scores for projects...")
    project_data["ESG Score"] = project_data.apply(
        lambda row: predict_esg(row["Country Code"], row["Indicator"], row["Year"]), axis=1
    )
else:
    print("Project data does not contain ESG-related columns. Skipping ESG Score prediction.")

# Optimization Setup
def optimize_projects(data, budget_constraint=1.0, risk_tolerance=1.0):
    budgets = data["Budget"].values
    risk_scores = data["Risk Score"].values

    # Add ESG scores as an additional constraint, if available
    if "ESG Score" in data.columns:
        esg_scores = data["ESG Score"].values
    else:
        esg_scores = None

    # Objective: Maximize project budgets
    c = -1 * budgets

    # Constraints
    A = [risk_scores]  # Risk constraint
    b = [risk_tolerance]

    if esg_scores is not None:
        A.append(esg_scores)
        b.append(budget_constraint)  # Adjust ESG-based constraint if needed

    # Bounds: Binary decision variables (0 or 1)
    bounds = [(0, 1) for _ in budgets]

    # Solve optimization problem
    print("Running optimization...")
    result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    return result

# Run Optimization
result = optimize_projects(project_data, budget_constraint=0.8, risk_tolerance=1.0)
if result.success:
    selected_projects = [i for i, x in enumerate(result.x) if x > 0.5]
    selected_data = project_data.iloc[selected_projects]
    print("Selected Projects:")
    print(selected_data)

    # Save results for Flask integration
    with open(optimization_model_path, "wb") as f:
        pickle.dump(result, f)
    print(f"Optimization model saved to {optimization_model_path}.")
else:
    print("Optimization failed.")
