from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import os
from scripts.esg_analysis import predict_esg
from scripts.optimization import optimize_projects

app = Flask(__name__)

# Load Models
models_path = "../models"

esg_model_path = os.path.join(models_path, "esg_model.pkl")
optimization_model_path = os.path.join(models_path, "optimization_model.pkl")

if not os.path.exists(esg_model_path):
    raise FileNotFoundError("The ESG model file is missing.")
if not os.path.exists(optimization_model_path):
    raise FileNotFoundError("The optimization model file is missing.")

with open(esg_model_path, "rb") as f:
    esg_model = pickle.load(f)

with open(optimization_model_path, "rb") as f:
    optimization_model = pickle.load(f)

# Routes
@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        country_code = request.form.get("country_code")
        indicator = request.form.get("indicator")
        year = int(request.form.get("year"))

        esg_score = predict_esg(country_code, indicator, year)
        return jsonify({"esg_score": esg_score})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/search", methods=["GET"])
def search():
    try:
        country = request.args.get("country", "").strip()
        indicator = request.args.get("indicator", "").strip()
        year = request.args.get("year", "").strip()

        esg_data_path = "../data/esg_data.csv"
        if not os.path.exists(esg_data_path):
            return jsonify({"error": "ESG data file is missing."})

        data = pd.read_csv(esg_data_path)

        if country:
            data = data[data["Country"].str.contains(country, case=False, na=False)]
        if indicator:
            data = data[data["Indicator"].str.contains(indicator, case=False, na=False)]
        if year:
            data = data[data["Year"] == int(year)]

        return jsonify({"data": data.to_dict(orient="records")})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/optimize", methods=["POST"])
def optimize():
    try:
        budget_constraint = float(request.form.get("budget_constraint", 1.0))
        risk_tolerance = float(request.form.get("risk_tolerance", 1.0))

        projects_path = "../data/projects.csv"
        if not os.path.exists(projects_path):
            raise FileNotFoundError("The project data file is missing.")

        project_data = pd.read_csv(projects_path)

        if "Budget" not in project_data.columns or "Risk Score" not in project_data.columns:
            raise KeyError("Project data must contain 'Budget' and 'Risk Score' columns.")

        result = optimize_projects(project_data, budget_constraint, risk_tolerance)

        if result.success:
            selected_projects = [i for i, x in enumerate(result.x) if x > 0.5]
            selected_data = project_data.iloc[selected_projects]
            return jsonify({"selected_projects": selected_data.to_dict(orient="records")})
        else:
            return jsonify({"error": "Optimization failed. Please adjust constraints."})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
