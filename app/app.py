from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
from scipy.optimize import linprog

app = Flask(__name__)

# Load models and data
model_path = "../models/esg_model.pkl"
with open(model_path, "rb") as f:
    esg_model = pickle.load(f)

optimization_model_path = "../models/optimization_model.pkl"
with open(optimization_model_path, "rb") as f:
    optimization_model = pickle.load(f)

esg_data_path = "../data/collected_data.csv"
project_data_path = "../data/projects.csv"

esg_df = pd.read_csv(esg_data_path)
project_df = pd.read_csv(project_data_path)

# Preprocess the data for one-hot encoding reference
encoded_features = pd.get_dummies(esg_df[["Country Code", "Indicator", "Year"]], drop_first=True)

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect inputs from form or API
        country_code = request.form.get("country_code", "IND")
        indicator = request.form.get("indicator", "SP.POP.TOTL")
        year = int(request.form.get("year", 2022))

        # Prepare input data for prediction
        input_data = pd.DataFrame({
            "Country Code": [country_code],
            "Indicator": [indicator],
            "Year": [year]
        })
        encoded_input = pd.get_dummies(input_data, drop_first=True).reindex(columns=encoded_features.columns, fill_value=0)

        # Make prediction
        esg_score = esg_model.predict(encoded_input)[0]

        return jsonify({"esg_score": esg_score})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/search", methods=["GET"])
def search():
    try:
        # Get search parameters
        country = request.args.get("country", "").strip()
        indicator = request.args.get("indicator", "").strip()
        year = request.args.get("year", "").strip()

        # Filter the dataframe
        filtered_df = esg_df.copy()
        if country:
            filtered_df = filtered_df[filtered_df["Country"].str.contains(country, case=False, na=False)]
        if indicator:
            filtered_df = filtered_df[filtered_df["Indicator"].str.contains(indicator, case=False, na=False)]
        if year:
            filtered_df = filtered_df[filtered_df["Year"] == int(year)]

        # Convert filtered data to JSON
        filtered_data = filtered_df.to_dict(orient="records")

        return jsonify({"data": filtered_data})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/optimize", methods=["GET"])
def optimize():
    try:
        budgets = project_df['Budget']
        risk_scores = project_df['Risk Score']

        # Optimization problem: Maximize budgets under risk constraints
        c = -1 * budgets  # Objective function (negated for maximization)
        A = [risk_scores]  # Risk constraints
        b = [1.0]  # Example risk tolerance
        bounds = [(0, 1) for _ in budgets]  # Binary project selection

        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

        if result.success:
            selected_projects = [i for i, x in enumerate(result.x) if x > 0.5]
            optimized_projects = project_df.iloc[selected_projects]
            optimized_data = optimized_projects.to_dict(orient="records")
            return jsonify({"optimized_projects": optimized_data})
        else:
            return jsonify({"error": "Optimization failed"})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/project_eval", methods=["GET"])
def project_eval():
    try:
        # Display all project data with scores
        project_scores = project_df.copy()
        project_scores["ESG Impact Score"] = optimization_model.predict(project_scores[["Budget", "Risk Score"]])
        return jsonify({"projects": project_scores.to_dict(orient="records")})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
