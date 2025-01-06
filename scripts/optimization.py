import pandas as pd
from scipy.optimize import linprog

def optimize_portfolio():
    project_data = pd.read_csv("../data/projects.csv")
    budgets = project_data['Budget']
    risk_scores = project_data['Risk Score']

    c = -1 * budgets  # Objective: Maximize budgets (converted to minimization)
    A = [risk_scores]  # Constraints
    b = [1.0]  # Risk tolerance (example constraint)
    bounds = [(0, 1) for _ in budgets]  # Project selection (binary decision)

    result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

    if result.success:
        selected_projects = [i for i, x in enumerate(result.x) if x > 0.5]
        print("Selected projects:", project_data.iloc[selected_projects])
    else:
        print("Optimization failed.")

if __name__ == "__main__":
    optimize_portfolio()
