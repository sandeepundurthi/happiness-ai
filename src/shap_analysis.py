import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/happiness_clean.csv")
model = joblib.load("models/happiness_xgb.pkl")

features = [
    "explained_log_gdp_per_capita",
    "explained_social_support",
    "explained_healthy_life_expectancy",
    "explained_freedom",
    "explained_generosity",
    "explained_corruption",
    "dystopia_plus_residual"
]

X = df[features]

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Summary plot
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig("reports/shap_summary.png", dpi=300)
print("Saved SHAP summary plot.")
