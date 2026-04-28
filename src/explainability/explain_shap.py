from pathlib import Path
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from src.utils.cleaning import clean_feature_names

X = pd.read_csv("data/gse63060/processed/X.csv", index_col=0)
X = clean_feature_names(X)

model = joblib.load("models/gse63060_xgboost/xgboost.joblib")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

OUT = Path("reports/shap")
OUT.mkdir(parents=True, exist_ok=True)

plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.savefig(OUT / "shap_summary.png", bbox_inches="tight", dpi=200)
plt.close()

importance = pd.DataFrame({
    "GeneID": X.columns,
    "mean_abs_shap": abs(shap_values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)

importance.to_csv(OUT / "shap_top_features.csv", index=False)

print("SHAP saved to reports/shap/")
print(importance.head(20))
