from pathlib import Path
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

PROCESSED = Path("data/processed")
FIGS = Path("reports/figures")
TABLES = Path("reports/tables")
FIGS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

X = pd.read_csv("data/gse63060/processed/X.csv", index_col=0)
model = joblib.load("models/gse63060_xgboost/xgboost.joblib")

explainer = shap.Explainer(model, X)
shap_values = explainer(X)

shap.plots.beeswarm(shap_values, show=False)
plt.tight_layout()
plt.savefig(FIGS / "shap_beeswarm.png", dpi=200)
plt.close()

mean_abs = abs(shap_values.values).mean(axis=0)
importance = pd.DataFrame({
    "GeneID": X.columns,
    "mean_abs_shap": mean_abs
}).sort_values("mean_abs_shap", ascending=False)

importance.to_csv(TABLES / "top_shap_genes.csv", index=False)
print(importance.head(20))