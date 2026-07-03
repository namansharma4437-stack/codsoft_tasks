"""
Sales Prediction using Machine Learning in Python
==================================================
Predicts product Sales from advertising spend on TV, Radio, and Newspaper.

Pipeline:
1. Load & explore the data
2. Visualize relationships (EDA)
3. Train/test split
4. Train a Linear Regression model (baseline, interpretable)
5. Train a Random Forest model (usually more accurate)
6. Evaluate both models
7. Show feature importance
8. Provide a reusable predict_sales() function
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

sns.set_style("whitegrid")
OUT = "outputs"

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
df = pd.read_csv("advertising.csv")
print("Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nSummary stats:\n", df.describe())
print("\nMissing values:\n", df.isnull().sum())

# ---------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------

# Correlation heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation between Ad Spend and Sales")
plt.tight_layout()
plt.savefig(f"{OUT}/correlation_heatmap.png", dpi=150)
plt.close()

# Scatter plots: each channel vs Sales
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ["TV", "Radio", "Newspaper"]):
    sns.regplot(x=col, y="Sales", data=df, ax=ax,
                scatter_kws={"alpha": 0.5}, line_kws={"color": "red"})
    ax.set_title(f"{col} vs Sales")
plt.tight_layout()
plt.savefig(f"{OUT}/spend_vs_sales.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# ---------------------------------------------------------
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# 4. LINEAR REGRESSION (baseline)
# ---------------------------------------------------------
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
lin_preds = lin_model.predict(X_test)

# ---------------------------------------------------------
# 5. RANDOM FOREST (usually stronger, captures non-linearity)
# ---------------------------------------------------------
rf_model = RandomForestRegressor(n_estimators=300, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# ---------------------------------------------------------
# 6. EVALUATE
# ---------------------------------------------------------
def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n{name}")
    print(f"  MAE  : {mae:.3f}")
    print(f"  RMSE : {rmse:.3f}")
    print(f"  R^2  : {r2:.3f}")
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2}

results = []
results.append(evaluate("Linear Regression", y_test, lin_preds))
results.append(evaluate("Random Forest", y_test, rf_preds))

results_df = pd.DataFrame(results)
results_df.to_csv(f"{OUT}/model_comparison.csv", index=False)
print("\nModel comparison:\n", results_df)

# Actual vs Predicted plot for the better model
best_preds = rf_preds if results[1]["R2"] > results[0]["R2"] else lin_preds
best_name = "Random Forest" if results[1]["R2"] > results[0]["R2"] else "Linear Regression"

plt.figure(figsize=(6, 6))
plt.scatter(y_test, best_preds, alpha=0.7, color="teal")
lims = [min(y_test.min(), best_preds.min()), max(y_test.max(), best_preds.max())]
plt.plot(lims, lims, "r--", label="Perfect prediction")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title(f"Actual vs Predicted Sales ({best_name})")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/actual_vs_predicted.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# 7. FEATURE IMPORTANCE / COEFFICIENTS
# ---------------------------------------------------------
print("\nLinear Regression coefficients (impact per $1000 spend):")
for col, coef in zip(X.columns, lin_model.coef_):
    print(f"  {col}: {coef:.4f}")
print(f"  Intercept: {lin_model.intercept_:.4f}")

importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nRandom Forest feature importances:\n", importances)

plt.figure(figsize=(6, 4))
importances.plot(kind="bar", color="steelblue")
plt.title("Feature Importance (Random Forest)")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUT}/feature_importance.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# 8. SAVE MODEL + PREDICTION HELPER
# ---------------------------------------------------------
joblib.dump(rf_model, f"{OUT}/sales_rf_model.pkl")
joblib.dump(lin_model, f"{OUT}/sales_linear_model.pkl")

def predict_sales(tv, radio, newspaper, model=rf_model):
    """Predict sales given ad spend (in $1000s) on each channel."""
    input_df = pd.DataFrame([[tv, radio, newspaper]], columns=["TV", "Radio", "Newspaper"])
    return model.predict(input_df)[0]

# Example usage
example = predict_sales(tv=150, radio=25, newspaper=20)
print(f"\nExample prediction — TV=150, Radio=25, Newspaper=20 -> Predicted Sales: {example:.2f}")

print("\nAll charts and the model comparison CSV were saved to the 'outputs' folder.")