"""
Movie Rating Prediction
------------------------
Predicts IMDb rating of Indian movies using Genre, Director, Actors,
Year, Duration and Votes as features.

Run this file directly in VS Code (or `python movie_rating_prediction.py`
from a terminal). It will:
  1. Load and clean data/IMDb Movies India.csv
  2. Engineer features (frequency + target encoding for categoricals)
  3. Train several regression models
  4. Print evaluation metrics and the most important features
  5. Save the best model to model/movie_rating_model.pkl
  6. Save a predictions-vs-actual plot to outputs/predictions_vs_actual.png
"""

import warnings
warnings.filterwarnings("ignore")

import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "data/IMDb Movies India.csv"
MODEL_PATH = "model/movie_rating_model.pkl"
PLOT_PATH = "outputs/predictions_vs_actual.png"
RANDOM_STATE = 42


# --------------------------------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin1")
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# --------------------------------------------------------------------------
# 2. CLEAN DATA
# --------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Rating is our target -> drop rows where it's missing
    df = df.dropna(subset=["Rating"])

    # Year: "(2019)" -> 2019
    df["Year"] = df["Year"].astype(str).str.extract(r"(\d{4})").astype(float)

    # Duration: "109 min" -> 109
    df["Duration"] = (
        df["Duration"].astype(str).str.extract(r"(\d+)").astype(float)
    )

    # Votes: "1,086" -> 1086
    df["Votes"] = (
        df["Votes"].astype(str).str.replace(",", "", regex=False)
    )
    df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce")

    # Fill remaining numeric NaNs with median
    for col in ["Year", "Duration", "Votes"]:
        df[col] = df[col].fillna(df[col].median())

    # Fill missing categorical text with a placeholder
    for col in ["Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    print(f"After cleaning: {df.shape[0]} rows remain (rows without a Rating were dropped)")
    return df


# --------------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# --------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Genre: split into a list, then multi-hot encode the top N genres ---
    df["Genre_list"] = df["Genre"].apply(
        lambda g: [x.strip() for x in g.split(",")] if g != "Unknown" else ["Unknown"]
    )
    df["Num_Genres"] = df["Genre_list"].apply(len)

    top_genres = (
        pd.Series([g for genres in df["Genre_list"] for g in genres])
        .value_counts()
        .head(15)
        .index
    )
    for genre in top_genres:
        df[f"Genre_{genre}"] = df["Genre_list"].apply(lambda genres: int(genre in genres))

    # --- Director / Actor: target-mean encoding (smoothed) ---
    # Encodes each Director/Actor as the average rating of movies they've been
    # in historically. Smoothing pulls rare people toward the global mean so
    # a director with 1 movie doesn't get an extreme, overfit score.
    global_mean = df["Rating"].mean()

    def smoothed_target_encode(series: pd.Series, target: pd.Series, m: int = 5):
        stats = target.groupby(series).agg(["mean", "count"])
        smoothed = (stats["mean"] * stats["count"] + global_mean * m) / (stats["count"] + m)
        return series.map(smoothed).fillna(global_mean)

    df["Director_encoded"] = smoothed_target_encode(df["Director"], df["Rating"])
    df["Actor1_encoded"] = smoothed_target_encode(df["Actor 1"], df["Rating"])
    df["Actor2_encoded"] = smoothed_target_encode(df["Actor 2"], df["Rating"])
    df["Actor3_encoded"] = smoothed_target_encode(df["Actor 3"], df["Rating"])

    # --- Frequency features: how prolific is this director/actor? ---
    df["Director_freq"] = df["Director"].map(df["Director"].value_counts())
    df["Actor1_freq"] = df["Actor 1"].map(df["Actor 1"].value_counts())

    return df


def build_feature_matrix(df: pd.DataFrame):
    genre_cols = [c for c in df.columns if c.startswith("Genre_") and c != "Genre_list"]
    feature_cols = [
        "Year",
        "Duration",
        "Votes",
        "Num_Genres",
        "Director_encoded",
        "Actor1_encoded",
        "Actor2_encoded",
        "Actor3_encoded",
        "Director_freq",
        "Actor1_freq",
    ] + genre_cols

    X = df[feature_cols]
    y = df["Rating"]
    return X, y, feature_cols


# --------------------------------------------------------------------------
# 4. TRAIN & EVALUATE MODELS
# --------------------------------------------------------------------------
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    cv_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean()

    print(f"\n{name}")
    print(f"  MAE           : {mae:.3f}")
    print(f"  RMSE          : {rmse:.3f}")
    print(f"  R^2 (test)    : {r2:.3f}")
    print(f"  R^2 (5-fold CV): {cv_r2:.3f}")

    return {"name": name, "model": model, "mae": mae, "rmse": rmse, "r2": r2, "preds": preds}


def main():
    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = engineer_features(df)
    X, y, feature_cols = build_feature_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE
        ),
    }

    print("\n=== Model Comparison ===")
    results = [
        evaluate_model(name, model, X_train, X_test, y_train, y_test)
        for name, model in models.items()
    ]

    best = max(results, key=lambda r: r["r2"])
    print(f"\nBest model: {best['name']} (R^2 = {best['r2']:.3f})")

    # Feature importance (tree-based models only)
    if hasattr(best["model"], "feature_importances_"):
        importances = pd.Series(
            best["model"].feature_importances_, index=feature_cols
        ).sort_values(ascending=False)
        print("\nTop 10 most important features:")
        print(importances.head(10).to_string())

    # Save the best model
    import os
    os.makedirs("model", exist_ok=True)
    joblib.dump({"model": best["model"], "feature_cols": feature_cols}, MODEL_PATH)
    print(f"\nSaved best model to {MODEL_PATH}")

    # Plot predictions vs actual
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, best["preds"], alpha=0.4, s=15)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Actual Rating")
    plt.ylabel("Predicted Rating")
    plt.title(f"Predicted vs Actual Rating ({best['name']})")
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"Saved plot to {PLOT_PATH}")


if __name__ == "__main__":
    main()
