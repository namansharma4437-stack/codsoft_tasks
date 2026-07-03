import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# ==========================================================
# 1. Load Dataset
# ==========================================================
df = pd.read_csv("Titanic-Dataset.csv")

print("Dataset Shape:", df.shape)
print("\nMissing Values:\n")
print(df.isnull().sum())

# ==========================================================
# 2. Feature Engineering
# ==========================================================

# Extract Title
df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")

# Merge uncommon titles
title_mapping = {
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs"
}
df["Title"] = df["Title"].replace(title_mapping)

rare_titles = df["Title"].value_counts()
rare_titles = rare_titles[rare_titles < 10].index

df["Title"] = df["Title"].replace(rare_titles, "Rare")

# Family Features
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# Cabin Deck
df["Cabin"] = df["Cabin"].fillna("Unknown")
df["Deck"] = df["Cabin"].str[0]

# Fill Age using Title median
df["Age"] = df.groupby("Title")["Age"].transform(
    lambda x: x.fillna(x.median())
)

df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill Embarked
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Fill Fare
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# Age Categories
df["AgeBin"] = pd.cut(
    df["Age"],
    bins=[0, 12, 20, 40, 60, 100],
    labels=["Child", "Teen", "Adult", "MiddleAge", "Senior"],
    include_lowest=True
)

# Fare Categories
df["FareBin"] = pd.qcut(
    df["Fare"],
    q=4,
    labels=["Low", "Mid", "High", "VeryHigh"],
    duplicates="drop"
)

# ==========================================================
# 3. Feature Selection
# ==========================================================

features = [
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "Embarked",
    "Title",
    "FamilySize",
    "IsAlone",
    "Deck",
    "AgeBin",
    "FareBin"
]

X = df[features].copy()
y = df["Survived"]

# One-Hot Encoding
categorical_cols = [
    "Sex",
    "Embarked",
    "Title",
    "Deck",
    "AgeBin",
    "FareBin"
]

X = pd.get_dummies(
    X,
    columns=categorical_cols,
    drop_first=True
)

# Final check
print("\nRemaining Missing Values:", X.isnull().sum().sum())

# ==========================================================
# 4. Train Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# ==========================================================
# 5. Logistic Regression
# ==========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr.fit(X_train_scaled, y_train)

lr_pred = lr.predict(X_test_scaled)
lr_prob = lr.predict_proba(X_test_scaled)[:, 1]

# ==========================================================
# 6. Random Forest
# ==========================================================

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]

# ==========================================================
# 7. Evaluation Function
# ==========================================================

def evaluate(name, y_true, y_pred, y_prob):

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print("Accuracy :", round(accuracy_score(y_true, y_pred), 4))
    print("ROC AUC  :", round(roc_auc_score(y_true, y_prob), 4))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report")
    print(classification_report(
        y_true,
        y_pred,
        target_names=["Died", "Survived"]
    ))

evaluate("Logistic Regression", y_test, lr_pred, lr_prob)
evaluate("Random Forest", y_test, rf_pred, rf_prob)

# ==========================================================
# 8. Cross Validation
# ==========================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    rf,
    X,
    y,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

print("\nCross Validation Accuracy")
print("Mean :", scores.mean())
print("Std  :", scores.std())

# ==========================================================
# 9. Feature Importance
# ==========================================================

importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 10 Important Features\n")
print(importance.head(10))