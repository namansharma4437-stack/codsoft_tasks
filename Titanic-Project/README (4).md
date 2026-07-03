# Titanic Survival Prediction 🚢

A machine learning project that predicts whether a passenger survived the Titanic disaster, based on features like age, gender, ticket class, and fare. Built as a beginner-friendly, end-to-end ML pipeline: data cleaning → feature engineering → model training → evaluation.

## 📊 Dataset

The dataset (`Titanic-Dataset.csv`) contains information on 891 passengers from the RMS Titanic, including:

| Column | Description |
|---|---|
| `Survived` | Target variable (0 = Died, 1 = Survived) |
| `Pclass` | Ticket class (1st, 2nd, 3rd) |
| `Sex` | Passenger gender |
| `Age` | Passenger age |
| `SibSp` | # of siblings/spouses aboard |
| `Parch` | # of parents/children aboard |
| `Fare` | Ticket fare |
| `Cabin` | Cabin number |
| `Embarked` | Port of embarkation (C/Q/S) |

## 🛠️ Approach

1. **Data Cleaning**
   - Filled missing `Age` using median age grouped by passenger title (Mr/Mrs/Miss/Master)
   - Filled missing `Embarked` with mode, `Fare` with median
   - Extracted deck letter from `Cabin`, treating missing cabins as "Unknown"

2. **Feature Engineering**
   - Extracted **Title** (Mr, Mrs, Miss, Master, Rare) from passenger names
   - Created **FamilySize** and **IsAlone** from `SibSp` + `Parch`
   - Binned `Age` and `Fare` into categorical ranges
   - One-hot encoded all categorical features

3. **Modeling**
   - Trained two models: **Logistic Regression** and **Random Forest Classifier**
   - Evaluated using accuracy, ROC AUC, confusion matrix, and 5-fold cross-validation

4. **Evaluation**

   | Model | Accuracy | ROC AUC |
   |---|---|---|
   | Logistic Regression | ~84% | — |
   | Random Forest | ~83% | ~0.84 |
   | Random Forest (5-fold CV) | ~83.6% (± 1.3%) | — |

5. **Feature Importance**

   Top predictors of survival: **Sex**, **Title**, **Fare**, **Pclass**, **Age** — consistent with the historical "women and children first" evacuation pattern and the strong effect of socioeconomic class on survival.

## 📁 Project Structure

```
titanic-survival-prediction/
├── Titanic-Dataset.csv       # Raw dataset
├── titanic_model.py          # Full pipeline as a Python script
├── Titanic_Analysis.ipynb    # Same pipeline with EDA visualizations (Jupyter notebook)
└── README.md
```

## ▶️ How to Run

**Requirements:**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

**Run the script version:**
```bash
python titanic_model.py
```

**Run the notebook version (with visualizations):**
Open `Titanic_Analysis.ipynb` in Jupyter or VS Code and run all cells.

## 📈 Key Visualizations (in the notebook)

- Survival rate by sex, class, and embarkation port
- Age and fare distribution by survival outcome
- Correlation heatmap
- Confusion matrices for both models
- Feature importance bar chart

## 🔍 Key Insights

- **Gender was the strongest predictor** — women had a significantly higher survival rate than men.
- **Passenger class mattered** — 1st class passengers survived at nearly double the rate of 3rd class passengers.
- **Fare** (closely tied to class) was also a strong predictor.
- **Age** played a role — children had somewhat better survival odds.

## 🧰 Tech Stack

- Python
- pandas, numpy — data manipulation
- scikit-learn — modeling (Logistic Regression, Random Forest)
- matplotlib, seaborn — visualization

---

*This project was built as part of an internship task, using the classic Titanic dataset for binary classification practice.*
