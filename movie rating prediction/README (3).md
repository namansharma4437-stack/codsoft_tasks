# Movie Rating Prediction (IMDb India)

Predicts a movie's IMDb rating from its Genre, Director, Actors, Year,
Duration, and Vote count.

## Project structure
```
movie-rating-prediction/
├── data/
│   └── IMDb Movies India.csv     ← dataset
├── model/
│   └── movie_rating_model.pkl    ← saved model (created after running)
├── outputs/
│   └── predictions_vs_actual.png ← evaluation plot (created after running)
├── movie_rating_prediction.py    ← main script
├── requirements.txt
└── README.md
```

## How to run this in VS Code

1. **Open the folder in VS Code**
   File → Open Folder → select the `movie-rating-prediction` folder.

2. **Create a virtual environment** (Terminal → New Terminal, then run):
   ```bash
   python -m venv venv
   ```
   Activate it:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Select the interpreter**: press `Ctrl+Shift+P` (Windows/Linux) or
   `Cmd+Shift+P` (Mac) → "Python: Select Interpreter" → choose the `venv`
   one you just created.

5. **Run the script**: open `movie_rating_prediction.py` and press the
   ▶ Run button (top right), or in the terminal:
   ```bash
   python movie_rating_prediction.py
   ```

## What the script does

1. **Load** — reads `data/IMDb Movies India.csv`.
2. **Clean**
   - Drops rows with no `Rating` (that's the target, can't train on it).
   - Parses `Year` (`"(2019)"` → `2019`), `Duration` (`"109 min"` → `109`),
     and `Votes` (`"1,086"` → `1086`) into numbers.
   - Fills remaining gaps (median for numbers, "Unknown" for text).
3. **Feature engineering**
   - Multi-hot encodes the 15 most common genres.
   - Target-encodes `Director` and each `Actor` column: each person is
     represented by the average rating of their past movies, smoothed
     toward the overall average so people with very few movies don't get
     extreme scores.
   - Adds "how many movies has this director/actor been in" as a frequency
     feature.
4. **Model training** — trains and compares Linear Regression, Random
   Forest, and Gradient Boosting, using an 80/20 train-test split plus
   5-fold cross-validation.
5. **Output** — prints MAE / RMSE / R² for each model, picks the best one,
   shows the top 10 most important features, saves the model to
   `model/movie_rating_model.pkl`, and saves a predicted-vs-actual scatter
   plot to `outputs/predictions_vs_actual.png`.

## Current results (this dataset)

| Model             | MAE   | RMSE  | R² (test) |
|-------------------|-------|-------|-----------|
| Linear Regression | 0.64  | 0.82  | 0.63      |
| Random Forest      | 0.43  | 0.65  | **0.77**  |
| Gradient Boosting  | 0.48  | 0.66  | 0.76      |

Random Forest wins. `Director_encoded` is by far the strongest predictor
(who directed the movie matters more than genre or year), followed by the
lead actors and vote count.

## Using the saved model to predict a new movie

```python
import joblib
import pandas as pd

saved = joblib.load("model/movie_rating_model.pkl")
model, feature_cols = saved["model"], saved["feature_cols"]

# Build one row with the same feature columns the model was trained on,
# then call model.predict(new_row_df[feature_cols])
```

## Ideas to extend this

- Try `TfidfVectorizer` on `Genre` instead of top-15 multi-hot encoding.
- Add `XGBoost` or `LightGBM` for potentially better accuracy.
- Hyperparameter-tune the Random Forest with `GridSearchCV`.
- Handle actors/directors with very few movies using a hierarchical
  Bayesian smoothing approach instead of the simple `m=5` smoothing used
  here.
