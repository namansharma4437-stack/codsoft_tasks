# Sales Prediction ML

Predicts product sales from advertising spend on TV, Radio, and Newspaper using machine learning in Python.

## Overview

Businesses want to know: *if we spend $X on TV, $Y on Radio, and $Z on Newspaper ads, how many units will we sell?*

This project trains and compares two regression models on 200 historical ad-spend/sales records to answer that question, then exposes a simple `predict_sales()` function for new predictions.

## Dataset

`advertising.csv` — 200 rows, 4 columns:

| Column | Description |
|---|---|
| TV | TV advertising spend ($ thousands) |
| Radio | Radio advertising spend ($ thousands) |
| Newspaper | Newspaper advertising spend ($ thousands) |
| Sales | Units sold (thousands) |

## Models & Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 1.27 | 1.71 | 0.906 |
| **Random Forest** | **0.90** | **1.18** | **0.955** |

Random Forest performs best — advertising's effect on sales isn't perfectly linear (diminishing returns at higher spend), which tree-based models capture better.

**Feature importance:** TV spend drives the majority of sales impact, Radio is moderate, Newspaper is minimal.

## Setup

```bash
# clone the repo
git clone https://github.com/<your-username>/sales-prediction-ml.git
cd sales-prediction-ml/project

# create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

## Usage

Run the full pipeline (EDA, training, evaluation, chart generation):

```bash
python sales_prediction.py
```

This prints dataset stats and model metrics to the terminal, and saves the following to `outputs/`:

- `correlation_heatmap.png`
- `spend_vs_sales.png`
- `actual_vs_predicted.png`
- `feature_importance.png`
- `model_comparison.csv`
- `sales_rf_model.pkl` / `sales_linear_model.pkl` (trained models)

### Making a prediction

```python
import joblib

model = joblib.load("outputs/sales_rf_model.pkl")
prediction = model.predict([[150, 25, 20]])  # [TV, Radio, Newspaper] spend in $1000s
print(prediction)  # predicted sales in thousands of units
```

## Charts

![Correlation Heatmap](outputs/correlation_heatmap.png)
![Ad Spend vs Sales](outputs/spend_vs_sales.png)
![Actual vs Predicted](outputs/actual_vs_predicted.png)
![Feature Importance](outputs/feature_importance.png)

## Tech Stack

- Python 3.12
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- joblib

## License

MIT
