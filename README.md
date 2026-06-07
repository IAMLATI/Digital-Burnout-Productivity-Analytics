# Digital Burnout & Productivity Analytics

Predicting workplace **productivity** from digital-behaviour and wellbeing signals — an end-to-end machine-learning project covering data cleaning, EDA, feature engineering, model selection, explainability, and live deployment.

🌐 **Live site:** [olabam.com](https://olabam.com) · 🔮 **Try the predictor:** [Digital Burnout Productivity Predictor](https://olabam.com/Digital_Burnout_Regression.html)

-----

## Overview

Digital habits — screen time, app-switching, doomscrolling, deep-work focus — interact with wellbeing factors like sleep, stress and motivation to shape how productive someone is. This project trains a regression model on a large-scale (5,000,000-row) behavioural dataset to **predict a continuous productivity score (0–100)** and serves it through an interactive web tool.

The final model is an **XGBoost regressor** achieving **R² = 0.866** and **RMSE ≈ 7.86** on a held-out test set.

## Key results

|Model            |Test R²  |Test RMSE|Overfit gap|Train time|
|-----------------|---------|---------|-----------|----------|
|**XGBoost** ⭐    |**0.866**|**7.86** |0.023      |39.5s     |
|Gradient Boosting|0.857    |8.12     |0.008      |304.7s    |
|Linear Regression|0.841    |8.58     |0.001      |1.8s      |
|Random Forest    |0.815    |9.25     |0.078      |187.7s    |
|Decision Tree    |0.711    |11.56    |0.053      |8.3s      |
|AdaBoost         |0.702    |11.74    |0.002      |58.7s     |

**Top drivers of productivity** (SHAP): `deep_work_hours`, `task_completion_rate`, `concentration_score`, `burnout_risk`, `motivation_level`, `focus_sessions`.

## Tech stack

- **Data & ML:** Python, pandas, NumPy, scikit-learn, XGBoost, SHAP
- **Visualisation:** matplotlib, seaborn
- **Serving:** Flask, flask-cors, joblib
- **Frontend:** HTML / CSS / JavaScript (interactive speedometer gauge, light/dark theme)
- **Deployment:** Render (model API) + GitHub Pages (static site)

## Project structure

```
Digital-Burnout-Productivity-Analytics/
├── app/
│   ├── models/
│   │   └── XGBoost_model.pkl                 # trained model artifact
│   ├── static/images/                        # logos, favicons, profile image
│   ├── templates/
│   │   ├── index.html                        # portfolio homepage
│   │   └── Digital_Burnout_Regression.html   # prediction UI
│   ├── __init__.py
│   └── routes.py
├── notebooks/
│   └── Digital_Burnout_&_Productivity_Analytics.ipynb   # full analysis
├── data/                                      # dataset (not tracked)
├── app.py                                     # Flask API entrypoint
├── requirements.txt
├── README.md
└── REPORT.md                                  # detailed project report
```

## The model API

The Flask service exposes a single prediction endpoint.

**`POST /predict`**

```json
{
  "burnout_risk": 20,
  "social_media_hours": 3,
  "app_switch_frequency": 40,
  "distraction_burden_work": 6,
  "deep_work_hours": 4,
  "focus_sessions": 5,
  "concentration_score": 7,
  "task_completion_rate": 80,
  "motivation_level": 8
}
```

**Response**

```json
{ "prediction": 73.52 }
```

## Run locally

```bash
# 1. clone and enter the repo
git clone https://github.com/IAMLATI/Digital-Burnout-Productivity-Analytics.git
cd Digital-Burnout-Productivity-Analytics

# 2. create a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. start the API
python app.py                   # serves on http://0.0.0.0:5000
```

Then send a request to `http://localhost:5000/predict`, or open the predictor page which calls the API.

## Methodology at a glance

1. **Cleaning** — mean-imputed <3% missing values; confirmed no duplicates; downcast dtypes to cut memory from ~1.3 GB to ~1.17 GB.
1. **EDA** — found deep work, focus and concentration as the strongest positive signals; burnout, stress and emotional exhaustion as the main negative ones; screen-usage metrics only weakly (and non-linearly) related.
1. **Feature engineering** — created 32 composite features (e.g. `deep_work_efficiency`, `psychological_readiness`, `digital_wellness`, `cognitive_overload`).
1. **Selection** — dropped weak (|r| < 0.03) and highly collinear (|r| > 0.95) features, leaving **39** predictors.
1. **Modelling** — 70/15/15 split, `StandardScaler`, six regressors benchmarked on a 1M-row stratified sample; **XGBoost** selected.
1. **Explainability** — SHAP confirmed deep work and task completion as dominant drivers.
1. **Deployment** — model serialised with joblib, served via Flask, fronted by an interactive web UI.

See **<REPORT.md>** for the full write-up.

## Author

**Olamide Bamigbola** — Data Scientist
[olabam.com](https://olabam.com) · [LinkedIn](https://www.linkedin.com/in/olamidebamigbola) · [GitHub](https://github.com/IAMLATI)

## License

Released for portfolio and educational use.