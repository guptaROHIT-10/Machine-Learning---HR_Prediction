# HR Hiring Prediction System

An AI-powered HR analytics project that predicts hiring outcomes using candidate information, resume text, job descriptions, and machine learning classification models. The project also compares multiple models to evaluate their performance in identifying hired candidates.

## Features

- Predicts hiring outcomes: Hired or Not Hired.
- Processes resume text and job descriptions using NLP.
- Uses TF-IDF and dimensionality reduction for text features.
- Compares CatBoost, Logistic Regression, and Random Forest.
- Evaluates models using Accuracy, Precision, Recall, F1-score, ROC-AUC, and PR-AUC.
- Visualises model performance through a Streamlit dashboard.
- Generates classification reports and confusion matrices.

## Tech Stack

- **Language:** Python
- **Machine Learning:** scikit-learn, CatBoost
- **NLP:** TF-IDF, Truncated SVD
- **Data Processing:** Pandas, NumPy
- **Visualisation:** Matplotlib
- **Web Application:** Streamlit
- **Model Persistence:** Joblib

## Project Structure

```text
HR-Prediction/
├── data/
│   └── hiring_data.csv
├── models/
├── outputs/
├── promotion_app.py
├── train_hiring_logistic.py
├── train_hiring_random_forest.py
├── hiring_model_comparison_app.py
└── README.md
```

*Adjust the file tree to match the final contents of your repository.*

## Installation

Clone the repository and navigate to the project directory:

```bash
git clone <your-repository-url>
cd HR-Prediction
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows, activate it with:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
python -m pip install pandas numpy scikit-learn catboost streamlit matplotlib joblib
```

## Run the Application

Launch the hiring model comparison dashboard:

```bash
python -m streamlit run hiring_model_comparison_app.py
```

To train the Logistic Regression model:

```bash
python train_hiring_logistic.py
```

To train the Random Forest model:

```bash
python train_hiring_random_forest.py
```

## Model Evaluation

The following results were obtained from the reported test evaluations.

| Metric | CatBoost | Logistic Regression | Random Forest |
|---|---:|---:|---:|
| Accuracy | 74.83% | 64.67% | 75.00% |
| Precision | 34.96% | 30.40% | 33.33% |
| Recall | 37.72% | 66.67% | 31.58% |
| F1-score | 36.29% | 41.76% | 32.43% |
| ROC-AUC | 0.6882 | 0.7198 | 0.6718 |
| PR-AUC | 0.3443 | 0.3888 | 0.3167 |

**Best model by F1-score:** Logistic Regression.

Logistic Regression identifies a greater proportion of actual hired candidates, although its precision is relatively low. The results demonstrate why model evaluation should consider multiple metrics rather than accuracy alone.

## Future Improvements

- Improve the quality and representativeness of the hiring dataset.
- Tune classification thresholds according to hiring requirements.
- Add resume upload and candidate-level prediction functionality.
- Improve explainability and investigate potential bias in hiring predictions.
- Evaluate models on independent, representative real-world data.

## Disclaimer

This project is intended for educational and experimental purposes. Predictions should not be used as the sole basis for real hiring decisions. Any practical deployment requires appropriate validation, fairness assessments, privacy safeguards, and human oversight.
