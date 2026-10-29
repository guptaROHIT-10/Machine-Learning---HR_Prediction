
# train_hiring_logistic.py

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

DATA_PATH = Path("data/hiring_data.csv")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Hiring_Outcome"

TEXT_COLUMNS = [
    "Resume_Text",
    "Job_Description"
]

# Exclude IDs, dates, and post-decision information.
DROP_COLUMNS = [
    "Application_ID",
    "Application_Date",
    "Application_Status"
]

RANDOM_STATE = 42
SVD_COMPONENTS = 30

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH.resolve()}"
    )

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

required_columns = [TARGET] + TEXT_COLUMNS

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )

df = df.dropna(subset=[TARGET]).copy()

# Remove exact duplicate rows.
df = df.drop_duplicates().reset_index(drop=True)

# --------------------------------------------------
# 3. ENCODE TARGET
# --------------------------------------------------

df[TARGET] = (
    df[TARGET]
    .astype(str)
    .str.strip()
    .str.lower()
)

target_mapping = {
    "hired": 1,
    "not hired": 0
}

unknown_labels = (
    set(df[TARGET].unique())
    - set(target_mapping.keys())
)

if unknown_labels:
    raise ValueError(
        f"Unexpected target labels: {unknown_labels}"
    )

y = df[TARGET].map(target_mapping).astype(int)

print("\nTarget distribution:")
print(y.map({0: "Not Hired", 1: "Hired"}).value_counts())

# --------------------------------------------------
# 4. PREPARE FEATURES
# --------------------------------------------------

columns_to_drop = [
    col for col in [TARGET] + DROP_COLUMNS
    if col in df.columns
]

X = df.drop(columns=columns_to_drop).copy()

for col in TEXT_COLUMNS:
    X[col] = X[col].fillna("").astype(str)

structured_columns = [
    col for col in X.columns
    if col not in TEXT_COLUMNS
]

numeric_columns = (
    X[structured_columns]
    .select_dtypes(include=["number"])
    .columns.tolist()
)

categorical_columns = [
    col for col in structured_columns
    if col not in numeric_columns
]

print("\nText columns:", TEXT_COLUMNS)
print("Numeric columns:", numeric_columns)
print("Categorical columns:", categorical_columns)

# --------------------------------------------------
# 5. STRATIFIED 60:20:20 SPLIT
# --------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=RANDOM_STATE,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp
)

print("\nSplit sizes:")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))

# --------------------------------------------------
# 6. TF-IDF + SVD TEXT FEATURES
# --------------------------------------------------

def combine_text(data):
    return (
        data[TEXT_COLUMNS]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .tolist()
    )

train_text = combine_text(X_train)
val_text = combine_text(X_val)
test_text = combine_text(X_test)

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    stop_words="english"
)

train_tfidf = tfidf.fit_transform(train_text)
val_tfidf = tfidf.transform(val_text)
test_tfidf = tfidf.transform(test_text)

if train_tfidf.shape[1] < 2:
    raise ValueError(
        "Insufficient text vocabulary for SVD."
    )

n_components = min(
    SVD_COMPONENTS,
    train_tfidf.shape[1] - 1,
    train_tfidf.shape[0] - 1
)

svd = TruncatedSVD(
    n_components=n_components,
    random_state=RANDOM_STATE
)

train_svd = svd.fit_transform(train_tfidf)
val_svd = svd.transform(val_tfidf)
test_svd = svd.transform(test_tfidf)

print("\nTF-IDF vocabulary:", len(tfidf.vocabulary_))
print("SVD components:", n_components)

# --------------------------------------------------
# 7. STRUCTURED FEATURE PREPROCESSING
# --------------------------------------------------

transformers = []

if numeric_columns:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    transformers.append(
        ("numeric", numeric_pipeline, numeric_columns)
    )

if categorical_columns:
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    transformers.append(
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )
    )

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop"
)

train_struct = preprocessor.fit_transform(
    X_train[structured_columns]
)

val_struct = preprocessor.transform(
    X_val[structured_columns]
)

test_struct = preprocessor.transform(
    X_test[structured_columns]
)

def combine_features(structured, text):
    if hasattr(structured, "toarray"):
        structured = structured.toarray()

    return np.hstack([structured, text])

X_train_final = combine_features(train_struct, train_svd)
X_val_final = combine_features(val_struct, val_svd)
X_test_final = combine_features(test_struct, test_svd)

print("Final training feature shape:", X_train_final.shape)

# --------------------------------------------------
# 8. LOGISTIC REGRESSION + GRID SEARCH
# --------------------------------------------------

pipeline = LogisticRegression(
    max_iter=3000,
    random_state=RANDOM_STATE,
    solver="liblinear"
)

param_grid = {
    "C": [0.01, 0.1, 1, 5, 10],
    "class_weight": [None, "balanced"]
}

search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=3,
    n_jobs=-1,
    verbose=1,
    refit=True
)

print("\nStarting Logistic Regression tuning...")

search.fit(X_train_final, y_train)

best_model = search.best_estimator_

print("\nBest parameters:", search.best_params_)
print("Best cross-validation F1:", round(search.best_score_, 4))

# --------------------------------------------------
# 9. VALIDATION RESULTS
# --------------------------------------------------

val_probabilities = best_model.predict_proba(
    X_val_final
)[:, 1]

val_predictions = (val_probabilities >= 0.5).astype(int)

print(
    "\nValidation F1:",
    round(f1_score(y_val, val_predictions, zero_division=0), 4)
)

# --------------------------------------------------
# 10. FINAL TEST RESULTS
# --------------------------------------------------

test_probabilities = best_model.predict_proba(
    X_test_final
)[:, 1]

test_predictions = (test_probabilities >= 0.5).astype(int)

metrics = {
    "model": "Logistic Regression + TF-IDF + SVD",
    "accuracy": float(
        accuracy_score(y_test, test_predictions)
    ),
    "precision": float(
        precision_score(y_test, test_predictions, zero_division=0)
    ),
    "recall": float(
        recall_score(y_test, test_predictions, zero_division=0)
    ),
    "f1_score": float(
        f1_score(y_test, test_predictions, zero_division=0)
    ),
    "roc_auc": float(
        roc_auc_score(y_test, test_probabilities)
    ),
    "pr_auc": float(
        average_precision_score(y_test, test_probabilities)
    ),
    "best_cv_f1": float(search.best_score_),
    "best_parameters": search.best_params_,
    "train_rows": int(len(X_train)),
    "validation_rows": int(len(X_val)),
    "test_rows": int(len(X_test))
}

print("\n========== TEST RESULTS ==========")

for key, value in metrics.items():
    print(
        f"{key}: {value:.4f}"
        if isinstance(value, float)
        else f"{key}: {value}"
    )

print("\nClassification report:")
print(
    classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["Not Hired", "Hired"],
        zero_division=0
    )
)

print("Confusion matrix:")
print(
    confusion_matrix(
        y_test,
        test_predictions,
        labels=[0, 1]
    )
)

# --------------------------------------------------
# 11. SAVE MODEL AND ARTIFACTS
# --------------------------------------------------

joblib.dump(
    best_model,
    MODEL_DIR / "hiring_logistic_nlp.joblib"
)

joblib.dump(
    tfidf,
    MODEL_DIR / "hiring_logistic_tfidf.joblib"
)

joblib.dump(
    svd,
    MODEL_DIR / "hiring_logistic_svd.joblib"
)

joblib.dump(
    preprocessor,
    MODEL_DIR / "hiring_logistic_preprocessor.joblib"
)

metadata = {
    "model_name": "Logistic Regression + TF-IDF + SVD",
    "target_column": TARGET,
    "target_mapping": target_mapping,
    "positive_class": 1,
    "negative_class": 0,
    "text_columns": TEXT_COLUMNS,
    "structured_columns": structured_columns,
    "numeric_columns": numeric_columns,
    "categorical_columns": categorical_columns,
    "dropped_columns": columns_to_drop,
    "svd_components": int(n_components),
    "random_state": RANDOM_STATE,
    "threshold": 0.5,
    "best_parameters": search.best_params_
}

with open(
    MODEL_DIR / "hiring_logistic_metadata.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(metadata, f, indent=4)

with open(
    OUTPUT_DIR / "hiring_logistic_metrics.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(metrics, f, indent=4)

predictions_df = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Predicted": test_predictions,
    "Probability_Hired": test_probabilities
})

predictions_df.to_csv(
    OUTPUT_DIR / "hiring_logistic_predictions.csv",
    index=False
)

pd.DataFrame(search.cv_results_).to_csv(
    OUTPUT_DIR / "hiring_logistic_tuning_results.csv",
    index=False
)

print("\nAll artifacts saved successfully.")
print("Model:", MODEL_DIR / "hiring_logistic_nlp.joblib")
print("Metrics:", OUTPUT_DIR / "hiring_logistic_metrics.json")
print("Predictions:", OUTPUT_DIR / "hiring_logistic_predictions.csv")
