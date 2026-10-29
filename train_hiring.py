import json
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import RandomizedSearchCV, train_test_split

warnings.filterwarnings("ignore")

# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "hiring_data.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_PATH = MODEL_DIR / "hiring_catboost_nlp.cbm"
METADATA_PATH = MODEL_DIR / "hiring_metadata.json"
VECTORIZER_PATH = MODEL_DIR / "hiring_tfidf_vectorizer.joblib"
METRICS_PATH = OUTPUT_DIR / "hiring_metrics.json"
PREDICTIONS_PATH = OUTPUT_DIR / "hiring_test_predictions.csv"
TUNING_RESULTS_PATH = OUTPUT_DIR / "hiring_hyperparameter_results.csv"

TARGET_COL = "Hiring_Outcome"
TEXT_COLS = ["Resume_Text", "Job_Description"]
DROP_COLS = ["Application_ID", "Application_Date", "Application_Status"]
RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20



# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================
def clean_text(text):
    """Normalise text before TF-IDF processing."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def calculate_similarity(resumes, descriptions, vectorizer):
    """Calculate pairwise resume/job-description TF-IDF cosine similarity."""
    cleaned_resumes = [clean_text(x) for x in resumes]
    cleaned_descriptions = [clean_text(x) for x in descriptions]
    resume_vectors = vectorizer.transform(cleaned_resumes)
    job_vectors = vectorizer.transform(cleaned_descriptions)
    return cosine_similarity(resume_vectors, job_vectors).diagonal().astype(float)


def convert_target(series):
    """Convert hiring outcome labels into binary values: Hired=1, Not Hired=0."""
    if pd.api.types.is_numeric_dtype(series):
        values = set(series.dropna().unique())
        if values.issubset({0, 1}):
            return series.astype(int)

    mapping = {
        "hired": 1,
        "yes": 1,
        "selected": 1,
        "1": 1,
        "not hired": 0,
        "not_hired": 0,
        "no": 0,
        "rejected": 0,
        "not selected": 0,
        "not_selected": 0,
        "0": 0,
    }
    converted = series.astype(str).str.strip().str.lower().map(mapping)
    if converted.isna().any():
        unknown = series[converted.isna()].unique().tolist()
        raise ValueError(f"Unknown values in {TARGET_COL}: {unknown}")
    return converted.astype(int)


def calculate_metrics(y_true, probabilities, threshold):
    """Calculate classification and ranking metrics."""
    predictions = (probabilities >= threshold).astype(int)
    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision_hired": float(precision_score(y_true, predictions, zero_division=0)),
        "recall_hired": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_hired": float(f1_score(y_true, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
    }
    metrics["roc_auc"] = (
        float(roc_auc_score(y_true, probabilities))
        if len(np.unique(y_true)) == 2
        else None
    )
    return metrics

# ============================================================
# 3. MAIN TRAINING PIPELINE
# ============================================================
def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place your CSV at data/hiring_data.csv."
        )

    df = pd.read_csv(DATA_PATH)
    print("\n========== DATASET INFORMATION ==========")
    print(f"Dataset shape: {df.shape}")
    print(f"Dataset path: {DATA_PATH}")

    required_columns = [TARGET_COL] + TEXT_COLS
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. The CSV must contain "
            "Hiring_Outcome, Resume_Text, and Job_Description."
        )

    df = df.dropna(subset=[TARGET_COL]).copy()
    y = convert_target(df[TARGET_COL]).reset_index(drop=True)
    df = df.reset_index(drop=True)

    if y.nunique() != 2:
        raise ValueError("Hiring_Outcome must contain both Hired and Not Hired classes.")
    if y.value_counts().min() < 3:
        raise ValueError("Each target class needs at least three rows for stratified splitting.")

    for col in TEXT_COLS:
        df[col] = df[col].fillna("").astype(str)

    print("\nTarget distribution:")
    print(y.value_counts().rename(index={0: "Not Hired", 1: "Hired"}))
    print("\nTarget percentages:")
    print((y.value_counts(normalize=True) * 100).round(2))

    # ========================================================
    # 4. TRAIN / VALIDATION / TEST SPLITS
    # ========================================================
    all_indices = np.arange(len(df))
    train_val_indices, test_indices = train_test_split(
        all_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    y_train_val = y.iloc[train_val_indices]
    relative_validation_size = VALIDATION_SIZE / (1.0 - TEST_SIZE)
    train_indices, validation_indices = train_test_split(
        train_val_indices,
        test_size=relative_validation_size,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    train_df = df.iloc[train_indices].copy().reset_index(drop=True)
    validation_df = df.iloc[validation_indices].copy().reset_index(drop=True)
    test_df = df.iloc[test_indices].copy().reset_index(drop=True)
    y_train = y.iloc[train_indices].reset_index(drop=True)
    y_validation = y.iloc[validation_indices].reset_index(drop=True)
    y_test = y.iloc[test_indices].reset_index(drop=True)

    print("\n========== DATA SPLITS ==========")
    print(f"Training rows:   {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")
    print(f"Test rows:       {len(test_df)}")

    # ========================================================
    # 5. NLP FEATURE: RESUME / JOB DESCRIPTION SIMILARITY
    # ========================================================
    training_corpus = (
        train_df["Resume_Text"].map(clean_text).tolist()
        + train_df["Job_Description"].map(clean_text).tolist()
    )
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
    )
    try:
        vectorizer.fit(training_corpus)
    except ValueError as exc:
        raise ValueError(
            "TF-IDF could not build a vocabulary. Check that CV and job description "
            "text contains meaningful words."
        ) from exc

    train_similarity = calculate_similarity(
        train_df["Resume_Text"], train_df["Job_Description"], vectorizer
    )
    validation_similarity = calculate_similarity(
        validation_df["Resume_Text"], validation_df["Job_Description"], vectorizer
    )
    test_similarity = calculate_similarity(
        test_df["Resume_Text"], test_df["Job_Description"], vectorizer
    )
    print("\nCreated NLP feature: Resume_Job_Similarity")


    # ========================================================
    excluded_columns = set(DROP_COLS + TEXT_COLS + [TARGET_COL])
    feature_columns = [col for col in df.columns if col not in excluded_columns]
    if not feature_columns:
        raise ValueError("No structured feature columns remain after excluding IDs/text/target.")

    X_train = train_df[feature_columns].copy().reset_index(drop=True)
    X_validation = validation_df[feature_columns].copy().reset_index(drop=True)
    X_test = test_df[feature_columns].copy().reset_index(drop=True)

    X_train["Resume_Job_Similarity"] = train_similarity
    X_validation["Resume_Job_Similarity"] = validation_similarity
    X_test["Resume_Job_Similarity"] = test_similarity

    categorical_columns = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    for col in categorical_columns:
        X_train[col] = X_train[col].fillna("Unknown").astype(str)
        X_validation[col] = X_validation[col].fillna("Unknown").astype(str)
        X_test[col] = X_test[col].fillna("Unknown").astype(str)

    numeric_columns = [col for col in X_train.columns if col not in categorical_columns]
    numeric_medians = {}
    for col in numeric_columns:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_validation[col] = pd.to_numeric(X_validation[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
        median = X_train[col].median()
        if pd.isna(median):
            median = 0.0
        numeric_medians[col] = float(median)
        X_train[col] = X_train[col].fillna(median)
        X_validation[col] = X_validation[col].fillna(median)
        X_test[col] = X_test[col].fillna(median)

    X_validation = X_validation.reindex(columns=X_train.columns)
    X_test = X_test.reindex(columns=X_train.columns)

    print("\n========== FEATURES ==========")
    print(f"Total features: {len(X_train.columns)}")
    print(f"Categorical features: {categorical_columns}")
    print(f"Numeric features: {numeric_columns}")

    # ========================================================
    # 7. HYPERPARAMETER TUNING
    # ========================================================
    base_model = CatBoostClassifier(
        loss_function="Logloss",
        auto_class_weights="Balanced",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        thread_count=1,
    )
    param_distributions = {
        "iterations": [400, 600, 800],
        "depth": [4, 5, 6, 7, 8],
        "learning_rate": [0.03, 0.05, 0.08, 0.10],
        "l2_leaf_reg": [3, 5, 7, 9],
        "random_strength": [0.5, 1, 2],
    }
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=12,
        scoring="average_precision",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=2,
        refit=True,
        error_score="raise",
    )

    print("\n========== HYPERPARAMETER TUNING ==========")
    print("Testing 12 parameter combinations with 3-fold cross-validation.")
    search.fit(X_train, y_train, cat_features=categorical_columns)
    print("Best parameters:", search.best_params_)
    print("Best CV average precision:", round(search.best_score_, 4))

    # Train the selected configuration with validation-based early stopping.
    model = CatBoostClassifier(
        **search.best_params_,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=RANDOM_STATE,
        verbose=100,
        allow_writing_files=False,
    )
    print("\n========== TRAINING FINAL MODEL ==========")
    model.fit(
        X_train,
        y_train,
        cat_features=categorical_columns,
        eval_set=(X_validation, y_validation),
        early_stopping_rounds=100,
        verbose=100,
    )
    print("Best iteration:", model.get_best_iteration())

    pd.DataFrame(search.cv_results_).sort_values("rank_test_score").to_csv(
        TUNING_RESULTS_PATH, index=False
    )

     # ========================================================
    # 8. SELECT THRESHOLD USING VALIDATION DATA
    # ========================================================
    validation_probabilities = model.predict_proba(X_validation)[:, 1]
    threshold_candidates = np.arange(0.10, 0.91, 0.01)
    threshold_results = []
    for threshold in threshold_candidates:
        predictions = (validation_probabilities >= threshold).astype(int)
        threshold_results.append({
            "threshold": float(threshold),
            "f1": float(f1_score(y_validation, predictions, zero_division=0)),
            "precision": float(precision_score(y_validation, predictions, zero_division=0)),
            "recall": float(recall_score(y_validation, predictions, zero_division=0)),
        })
    best_result = max(threshold_results, key=lambda result: result["f1"])
    selected_threshold = best_result["threshold"]
    print("\nSelected threshold:", round(selected_threshold, 2))
    print("Validation F1:", round(best_result["f1"], 4))

    # ========================================================
    # 9. FINAL TEST EVALUATION
    # ========================================================
    test_probabilities = model.predict_proba(X_test)[:, 1]
    test_metrics = calculate_metrics(y_test, test_probabilities, selected_threshold)
    test_predictions = (test_probabilities >= selected_threshold).astype(int)

    print("\n========== TEST RESULTS ==========")
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision_hired']:.4f}")
    print(f"Recall:    {test_metrics['recall_hired']:.4f}")
    print(f"F1 score:  {test_metrics['f1_hired']:.4f}")
    print(f"PR-AUC:    {test_metrics['pr_auc']:.4f}")
    if test_metrics["roc_auc"] is not None:
        print(f"ROC-AUC:   {test_metrics['roc_auc']:.4f}")

    print("\nConfusion matrix:")
    print(pd.DataFrame(
        test_metrics["confusion_matrix"],
        index=["Actual Not Hired", "Actual Hired"],
        columns=["Predicted Not Hired", "Predicted Hired"],
    ))
    print("\nClassification report:")
    print(classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["Not Hired", "Hired"],
        zero_division=0,
    ))

    # ========================================================
    # 10. SAVE MODEL, VECTORIZER, METADATA, AND METRICS
    # ========================================================
    model.save_model(str(MODEL_PATH))
    joblib.dump(vectorizer, VECTORIZER_PATH)
    metadata = {
        "target_column": TARGET_COL,
        "feature_columns": X_train.columns.tolist(),
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "numeric_medians": numeric_medians,
        "text_columns": TEXT_COLS,
        "nlp_feature": "Resume_Job_Similarity",
        "threshold": float(selected_threshold),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "validation_size": VALIDATION_SIZE,
        "model_type": "CatBoostClassifier with TF-IDF cosine similarity",
        "best_hyperparameters": search.best_params_,
        "synthetic_data_warning": (
            "If trained on synthetic data, metrics do not establish real-world hiring performance."
        ),
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)
    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump({
            "validation": {
                "selected_threshold": float(selected_threshold),
                "best_validation_f1": float(best_result["f1"]),
                "precision": float(best_result["precision"]),
                "recall": float(best_result["recall"]),
            },
            "test": test_metrics,
        }, file, indent=4)

 # ========================================================
    # 11. SAVE TEST PREDICTIONS
    # ========================================================
    output_df = test_df.copy()
    output_df["Actual_Hiring_Outcome"] = y_test.map({0: "Not Hired", 1: "Hired"}).values
    output_df["Predicted_Hiring_Outcome"] = pd.Series(test_predictions).map(
        {0: "Not Hired", 1: "Hired"}
    ).values
    output_df["Hiring_Probability"] = test_probabilities
    output_df["Resume_Job_Similarity"] = test_similarity
    output_df.to_csv(PREDICTIONS_PATH, index=False)

    # ========================================================
    # 12. SUMMARY
    # ========================================================
    print("\n========== FILES SAVED ==========")
    print(f"Model:       {MODEL_PATH}")
    print(f"Vectorizer:  {VECTORIZER_PATH}")
    print(f"Metadata:    {METADATA_PATH}")
    print(f"Metrics:     {METRICS_PATH}")
    print(f"Predictions: {PREDICTIONS_PATH}")
    print(f"Tuning CSV:  {TUNING_RESULTS_PATH}")
    print("\nTraining completed.")
    print(
        "Reminder: if your data is synthetic, test metrics do not prove real-world hiring accuracy."
    )


if __name__ == "__main__":
    main()
