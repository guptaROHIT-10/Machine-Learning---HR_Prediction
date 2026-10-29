
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Hiring Model Comparison",
    page_icon="📊",
    layout="wide"
)

st.title("📊 HR Hiring Prediction — Model Comparison")
st.write(
    "Compare CatBoost, Logistic Regression, and Random Forest "
    "on the same 600-record test set."
)

# =====================================================
# MODEL METRICS
# =====================================================

metrics = pd.DataFrame({
    "Model": [
        "CatBoost",
        "Logistic Regression",
        "Random Forest"
    ],
    "Accuracy": [0.7483, 0.6467, 0.7500],
    "Precision": [0.3496, 0.3040, 0.3333],
    "Recall": [0.3772, 0.6667, 0.3158],
    "F1 Score": [0.3629, 0.4176, 0.3243],
    "ROC-AUC": [0.6882, 0.7198, 0.6718],
    "PR-AUC": [0.3443, 0.3888, 0.3167],
})

metrics["Best CV F1"] = [
    np.nan,
    0.4079,
    0.2760
]

# =====================================================
# CONFUSION MATRICES
# Rows = actual classes, columns = predicted classes
# Class order: Not Hired, Hired
# =====================================================

confusion_matrices = {
    "CatBoost": np.array([
        [406, 80],
        [71, 43]
    ]),
    "Logistic Regression": np.array([
        [312, 174],
        [38, 76]
    ]),
    "Random Forest": np.array([
        [414, 72],
        [78, 36]
    ])
}

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Dashboard Controls")

selected_model = st.sidebar.selectbox(
    "Select model for detailed analysis",
    metrics["Model"].tolist()
)

st.sidebar.caption(
    "All models use a test set of 600 records: "
    "486 Not Hired and 114 Hired."
)

# =====================================================
# TOP KPI CARDS
# =====================================================

st.subheader("1. Overall Model Performance")

best_f1_model = metrics.loc[
    metrics["F1 Score"].idxmax(), "Model"
]

best_recall_model = metrics.loc[
    metrics["Recall"].idxmax(), "Model"
]

best_auc_model = metrics.loc[
    metrics["ROC-AUC"].idxmax(), "Model"
]

best_accuracy_model = metrics.loc[
    metrics["Accuracy"].idxmax(), "Model"
]

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Best Accuracy",
    f"{metrics['Accuracy'].max():.2%}",
    best_accuracy_model
)

c2.metric(
    "Best Hired F1",
    f"{metrics['F1 Score'].max():.2%}",
    best_f1_model
)

c3.metric(
    "Best Hired Recall",
    f"{metrics['Recall'].max():.2%}",
    best_recall_model
)

c4.metric(
    "Best ROC-AUC",
    f"{metrics['ROC-AUC'].max():.4f}",
    best_auc_model
)

st.info(
    "Logistic Regression performs best on F1-score, recall, "
    "ROC-AUC, and PR-AUC. CatBoost and Random Forest have "
    "higher accuracy, but accuracy alone does not show how "
    "well a model identifies hired candidates."
)

# =====================================================
# COMPARISON TABLE
# =====================================================

st.subheader("2. Complete Metrics Comparison")

display_metrics = metrics.copy()

percentage_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "PR-AUC"
]

for col in percentage_columns:
    display_metrics[col] = display_metrics[col].map(
        lambda x: f"{x:.2%}"
    )

display_metrics["ROC-AUC"] = metrics["ROC-AUC"].map(
    lambda x: f"{x:.4f}"
)

display_metrics["Best CV F1"] = metrics["Best CV F1"].map(
    lambda x: f"{x:.4f}" if pd.notna(x) else "Not provided"
)

st.dataframe(
    display_metrics,
    use_container_width=True,
    hide_index=True
)

# st.download_button(
#     label="⬇️ Download comparison as CSV",
#     data=metrics.to_csv(index=False).encode("utf-8"),
#     file_name="hiring_model_comparison.csv",
#     mime="text/csv"
# )

# =====================================================
# ACCURACY COMPARISON CHART
# =====================================================

st.subheader("3. Accuracy Comparison")

fig, ax = plt.subplots(figsize=(9, 4))

bars = ax.bar(
    metrics["Model"],
    metrics["Accuracy"] * 100
)

ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 100)
ax.set_title("Test Accuracy by Model")
ax.grid(axis="y", alpha=0.25)

for bar, value in zip(bars, metrics["Accuracy"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{value:.2%}",
        ha="center",
        va="bottom"
    )

plt.xticks(rotation=10)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# =====================================================
# PRECISION, RECALL, F1 COMPARISON
# =====================================================

st.subheader("4. Hired-Class Performance")

selected_metrics = ["Precision", "Recall", "F1 Score"]

fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(metrics["Model"]))
width = 0.24

for i, metric in enumerate(selected_metrics):
    values = metrics[metric].to_numpy()

    bars = ax.bar(
        x + (i - 1) * width,
        values,
        width,
        label=metric
    )

    ax.bar_label(
        bars,
        fmt="%.2f",
        padding=2,
        fontsize=8
    )

ax.set_xticks(x)
ax.set_xticklabels(metrics["Model"])
ax.set_ylabel("Score")
ax.set_ylim(0, 0.85)
ax.set_title("Precision, Recall and F1 for Hired Candidates")
ax.legend()
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.caption(
    "Precision measures how many predicted hires were actually hired. "
    "Recall measures how many actual hires were detected. "
    "F1 balances precision and recall."
)

# =====================================================
# ROC-AUC AND PR-AUC COMPARISON
# =====================================================

st.subheader("5. Ranking and Positive-Class Performance")

auc_metrics = ["ROC-AUC", "PR-AUC"]

fig, ax = plt.subplots(figsize=(9, 4))

x = np.arange(len(metrics["Model"]))
width = 0.32

for i, metric in enumerate(auc_metrics):
    values = metrics[metric].to_numpy()

    bars = ax.bar(
        x + (i - 0.5) * width,
        values,
        width,
        label=metric
    )

    ax.bar_label(
        bars,
        fmt="%.3f",
        padding=2,
        fontsize=9
    )

ax.set_xticks(x)
ax.set_xticklabels(metrics["Model"])
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.set_title("ROC-AUC and PR-AUC Comparison")
ax.legend()
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# =====================================================
# CONFUSION MATRIX
# =====================================================

st.subheader("6. Confusion Matrix")

st.write(f"Selected model: **{selected_model}**")

cm = confusion_matrices[selected_model]

fig, ax = plt.subplots(figsize=(6, 4))

image = ax.imshow(cm, interpolation="nearest")
fig.colorbar(image, ax=ax, label="Number of candidates")

ax.set(
    xticks=np.arange(2),
    yticks=np.arange(2),
    xticklabels=["Not Hired", "Hired"],
    yticklabels=["Not Hired", "Hired"],
    xlabel="Predicted Class",
    ylabel="Actual Class",
    title=f"{selected_model} — Confusion Matrix"
)

threshold = cm.max() / 2

for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color="white" if cm[i, j] > threshold else "black",
            fontsize=13
        )

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

tn, fp, fn, tp = cm.ravel()

m1, m2, m3, m4 = st.columns(4)

m1.metric("Correctly Not Hired", int(tn))
m2.metric("False Hire Predictions", int(fp))
m3.metric("Missed Hired Candidates", int(fn))
m4.metric("Correctly Identified Hires", int(tp))

# =====================================================
# DETAILED SELECTED MODEL
# =====================================================

st.subheader("7. Selected Model Details")

row = metrics.loc[
    metrics["Model"] == selected_model
].iloc[0]

d1, d2, d3 = st.columns(3)

d1.metric("Accuracy", f"{row['Accuracy']:.2%}")
d2.metric("Hired Precision", f"{row['Precision']:.2%}")
d3.metric("Hired Recall", f"{row['Recall']:.2%}")

d4, d5, d6 = st.columns(3)

d4.metric("Hired F1 Score", f"{row['F1 Score']:.4f}")
d5.metric("ROC-AUC", f"{row['ROC-AUC']:.4f}")
d6.metric("PR-AUC", f"{row['PR-AUC']:.4f}")

# =====================================================
# CONCLUSION
# =====================================================

st.subheader("8. Conclusion")

st.markdown("""
**Based on the supplied test results:**

- **Logistic Regression:** strongest choice if detecting hired candidates
  is the priority, with the highest recall and F1-score.
- **CatBoost:** higher accuracy than Logistic Regression, but lower
  recall for the Hired class.
- **Random Forest:** highest accuracy by a small margin, but lowest
  Hired-class F1-score among these three models.
""")

# st.warning(
#     "The confusion matrices for Logistic Regression and Random Forest "
#     "are reconstructed from the reported rounded metrics and class "
#     "supports."
# )

# st.caption(
#     "HR Prediction Project | Model evaluation dashboard"
# )
