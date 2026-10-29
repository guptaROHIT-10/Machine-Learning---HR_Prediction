
import json
import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier
from sklearn.metrics.pairwise import cosine_similarity

from pypdf import PdfReader
from docx import Document


# healper funtion...
def extract_cv_text(uploaded_file):
    """Extract text from an uploaded PDF or DOCX CV."""

    if uploaded_file is None:
        return ""

    file_name = uploaded_file.name.lower()

    try:
        if file_name.endswith(".pdf"):
            reader = PdfReader(uploaded_file)

            text = "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )

        elif file_name.endswith(".docx"):
            document = Document(uploaded_file)

            text = "\n".join(
                paragraph.text
                for paragraph in document.paragraphs
            )

        else:
            raise ValueError(
                "Unsupported file type. Upload a PDF or DOCX file."
            )

        return text.strip()

    except Exception as exc:
        raise ValueError(
            f"Could not read the uploaded CV: {exc}"
        ) from exc


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "hiring_catboost_nlp.cbm"
METADATA_PATH = BASE_DIR / "models" / "hiring_metadata.json"
VECTORIZER_PATH = (
    BASE_DIR / "models" / "hiring_tfidf_vectorizer.joblib"
)

# Demonstration threshold only.
THRESHOLD = 0.40


# ==========================================
# 2. STREAMLIT CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Hiring Assistant",
    page_icon="🌐",
    layout="centered"
)

st.title("AI Hiring Assistant")
st.write(
    "Enter candidate details and provide a CV and job description "
    "to generate a model prediction for recruiter review."
)


# ==========================================
# 3. LOAD MODEL, METADATA AND VECTORIZER
# ==========================================

@st.cache_resource
def load_artifacts():
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))

    with open(METADATA_PATH, "r") as file:
        metadata = json.load(file)

    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, metadata, vectorizer


required_files = [
    MODEL_PATH,
    METADATA_PATH,
    VECTORIZER_PATH
]

missing_files = [
    str(path.name)
    for path in required_files
    if not path.exists()
]

if missing_files:
    st.error(
        "Required model files are missing: "
        + ", ".join(missing_files)
        + ". Train the NLP-enhanced model first."
    )
    st.stop()

try:
    model, metadata, vectorizer = load_artifacts()
except Exception as error:
    st.error(f"Could not load the model artifacts: {error}")
    st.stop()


# ==========================================
# 4. TEXT PREPROCESSING
# ==========================================

def clean_text(text):
    if not isinstance(text, str):
        return ""

    return re.sub(r"\s+", " ", text.lower()).strip()


# ==========================================
# 5. CANDIDATE INPUT FORM
# ==========================================

with st.form("candidate_form"):

    st.header("Candidate Information")

    job_role = st.selectbox(
        "Job Role",
        [
            "Software Engineer",
            "Data Analyst",
            "Data Scientist",
            "Frontend Developer",
            "Backend Developer",
            "QA Engineer",
            "DevOps Engineer",
            "Business Analyst",
            "Cybersecurity Analyst",
            "IT Support Engineer"
        ]
    )

    education = st.selectbox(
        "Education Level",
        ["Diploma", "Bachelor's", "Master's", "PhD"]
    )

    experience = st.number_input(
        "Relevant Experience (years)",
        min_value=0.0,
        max_value=50.0,
        value=2.0,
        step=0.5
    )

    st.subheader("Assessment Scores")

    technical_score = st.slider(
        "Technical Score", 0, 100, 75
    )

    interview_score = st.slider(
        "Interview Score", 0, 100, 75
    )

    coding_score = st.slider(
        "Coding Test Score", 0, 100, 75
    )

    skills_match = st.slider(
        "Required Skills Match (%)", 0, 100, 75
    )

    communication_score = st.slider(
        "Communication Score", 0, 100, 75
    )

    st.subheader("Additional Information")

    certifications = st.number_input(
        "Relevant Certifications",
        min_value=0,
        max_value=50,
        value=1
    )

    projects = st.number_input(
        "Relevant Projects",
        min_value=0,
        max_value=50,
        value=2
    )

    notice_period = st.selectbox(
        "Notice Period (days)",
        [0, 7, 15, 30, 45, 60, 90]
    )

    expected_salary = st.number_input(
        "Expected Salary (LPA)",
        min_value=0.0,
        max_value=100.0,
        value=6.0,
        step=0.5
    )

    salary_budget = st.number_input(
        "Offered Salary Budget (LPA)",
        min_value=0.0,
        max_value=100.0,
        value=8.0,
        step=0.5
    )

    st.divider()
    st.header("NLP - Resume Analysis")

    job_description = st.text_area(
    "Job Description",
    placeholder="Enter the job responsibilities and required skills..."
)

    uploaded_cv = st.file_uploader(
    "Upload candidate CV",
    type=["pdf", "docx"],
    help="Upload a PDF or Word document."

)

    submitted = st.form_submit_button(
        "Analyse Candidate",
        type="primary",
        use_container_width=True
    )


# ==========================================
# 6. PROCESS INPUT AND PREDICT
# ==========================================

if submitted:

    if uploaded_cv is None:
        st.error("Please upload a candidate CV.")
        st.stop()

    if not job_description.strip():
        st.error("Please enter the job description.")
        st.stop()

    try:
        resume_text = extract_cv_text(uploaded_cv)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    if not resume_text:
        st.error(
            "No readable text was found. "
            "If this is a scanned PDF, OCR is required."
        )
        st.stop()

    # --------------------------------------
    # A. Calculate NLP similarity
    # --------------------------------------

    resume_vector = vectorizer.transform([
        clean_text(resume_text)
    ])

    job_vector = vectorizer.transform([
        clean_text(job_description)
    ])

    similarity = float(
        cosine_similarity(
            resume_vector,
            job_vector
        )[0, 0]
    )

    st.write(f"Resume–job similarity: {similarity:.1%}")

    # --------------------------------------
    # B. Continue with your existing code
    # --------------------------------------
    # Add your candidate dictionary, feature preparation,
    # model prediction, and result display here.

    candidate = {
        "Job_Role": job_role,
        "Education_Level": education,
        "Relevant_Experience_Years": experience,
        "Technical_Score": technical_score,
        "Interview_Score": interview_score,
        "Coding_Test_Score": coding_score,
        "Required_Skills_Match_Percent": skills_match,
        "Communication_Score": communication_score,
        "Relevant_Certifications_Count": certifications,
        "Relevant_Projects_Count": projects,
        "Notice_Period_Days": notice_period,
        "Expected_Salary_LPA": expected_salary,
        "Offered_Salary_Budget_LPA": salary_budget,
        "Resume_Job_Similarity": similarity
    }

    # Use precisely the feature columns from training.
    feature_columns = metadata["feature_columns"]

    X_input = pd.DataFrame([candidate]).reindex(
        columns=feature_columns
    )

    # --------------------------------------
    # C. Apply saved preprocessing conventions
    # --------------------------------------

    categorical_columns = metadata["categorical_columns"]

    for col in categorical_columns:
        if col in X_input.columns:
            X_input[col] = (
                X_input[col]
                .fillna("Missing")
                .astype(str)
            )

    numeric_medians = metadata.get("numeric_medians", {})

    for col, median in numeric_medians.items():
        if col in X_input.columns:
            X_input[col] = X_input[col].fillna(median)

    if X_input.isnull().any().any():
        st.error(
            "Some required model features are missing. "
            "Check your training metadata and form fields."
        )
        st.stop()

    # --------------------------------------
    # D. Generate prediction
    # --------------------------------------

    try:
        probability = float(
            model.predict_proba(X_input)[0][1]
        )
    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.stop()


       
    # Generate prediction using BOTH conditions

    PROBABILITY_THRESHOLD = metadata.get("threshold", 0.50)
    MIN_SIMILARITY = 0.40  # Rule: 40% minimum similarity

    if (
        similarity >= MIN_SIMILARITY
        and probability >= PROBABILITY_THRESHOLD
    ):
        prediction = "Hired"
    else:
        prediction = "Not Hired"



    # --------------------------------------
    # E. Display results
    # --------------------------------------

    st.divider()
    st.header("Analysis Results")

    st.metric(
        "Resume–Job Description Similarity",
        f"{similarity:.1%}"
    )

    st.progress(similarity)

    st.metric(
        "Model-Estimated Probability of Hired",
        f"{probability:.1%}"
    )

    if prediction == "Hired":
        st.success(
            "Model prediction: Hired"
        )
    else:
        st.warning(
            "Model prediction: Not Hired"
        )

    # st.caption(
    #     f"Demonstration threshold: {THRESHOLD:.2f}. "
    #     "The predicted probability is not necessarily a calibrated "
    #     "real-world hiring probability."
    # )

    with st.expander("View submitted information"):
        st.write("**Job role:**", job_role)
        st.write("**Education:**", education)
        st.write("**Experience:**", f"{experience} years")
        st.write(
            "**Technical / Interview / Coding scores:**",
            f"{technical_score} / {interview_score} / {coding_score}"
        )
        st.write("**NLP similarity:**", f"{similarity:.4f}")

    st.info(
        "This is as an automatic hiring or rejection decision. "
        "A qualified recruiter should also review & varify the candidate's "
        "qualifications and supporting evidence."
    )
