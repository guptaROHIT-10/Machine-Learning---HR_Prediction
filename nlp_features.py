
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.lower()).strip()


def fit_text_vectorizer(resumes, descriptions):
    # Fit only on training data to prevent data leakage.
    corpus = [
        clean_text(text)
        for text in list(resumes) + list(descriptions)
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000
    )
    vectorizer.fit(corpus)
    return vectorizer


def similarity_features(resumes, descriptions, vectorizer):
    resume_vectors = vectorizer.transform(
        [clean_text(text) for text in resumes]
    )
    job_vectors = vectorizer.transform(
        [clean_text(text) for text in descriptions]
    )

    scores = cosine_similarity(
        resume_vectors, job_vectors
    ).diagonal()

    return scores
