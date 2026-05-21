import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def create_collection(chunks: list[str]) -> dict:
    """Build a TF-IDF index from document chunks."""
    if not chunks:
        raise ValueError("No chunks provided.")

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(chunks)

    return {"vectorizer": vectorizer, "matrix": matrix, "chunks": chunks}


def query_collection(
    collection: dict,
    question: str,
    n_results: int = 4,
) -> tuple[list[str], list[float]]:
    """Return (documents, distances) — distance = 1 - cosine_similarity."""
    vectorizer = collection["vectorizer"]
    matrix = collection["matrix"]
    chunks = collection["chunks"]

    q_vec = vectorizer.transform([question])
    scores = cosine_similarity(q_vec, matrix)[0]

    top_n = min(n_results, len(chunks))
    top_indices = np.argsort(scores)[-top_n:][::-1]

    docs = [chunks[i] for i in top_indices]
    distances = [float(1 - scores[i]) for i in top_indices]  # 0=identical, 1=unrelated

    return docs, distances
