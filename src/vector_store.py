import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

COLLECTION_NAME = "documind"
# DefaultEmbeddingFunction uses all-MiniLM-L6-v2 via onnxruntime — no PyTorch needed


def create_collection(chunks: list[str]):
    """
    Embed chunks and store in a new in-memory ChromaDB collection.
    Returns (client, collection) — keep both in session state.
    """
    if not chunks:
        raise ValueError("No chunks provided for embedding.")

    client = chromadb.Client()
    ef = DefaultEmbeddingFunction()

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 5000
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.add(
            documents=batch,
            ids=[f"chunk_{i + j}" for j in range(len(batch))],
        )

    return client, collection


def query_collection(
    collection,
    question: str,
    n_results: int = 4,
) -> tuple[list[str], list[float]]:
    """Return (documents, distances) for the most relevant chunks."""
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    actual_n = min(n_results, collection.count())
    if actual_n == 0:
        return [], []

    results = collection.query(
        query_texts=[question],
        n_results=actual_n,
    )

    return results["documents"][0], results["distances"][0]
