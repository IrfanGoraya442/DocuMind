import os

# Cosine distance threshold — chunks above this are considered irrelevant
RELEVANCE_THRESHOLD = 0.55

NOT_FOUND_MSG = "I could not find this information in the uploaded document."


def _build_prompt(question: str, context: str) -> str:
    return (
        "You are a helpful assistant. Answer the question using ONLY the context below.\n"
        f'If the answer is not in the context, respond exactly with: "{NOT_FOUND_MSG}"\n\n'
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def _answer_openai(question: str, context: str, sources: list[str]) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": _build_prompt(question, context)}],
        temperature=0,
        max_tokens=500,
    )
    return {
        "answer": response.choices[0].message.content.strip(),
        "sources": sources,
    }


def _answer_extractive(question: str, sources: list[str]) -> dict:
    """
    Free fallback: return the most relevant passage with a clean label.
    No LLM download required — works offline.
    """
    top = sources[0]
    answer = (
        f"**From the document:**\n\n{top}\n\n"
        f"*Tip: Add an `OPENAI_API_KEY` to `.env` for AI-generated answers.*"
    )
    return {"answer": answer, "sources": sources}


def get_answer(
    question: str,
    chunks: list[str],
    distances: list[float],
) -> dict:
    """
    Generate an answer from retrieved chunks.
    Uses OpenAI if key is set, otherwise returns the best matching passage.
    """
    relevant = [
        (chunk, dist)
        for chunk, dist in zip(chunks, distances)
        if dist <= RELEVANCE_THRESHOLD
    ]

    if not relevant:
        return {"answer": NOT_FOUND_MSG, "sources": []}

    sources = [c for c, _ in relevant]
    context = "\n\n---\n\n".join(sources)

    if os.environ.get("OPENAI_API_KEY"):
        return _answer_openai(question, context, sources)

    return _answer_extractive(question, sources)
