import os
from spellchecker import SpellChecker

RELEVANCE_THRESHOLD = 0.85  # TF-IDF cosine distance: 0=identical, 1=unrelated
NOT_FOUND_MSG = "I could not find this information in the uploaded document."

_spell = SpellChecker()


def correct_query(text: str) -> tuple[str, bool]:
    """
    Spell-correct each word in the query.
    Returns (corrected_text, was_changed).
    """
    words = text.split()
    corrected = []
    for word in words:
        fixed = _spell.correction(word)
        corrected.append(fixed if fixed else word)
    result = " ".join(corrected)
    return result, result.lower() != text.lower()


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


def _answer_extractive(sources: list[str]) -> dict:
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

    return _answer_extractive(sources)
