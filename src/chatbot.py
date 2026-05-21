import os
from spellchecker import SpellChecker

RELEVANCE_THRESHOLD = 0.85
NOT_FOUND_MSG = "I could not find this information in the uploaded document."

_spell = SpellChecker()


def correct_query(text: str) -> tuple[str, bool]:
    """Spell-correct each word in the query. Returns (corrected, was_changed)."""
    words = text.split()
    corrected = [_spell.correction(w) or w for w in words]
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


_GEMINI_MODELS = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-pro",
    "gemini-1.0-pro",
]


def _answer_gemini(question: str, context: str, sources: list[str]) -> dict:
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    prompt = _build_prompt(question, context)
    last_error = None

    for model_name in _GEMINI_MODELS:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return {"answer": response.text.strip(), "sources": sources}
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"No Gemini model available: {last_error}")


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
        f"*Tip: Add a `GEMINI_API_KEY` to Streamlit secrets for AI-generated answers.*"
    )
    return {"answer": answer, "sources": sources}


def get_answer(question: str, chunks: list[str], distances: list[float]) -> dict:
    """
    Priority: Gemini (free) → OpenAI → Extractive fallback.
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

    if os.environ.get("GEMINI_API_KEY"):
        return _answer_gemini(question, context, sources)

    if os.environ.get("OPENAI_API_KEY"):
        return _answer_openai(question, context, sources)

    return _answer_extractive(sources)
