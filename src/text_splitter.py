def split_text(
    text: str,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> list[str]:
    """Split text into overlapping chunks on paragraph/sentence boundaries."""
    if not text.strip():
        raise ValueError("Cannot split empty text.")

    # Try to split on double newlines first, then single, then spaces
    separators = ["\n\n", "\n", ". ", " "]
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        # Find the best split point near `end`
        split_at = end
        for sep in separators:
            pos = text.rfind(sep, start, end)
            if pos > start:
                split_at = pos + len(sep)
                break

        chunk = text[start:split_at].strip()
        if chunk:
            chunks.append(chunk)

        start = split_at - chunk_overlap  # overlap
        if start <= 0:
            start = split_at

    if not chunks:
        raise ValueError("Text splitting produced no chunks.")

    return chunks
