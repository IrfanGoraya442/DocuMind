from pypdf import PdfReader


def load_pdf(file) -> str:
    """Extract all text from an uploaded PDF file."""
    try:
        reader = PdfReader(file)

        if len(reader.pages) == 0:
            raise ValueError("The PDF is empty (0 pages).")

        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        if not text.strip():
            raise ValueError(
                "No readable text found. This PDF may be scanned or image-based. "
                "Try a text-based PDF."
            )

        return text

    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF: {e}")
