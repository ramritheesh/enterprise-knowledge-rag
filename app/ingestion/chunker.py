
import re


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Keep policy headings and paragraphs together where possible."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size.")

    if not text or not text.strip():
        return []

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]

    chunks = []
    current_chunk = ""

    def split_long_text(section: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", section)
        pieces = []
        current = ""

        for sentence in sentences:
            if len(sentence) > chunk_size:
                if current:
                    pieces.append(current)
                    current = ""

                start = 0
                while start < len(sentence):
                    end = min(start + chunk_size, len(sentence))
                    pieces.append(sentence[start:end].strip())

                    if end >= len(sentence):
                        break

                    start = end - overlap

                continue

            candidate = f"{current} {sentence}".strip()

            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    pieces.append(current)
                current = sentence

        if current:
            pieces.append(current)

        return pieces

    for paragraph in paragraphs:
        candidate = (
            f"{current_chunk}\n\n{paragraph}"
            if current_chunk
            else paragraph
        )

        if len(candidate) <= chunk_size:
            current_chunk = candidate
            continue

        if current_chunk:
            chunks.append(current_chunk)
            current_chunk = ""

        if len(paragraph) > chunk_size:
            chunks.extend(split_long_text(paragraph))
        else:
            current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return [chunk for chunk in chunks if chunk.strip()]
