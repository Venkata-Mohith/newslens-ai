"""
chunker.py - Splits article text into overlapping NLP chunks
for embedding and vector storage.
"""

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source_url: str
    source_title: str
    chunk_index: int
    char_start: int
    char_end: int


def _split_into_sentences(text: str) -> list[str]:
    """Simple sentence splitter using regex (no NLTK dependency needed)."""
    # Split on . ! ? followed by space and capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def chunk_text(
    text: str,
    source_url: str,
    source_title: str,
    chunk_size: int = 400,
    overlap: int = 80,
) -> list[Chunk]:
    """
    Split text into overlapping chunks of approximately chunk_size words.
    Overlap ensures context isn't lost at chunk boundaries.
    """
    words = text.split()
    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)

        # Approximate char positions
        char_start = len(" ".join(words[:start]))
        char_end = char_start + len(chunk_text_str)

        if len(chunk_text_str.strip()) > 50:
            chunks.append(Chunk(
                text=chunk_text_str,
                source_url=source_url,
                source_title=source_title,
                chunk_index=chunk_idx,
                char_start=char_start,
                char_end=char_end,
            ))
            chunk_idx += 1

        if end == len(words):
            break
        start += chunk_size - overlap

    return chunks


def chunk_articles(articles: list, chunk_size: int = 400, overlap: int = 80) -> list[Chunk]:
    """Chunk all articles and return flat list of chunks."""
    all_chunks = []
    for article in articles:
        chunks = chunk_text(
            text=article.text,
            source_url=article.url,
            source_title=article.title,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        all_chunks.extend(chunks)
    return all_chunks
