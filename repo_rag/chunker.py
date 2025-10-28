from __future__ import annotations

from typing import Iterable, Iterator, List, Sequence

from .documents import Document


def chunk_document(
    document: Document,
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[Document]:
    """Split a document into overlapping chunks."""

    if chunk_overlap >= chunk_size:
        raise ValueError("`chunk_overlap` must be smaller than `chunk_size`.")

    text = document.content
    chunks: List[Document] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk_text = text[start:end]
        metadata = document.as_metadata()
        metadata["chunk_range"] = f"{start}:{end}"
        if text:
            line_start = text.count("\n", 0, start) + 1
            trimmed_chunk = chunk_text.rstrip("\n")
            if trimmed_chunk:
                effective_end = start + len(trimmed_chunk)
                line_end = text.count("\n", 0, effective_end) + 1
            else:
                line_end = line_start
        else:
            line_start = 0
            line_end = 0
        metadata["line_range"] = f"{line_start}:{line_end}"
        chunk_doc = Document(
            path=document.path,
            content=chunk_text,
            metadata=metadata,
        )
        chunks.append(chunk_doc)
        if end == len(text):
            break
        start = end - chunk_overlap
        if start < 0:
            start = 0
    return chunks


def chunk_documents(
    documents: Iterable[Document],
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> Iterator[Document]:
    """Yield chunked documents from an iterable of full documents."""

    for doc in documents:
        for chunk in chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
            yield chunk
