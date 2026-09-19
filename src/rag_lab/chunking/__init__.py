"""Ventanas manuales sobre caracteres del texto normalizado."""

from dataclasses import asdict, dataclass

from rag_lab.config import ChunkingConfig
from rag_lab.identity import sha256_bytes, stable_id
from rag_lab.ingestion import Document


@dataclass(frozen=True)
class Chunk:
    document_id: str
    chunk_id: str
    chunk_index: int
    start_char: int
    end_char: int
    text: str
    text_sha256: str


def chunk_document(
    document: Document,
    config: ChunkingConfig,
    ingestion_version: str = "1",
    chunking_version: str = "1",
) -> tuple[Chunk, ...]:
    chunks = []
    start = 0
    while start < len(document.text):
        end = min(start + config.chunk_size, len(document.text))
        text = document.text[start:end]
        text_sha256 = sha256_bytes(text.encode("utf-8"))
        chunk_index = len(chunks)
        chunk_id = stable_id("chunk", {
            "document_id": document.document_id,
            "ingestion_version": ingestion_version,
            "normalization": document.normalization,
            "chunking_version": chunking_version,
            "config": asdict(config),
            "chunk_index": chunk_index,
            "start_char": start,
            "end_char": end,
            "text_sha256": text_sha256,
        })
        chunks.append(Chunk(
            document.document_id, chunk_id, chunk_index, start, end, text, text_sha256,
        ))
        if end == len(document.text):
            break
        start += config.chunk_size - config.overlap
    return tuple(chunks)
