"""Composición de ingestion y chunking, sin artefactos de etapas posteriores."""

from dataclasses import asdict, dataclass

from rag_lab.chunking import Chunk, chunk_document
from rag_lab.config import PipelineConfig
from rag_lab.identity import stable_id
from rag_lab.ingestion import Document, corpus_id


@dataclass(frozen=True)
class ChunkDataset:
    corpus_id: str
    chunk_dataset_id: str
    chunks: tuple[Chunk, ...]


def build_chunk_dataset(
    documents: tuple[Document, ...], config: PipelineConfig,
) -> ChunkDataset:
    corpus_identity = corpus_id(documents)
    ordered = sorted(documents, key=lambda document: document.source_path)
    chunks = tuple(
        chunk
        for document in ordered
        for chunk in chunk_document(
            document, config.chunking, config.versions.ingestion, config.versions.chunking,
        )
    )
    dataset_identity = stable_id("chunk-dataset", {
        "corpus_id": corpus_identity,
        "ingestion_version": config.versions.ingestion,
        "normalization": sorted({document.normalization for document in documents}),
        "chunking_version": config.versions.chunking,
        "chunking_config": asdict(config.chunking),
        "chunk_ids": [chunk.chunk_id for chunk in chunks],
    })
    return ChunkDataset(corpus_identity, dataset_identity, chunks)
