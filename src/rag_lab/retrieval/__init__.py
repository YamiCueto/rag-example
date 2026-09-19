"""Cosine manual, ranking exhaustivo y Top-K como pasos separados."""

from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from typing import Sequence

from rag_lab.embeddings import QueryEmbedder
from rag_lab.identity import stable_id
from rag_lab.indexing import VectorIndex


@dataclass(frozen=True)
class RetrievalConfig:
    retrieval_version: str = "1"
    similarity_metric: str = "cosine"
    algorithm: str = "exhaustive"
    top_k: int = 3

    def __post_init__(self) -> None:
        if not isinstance(self.retrieval_version, str) or not self.retrieval_version.strip():
            raise ValueError("retrieval_version debe ser un texto no vacío")
        if self.similarity_metric != "cosine" or self.algorithm != "exhaustive":
            raise ValueError("Solo están implementados cosine y exhaustive")
        validate_top_k(self.top_k)


@dataclass(frozen=True)
class SearchResult:
    rank: int
    score: float
    chunk_id: str
    document_id: str
    chunk_index: int
    start_char: int
    end_char: int
    text: str


@dataclass(frozen=True)
class SearchExperiment:
    retrieval_config_id: str
    retrieval_experiment_id: str
    query: str
    query_prefix: str
    query_embedding: dict
    query_runtime: dict
    vector_index_id: str
    artifact_sha256: str
    config: RetrievalConfig
    candidates: tuple[SearchResult, ...]
    ranking: tuple[SearchResult, ...]
    top_results: tuple[SearchResult, ...]


def load_retrieval_config(path: str | Path) -> RetrievalConfig:
    return RetrievalConfig(**json.loads(Path(path).read_text(encoding="utf-8")))


def validate_top_k(k: int) -> None:
    if type(k) is not int or k <= 0:
        raise ValueError("top_k debe ser un entero positivo")


def unit_vector(vector: Sequence[float]) -> tuple[float, ...]:
    if len(vector) == 0:
        raise ValueError("El vector no puede estar vacío")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in vector):
        raise ValueError("El vector debe contener números finitos, no bool ni texto")
    scale = max(abs(value) for value in vector)
    if scale == 0:
        raise ValueError("El vector tiene norma cero")
    scaled = tuple(value / scale for value in vector)
    norm = math.hypot(*scaled)
    return tuple(value / norm for value in scaled)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Los vectores deben tener las mismas dimensiones")
    left_unit = unit_vector(left)
    right_unit = unit_vector(right)
    score = math.fsum(a * b for a, b in zip(left_unit, right_unit))
    return max(-1.0, min(1.0, score))


def score_all(query_vector: Sequence[float], index: VectorIndex) -> tuple[SearchResult, ...]:
    unit_vector(query_vector)
    if len(query_vector) != index.build_spec["embedding"]["dimensions"]:
        raise ValueError("Dimensión de query incompatible con el índice")
    candidates = []
    for record in index.records:
        metadata = record["metadata"]
        candidates.append(SearchResult(
            rank=0,
            score=cosine_similarity(query_vector, record["vector"]),
            chunk_id=record["chunk_id"],
            document_id=metadata["document_id"],
            chunk_index=metadata["chunk_index"],
            start_char=metadata["start_char"],
            end_char=metadata["end_char"],
            text=metadata["text"],
        ))
    return tuple(candidates)


def rank_candidates(candidates: Sequence[SearchResult]) -> tuple[SearchResult, ...]:
    ordered = sorted(candidates, key=lambda result: (-result.score, result.chunk_id))
    return tuple(replace(result, rank=rank) for rank, result in enumerate(ordered, start=1))


def top_k(ranking: Sequence[SearchResult], k: int) -> tuple[SearchResult, ...]:
    validate_top_k(k)
    return tuple(ranking[:k])


def search(
    query: str, index: VectorIndex, artifact_sha256: str,
    embedder: QueryEmbedder, config: RetrievalConfig,
) -> SearchExperiment:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("La query no puede estar vacía")
    if asdict(embedder.spec) != index.build_spec["embedding"]:
        raise ValueError("La configuración de embeddings de query es incompatible con el índice")
    if embedder.spec.model == "intfloat/multilingual-e5-small" and embedder.query_prefix != "query: ":
        raise ValueError("E5 requiere el prefijo query: para consultas")
    query_vector = embedder.embed_query(query)
    candidates = score_all(query_vector, index)
    ranking = rank_candidates(candidates)
    selected = top_k(ranking, config.top_k)
    config_id = stable_id("retrieval-config", asdict(config))
    experiment_id = stable_id("retrieval-experiment", {
        "vector_index_id": index.vector_index_id,
        "artifact_sha256": artifact_sha256,
        "query": query,
        "query_prefix": embedder.query_prefix,
        "query_embedding": asdict(embedder.spec),
        "retrieval_config_id": config_id,
    })
    return SearchExperiment(
        config_id, experiment_id, query, embedder.query_prefix,
        asdict(embedder.spec), embedder.provenance, index.vector_index_id,
        artifact_sha256, config, candidates, ranking, selected,
    )
