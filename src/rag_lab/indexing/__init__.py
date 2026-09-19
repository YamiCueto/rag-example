"""Índice como colección explícita de vectores; todavía no ofrece búsqueda."""

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile

from rag_lab.dataset import ChunkDataset
from rag_lab.embeddings import Embedder, EmbeddingSpec
from rag_lab.identity import sha256_bytes, stable_id


INDEX_FORMAT = "rag-lab-vector-index-json-v1"


@dataclass(frozen=True)
class VectorIndex:
    vector_index_id: str
    build_spec: dict
    records: tuple[dict, ...]
    provenance: dict


@dataclass(frozen=True)
class IndexArtifact:
    path: Path
    sha256: str
    reused: bool


def load_index(path: str | Path) -> tuple[VectorIndex, IndexArtifact]:
    path = Path(path)
    raw = path.read_bytes()
    checksum = sha256_bytes(raw)
    if path.with_suffix(".sha256").read_text(encoding="ascii").strip() != checksum:
        raise ValueError("El índice no coincide con su checksum")
    data = json.loads(raw)
    if not isinstance(data, dict) or set(data) != {"vector_index_id", "build_spec", "records", "provenance"}:
        raise ValueError("Estructura de índice inválida")
    spec = data["build_spec"]
    if not isinstance(spec, dict) or spec.get("index_format") != INDEX_FORMAT:
        raise ValueError("Formato de índice no compatible")
    if data["vector_index_id"] != stable_id("vector-index", spec):
        raise ValueError("Identidad lógica del índice inválida")
    for name in ("chunk_dataset_id", "embedding_version", "index_version"):
        if not isinstance(spec.get(name), str) or not spec[name]:
            raise ValueError(f"Falta {name} en las dependencias del índice")
    embedding = EmbeddingSpec(**spec["embedding"])
    if not isinstance(data["records"], list) or not isinstance(data["provenance"], dict):
        raise ValueError("Records/provenance inválidos")
    seen = set()
    for record in data["records"]:
        if not isinstance(record, dict) or set(record) != {"chunk_id", "embedding", "vector", "metadata"}:
            raise ValueError("Estructura de vector record inválida")
        if record["embedding"] != asdict(embedding):
            raise ValueError("El record usa una configuración de embedding diferente")
        chunk_id = record["chunk_id"]
        if not isinstance(chunk_id, str) or not chunk_id or chunk_id in seen:
            raise ValueError("Chunk ID inválido o duplicado")
        seen.add(chunk_id)
        vector = record["vector"]
        if (not isinstance(vector, list) or len(vector) != embedding.dimensions
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in vector)):
            raise ValueError("Vector con dimensiones o valores inválidos")
        if not any(value != 0 for value in vector):
            raise ValueError("Vector con norma cero")
        metadata = record["metadata"]
        if not isinstance(metadata, dict) or metadata.get("chunk_id") != chunk_id:
            raise ValueError("Metadatos desconectados del chunk")
        if not isinstance(metadata.get("document_id"), str) or not metadata["document_id"]:
            raise ValueError("Falta document_id")
        for name in ("chunk_index", "start_char", "end_char"):
            if type(metadata.get(name)) is not int or metadata[name] < 0:
                raise ValueError(f"{name} inválido")
        text = metadata.get("text")
        if (not isinstance(text, str) or metadata["end_char"] <= metadata["start_char"]
                or len(text) != metadata["end_char"] - metadata["start_char"]
                or sha256_bytes(text.encode("utf-8")) != metadata.get("text_sha256")):
            raise ValueError("Texto, offsets o checksum del chunk inválidos")
    return (VectorIndex(data["vector_index_id"], spec, tuple(data["records"]), data["provenance"]),
            IndexArtifact(path, checksum, False))


def build_vector_index(
    dataset: ChunkDataset, embedder: Embedder,
    embedding_version: str = "1", index_version: str = "1",
) -> VectorIndex:
    build_spec = {
        "index_format": INDEX_FORMAT,
        "chunk_dataset_id": dataset.chunk_dataset_id,
        "embedding": asdict(embedder.spec),
        "embedding_version": embedding_version,
        "index_version": index_version,
    }
    records = []
    seen = set()
    for chunk in dataset.chunks:
        if chunk.chunk_id in seen:
            raise ValueError("El dataset contiene chunk IDs duplicados")
        seen.add(chunk.chunk_id)
        vector = list(embedder.embed(chunk.text))
        if len(vector) != embedder.spec.dimensions:
            raise ValueError("La dimensión del vector no coincide con el contrato")
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in vector):
            raise ValueError("Los vectores deben contener únicamente números finitos")
        records.append({
            "chunk_id": chunk.chunk_id,
            "embedding": asdict(embedder.spec),
            "vector": vector,
            "metadata": asdict(chunk),
        })
    return VectorIndex(
        stable_id("vector-index", build_spec), build_spec, tuple(records),
        {"corpus_id": dataset.corpus_id, "runtime": embedder.provenance},
    )


def persist_index(index: VectorIndex, root: str | Path) -> IndexArtifact:
    expected_id = stable_id("vector-index", index.build_spec)
    if index.vector_index_id != expected_id:
        raise ValueError("El ID no coincide con las dependencias del índice")
    payload = json.dumps(asdict(index), sort_keys=True, ensure_ascii=False,
                         indent=2, allow_nan=False).encode("utf-8") + b"\n"
    checksum = sha256_bytes(payload)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    directory = root / index.vector_index_id
    artifact = directory / "index.json"

    def verify_existing() -> IndexArtifact:
        existing = artifact.read_bytes()
        recorded = (directory / "index.sha256").read_text(encoding="ascii").strip()
        if sha256_bytes(existing) != recorded:
            raise ValueError("El artefacto existente no coincide con su checksum; no se sobrescribe")
        if existing != payload:
            raise FileExistsError("Misma identidad lógica con bytes diferentes; no se sobrescribe")
        return IndexArtifact(artifact, checksum, True)

    if directory.exists():
        return verify_existing()
    with tempfile.TemporaryDirectory(prefix=".building-", dir=root) as staging:
        staged = Path(staging) / "complete"
        staged.mkdir()
        for filename, content in (("index.json", payload), ("index.sha256", (checksum + "\n").encode("ascii"))):
            with (staged / filename).open("xb") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
        try:
            staged.rename(directory)
        except FileExistsError:
            return verify_existing()
    return IndexArtifact(artifact, checksum, False)
