"""Contrato visible para transformar un texto en un vector."""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class EmbeddingSpec:
    provider: str
    model: str
    revision: str
    dimensions: int
    settings: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("provider", "model", "revision"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} debe ser un texto no vacío")
        if type(self.dimensions) is not int or self.dimensions <= 0:
            raise ValueError("dimensions debe ser un entero positivo")
        if not isinstance(self.settings, dict):
            raise ValueError("settings debe ser un objeto JSON")


class Embedder(Protocol):
    spec: EmbeddingSpec
    provenance: dict

    def embed(self, text: str) -> tuple[float, ...]: ...


class QueryEmbedder(Embedder, Protocol):
    query_prefix: str

    def embed_query(self, text: str) -> tuple[float, ...]: ...


def load_embedding_spec(path: str | Path) -> EmbeddingSpec:
    return EmbeddingSpec(**json.loads(Path(path).read_text(encoding="utf-8")))
