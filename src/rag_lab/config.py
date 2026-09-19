"""Configuración explícita; no ejecuta ninguna etapa del pipeline."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass(frozen=True)
class PipelineVersions:
    corpus: str = "1"
    ingestion: str = "1"
    chunking: str = "1"
    embedding: str = "1"
    index: str = "1"
    retrieval: str = "1"
    prompt: str = "1"
    generation: str = "1"
    eval_dataset: str = "1"

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"versions.{name} debe ser un texto no vacío")


@dataclass(frozen=True)
class ChunkingConfig:
    strategy: str = "fixed_characters"
    chunk_size: int = 500
    overlap: int = 50

    def __post_init__(self) -> None:
        if self.strategy != "fixed_characters":
            raise ValueError("La estrategia prevista es fixed_characters")
        if type(self.chunk_size) is not int or self.chunk_size <= 0:
            raise ValueError("chunk_size debe ser un entero positivo")
        if type(self.overlap) is not int or not 0 <= self.overlap < self.chunk_size:
            raise ValueError("overlap debe ser un entero entre 0 y chunk_size - 1")


@dataclass(frozen=True)
class ModelConfig:
    model_id: str | None = None
    revision: str | None = None

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} debe ser null o un texto no vacío")
        if self.revision is not None and self.model_id is None:
            raise ValueError("Una revisión requiere model_id")


@dataclass(frozen=True)
class PipelineConfig:
    rag_release: str = "0.1.0"
    versions: PipelineVersions = field(default_factory=PipelineVersions)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: ModelConfig = field(default_factory=ModelConfig)
    generation: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.rag_release, str) or not self.rag_release.strip():
            raise ValueError("rag_release debe ser un texto no vacío")
        for name, expected in (
            ("versions", PipelineVersions),
            ("chunking", ChunkingConfig),
            ("embedding", ModelConfig),
            ("generation", ModelConfig),
        ):
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"{name} debe ser {expected.__name__}")

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path) -> PipelineConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("La configuración debe ser un objeto JSON")
    for name, cls in (
        ("versions", PipelineVersions),
        ("chunking", ChunkingConfig),
        ("embedding", ModelConfig),
        ("generation", ModelConfig),
    ):
        if name in data:
            if not isinstance(data[name], dict):
                raise ValueError(f"{name} debe ser un objeto JSON")
            if name == "chunking" and "chunk_overlap" in data[name]:
                if "overlap" in data[name]:
                    raise ValueError("Usa overlap o chunk_overlap, no ambos")
                data[name]["overlap"] = data[name].pop("chunk_overlap")
            data[name] = cls(**data[name])
    return PipelineConfig(**data)
