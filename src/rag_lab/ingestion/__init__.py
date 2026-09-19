"""Ingesta de texto UTF-8; no ejecuta chunking."""

from dataclasses import dataclass
from pathlib import Path

from rag_lab.identity import sha256_bytes, stable_id


NORMALIZATION = "newlines_to_lf_v1"


@dataclass(frozen=True)
class Document:
    document_id: str
    source_path: str
    source_sha256: str
    source_text: str
    text: str
    normalization: str = NORMALIZATION
    encoding: str = "utf-8"


def load_document(path: str | Path, corpus_root: str | Path) -> Document:
    path = Path(path)
    if path.suffix.lower() != ".txt":
        raise ValueError(f"Solo se admiten documentos .txt: {path}")
    source_path = path.resolve().relative_to(Path(corpus_root).resolve()).as_posix()
    source = path.read_bytes()
    source_text = source.decode("utf-8")
    source_sha256 = sha256_bytes(source)
    document_id = stable_id("document", {
        "source_path": source_path, "source_sha256": source_sha256,
    })
    return Document(
        document_id=document_id,
        source_path=source_path,
        source_sha256=source_sha256,
        source_text=source_text,
        text=source_text.replace("\r\n", "\n").replace("\r", "\n"),
    )


def load_corpus(root: str | Path) -> tuple[Document, ...]:
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"El corpus debe ser un directorio existente: {root}")
    paths = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".txt"),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    return tuple(load_document(path, root) for path in paths)


def corpus_id(documents: tuple[Document, ...]) -> str:
    sources = sorted(
        ({"source_path": doc.source_path, "source_sha256": doc.source_sha256}
         for doc in documents),
        key=lambda source: source["source_path"],
    )
    if len({source["source_path"] for source in sources}) != len(sources):
        raise ValueError("El corpus contiene rutas de documento duplicadas")
    return stable_id("corpus", sources)
