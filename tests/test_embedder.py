"""TEST DOUBLE: genera números deterministas, no embeddings semánticos."""

from dataclasses import asdict
import hashlib
import json

from rag_lab.embeddings import EmbeddingSpec


class DeterministicTestEmbedder:
    query_prefix = "query: "

    def __init__(self, spec: EmbeddingSpec | None = None):
        self.spec = spec or EmbeddingSpec("test-double", "non-semantic-test-model", "1", 4)
        self.provenance = {"kind": "TEST DOUBLE - NOT SEMANTIC"}

    def embed(self, text: str) -> tuple[float, ...]:
        recipe = json.dumps(asdict(self.spec), sort_keys=True) + text
        return tuple(
            int.from_bytes(hashlib.sha256(f"{recipe}:{index}".encode("utf-8")).digest()[:4], "big")
            / (2 ** 32 - 1)
            for index in range(self.spec.dimensions)
        )

    def embed_query(self, text: str) -> tuple[float, ...]:
        return self.embed(self.query_prefix + text)
