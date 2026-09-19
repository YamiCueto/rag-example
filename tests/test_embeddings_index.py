from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from rag_lab.config import ChunkingConfig, PipelineConfig
from rag_lab.dataset import build_chunk_dataset
from rag_lab.embeddings import EmbeddingSpec
from rag_lab.identity import stable_id
from rag_lab.indexing import build_vector_index, persist_index
from rag_lab.ingestion import load_corpus
from test_embedder import DeterministicTestEmbedder


class EmbeddingIndexTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        (self.root / "source.txt").write_text("Migration rule. " * 70, encoding="utf-8")
        self.config = PipelineConfig()
        self.documents = load_corpus(self.root)
        self.dataset = build_chunk_dataset(self.documents, self.config)
        self.embedder = DeterministicTestEmbedder()

    def index(self):
        return build_vector_index(self.dataset, self.embedder)

    def test_test_double_is_explicit_and_reproducible(self):
        self.assertEqual(self.embedder.spec.provider, "test-double")
        text = self.dataset.chunks[0].text
        self.assertEqual(self.embedder.embed(text), self.embedder.embed(text))
        self.assertEqual(len(self.embedder.embed(text)), self.embedder.spec.dimensions)

    def test_same_dependencies_have_same_logical_id(self):
        self.assertEqual(self.index().vector_index_id, self.index().vector_index_id)

    def test_changed_corpus_or_chunking_changes_index_id(self):
        first = self.index()
        changed = build_chunk_dataset(self.documents, replace(self.config, chunking=ChunkingConfig(chunk_size=1000, overlap=100)))
        self.assertNotEqual(first.vector_index_id, build_vector_index(changed, self.embedder).vector_index_id)
        (self.root / "source.txt").write_text("changed", encoding="utf-8")
        changed = build_chunk_dataset(load_corpus(self.root), self.config)
        self.assertNotEqual(first.vector_index_id, build_vector_index(changed, self.embedder).vector_index_id)

    def test_embedding_and_format_dependencies_change_id(self):
        first = self.index()
        for key, value in (("provider", "other-test-provider"), ("model", "other-test-model"),
                           ("revision", "2"), ("dimensions", 8), ("settings", {"pooling": "different"})):
            with self.subTest(key=key):
                embedder = DeterministicTestEmbedder(replace(self.embedder.spec, **{key: value}))
                self.assertNotEqual(first.vector_index_id, build_vector_index(self.dataset, embedder).vector_index_id)
        self.assertNotEqual(first.vector_index_id, build_vector_index(self.dataset, self.embedder, index_version="2").vector_index_id)
        changed_spec = {**first.build_spec, "index_format": "different-format"}
        self.assertNotEqual(first.vector_index_id, stable_id("vector-index", changed_spec))

    def test_records_preserve_order_chunk_text_and_embedding_configuration(self):
        index = self.index()
        self.assertEqual(len(index.records), len(self.dataset.chunks))
        for record, chunk in zip(index.records, self.dataset.chunks):
            self.assertEqual(record["chunk_id"], chunk.chunk_id)
            self.assertEqual(record["metadata"], asdict(chunk))
            self.assertEqual(record["embedding"], asdict(self.embedder.spec))
            self.assertEqual(record["vector"], list(self.embedder.embed(chunk.text)))

    def test_invalid_dimensions_and_nonfinite_vectors_are_rejected(self):
        class InvalidEmbedder(DeterministicTestEmbedder):
            result = ()

            def embed(self, text):
                return self.result

        embedder = InvalidEmbedder()
        for result in ((1.0,), (float("nan"), 0, 0, 0), (float("inf"), 0, 0, 0), (True, 0, 0, 0), ("1", 0, 0, 0)):
            with self.subTest(result=result), self.assertRaises(ValueError):
                embedder.result = result
                build_vector_index(self.dataset, embedder)

    def test_duplicate_chunk_ids_are_rejected(self):
        dataset = replace(self.dataset, chunks=(self.dataset.chunks[0], self.dataset.chunks[0]))
        with self.assertRaises(ValueError):
            build_vector_index(dataset, self.embedder)

    def test_persistence_preserves_links_and_checksum(self):
        index = self.index()
        artifact = persist_index(index, self.root / "indexes")
        raw = artifact.path.read_bytes()
        stored = json.loads(raw)
        self.assertEqual(stored["vector_index_id"], index.vector_index_id)
        self.assertEqual(stored["build_spec"]["chunk_dataset_id"], self.dataset.chunk_dataset_id)
        self.assertEqual(stored["build_spec"]["embedding"], asdict(self.embedder.spec))
        self.assertEqual(len(stored["records"]), len(self.dataset.chunks))
        self.assertEqual(artifact.sha256, hashlib.sha256(raw).hexdigest())
        self.assertEqual(artifact.sha256, artifact.path.with_suffix(".sha256").read_text().strip())
        self.assertNotEqual(index.vector_index_id.removeprefix("vector-index-"), artifact.sha256)

    def test_same_bytes_have_same_checksum_and_reuse_verified_artifact(self):
        first = persist_index(self.index(), self.root / "indexes")
        second = persist_index(self.index(), self.root / "indexes")
        third = persist_index(self.index(), self.root / "other-indexes")
        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual(first.sha256, third.sha256)
        self.assertFalse(first.reused)
        self.assertTrue(second.reused)

    def test_changed_bytes_under_same_id_are_not_overwritten(self):
        first = self.index()
        artifact = persist_index(first, self.root / "indexes")
        original = artifact.path.read_bytes()
        changed = replace(first, provenance={"different_runtime": True})
        self.assertEqual(first.vector_index_id, changed.vector_index_id)
        with self.assertRaises(FileExistsError):
            persist_index(changed, self.root / "indexes")
        self.assertEqual(artifact.path.read_bytes(), original)

    def test_corruption_is_detected_without_overwrite(self):
        index = self.index()
        artifact = persist_index(index, self.root / "indexes")
        artifact.path.write_bytes(b"corrupt")
        with self.assertRaises(ValueError):
            persist_index(index, self.root / "indexes")
        self.assertEqual(artifact.path.read_bytes(), b"corrupt")

    def test_mismatched_logical_id_is_rejected_before_writing(self):
        with self.assertRaises(ValueError):
            persist_index(replace(self.index(), vector_index_id="wrong"), self.root / "indexes")
        self.assertFalse((self.root / "indexes").exists())

    def test_embedding_contract_rejects_empty_identity_and_invalid_dimensions(self):
        for changes in ({"provider": ""}, {"revision": ""}, {"dimensions": 0}, {"dimensions": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.embedder.spec, **changes)


if __name__ == "__main__":
    unittest.main()
