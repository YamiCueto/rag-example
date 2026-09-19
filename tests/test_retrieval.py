from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rag_lab.config import PipelineConfig
from rag_lab.dataset import build_chunk_dataset
from rag_lab.embeddings.onnx_local import LocalOnnxEmbedder
from rag_lab.identity import stable_id
from rag_lab.indexing import build_vector_index, load_index, persist_index
from rag_lab.ingestion import load_corpus
from rag_lab.retrieval import (
    RetrievalConfig, cosine_similarity, load_retrieval_config,
    rank_candidates, score_all, search, top_k,
)
from test_embedder import DeterministicTestEmbedder


class CosineTests(unittest.TestCase):
    def test_identical_orthogonal_and_opposite_vectors(self):
        self.assertAlmostEqual(cosine_similarity((1, 2, 3), (1, 2, 3)), 1)
        self.assertAlmostEqual(cosine_similarity((1, 0), (0, 1)), 0)
        self.assertAlmostEqual(cosine_similarity((1, 0), (-1, 0)), -1)

    def test_empty_dimension_mismatch_and_zero_norm_fail(self):
        for left, right in (((), ()), ((1,), (1, 2)), ((0, 0), (1, 2)), ((1, 2), (0, 0))):
            with self.subTest(left=left, right=right), self.assertRaises(ValueError):
                cosine_similarity(left, right)

    def test_invalid_numbers_fail(self):
        for value in (True, "1", None, float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                cosine_similarity((value, 1), (1, 2))

    def test_large_and_small_finite_values_do_not_overflow_or_underflow(self):
        self.assertAlmostEqual(cosine_similarity((1e308, 1e308), (1e308, 1e308)), 1)
        self.assertAlmostEqual(cosine_similarity((1e-308, 0), (1e-308, 0)), 1)


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        (self.root / "source.txt").write_text("Events and migration. " * 70, encoding="utf-8")
        self.dataset = build_chunk_dataset(load_corpus(self.root), PipelineConfig())
        self.embedder = DeterministicTestEmbedder()
        self.index = build_vector_index(self.dataset, self.embedder)
        self.artifact = persist_index(self.index, self.root / "indexes")

    def test_same_query_has_reproducible_ranking(self):
        first = search("events", self.index, self.artifact.sha256, self.embedder, RetrievalConfig())
        second = search("events", self.index, self.artifact.sha256, self.embedder, RetrievalConfig())
        self.assertEqual(first, second)
        self.assertEqual([r.rank for r in first.ranking], list(range(1, len(self.index.records) + 1)))
        self.assertEqual([r.score for r in first.ranking], sorted((r.score for r in first.ranking), reverse=True))

    def test_search_evaluates_every_vector_before_top_k(self):
        with patch("rag_lab.retrieval.cosine_similarity", wraps=cosine_similarity) as cosine:
            result = search("events", self.index, self.artifact.sha256, self.embedder, RetrievalConfig(top_k=1))
        self.assertEqual(cosine.call_count, len(self.index.records))
        self.assertEqual(len(result.candidates), len(self.index.records))
        self.assertEqual(len(result.top_results), 1)
        self.assertEqual(result.top_results, result.ranking[:1])

    def test_top_k_bounds_no_threshold_and_tie_break(self):
        candidates = score_all((1, 0, 0, 0), self.index)
        candidates = tuple(replace(row, score=-0.9) for row in candidates)
        ranking = rank_candidates(tuple(reversed(candidates)))
        self.assertEqual([row.chunk_id for row in ranking], sorted(row.chunk_id for row in candidates))
        self.assertEqual(len(top_k(ranking, 2)), 2)
        self.assertEqual(top_k(ranking, 100), ranking)
        self.assertTrue(all(row.score < 0 for row in top_k(ranking, 2)))
        self.assertEqual(top_k((), 3), ())

    def test_invalid_top_k_fails(self):
        for k in (0, -1, 1.5, True, "3"):
            with self.subTest(k=k), self.assertRaises(ValueError):
                top_k((), k)
            with self.subTest(config_k=k), self.assertRaises(ValueError):
                RetrievalConfig(top_k=k)

    def test_results_preserve_traceability(self):
        result = search("events", self.index, self.artifact.sha256, self.embedder, RetrievalConfig())
        for candidate, original in zip(result.candidates, self.dataset.chunks):
            self.assertEqual(candidate.chunk_id, original.chunk_id)
            for name in ("document_id", "chunk_index", "start_char", "end_char", "text"):
                self.assertEqual(getattr(candidate, name), getattr(original, name))

    def test_incompatible_embedding_fails_before_inference(self):
        for key, value in (("provider", "other"), ("model", "other"), ("revision", "2"),
                           ("dimensions", 8), ("settings", {"pooling": "other"})):
            with self.subTest(key=key):
                embedder = DeterministicTestEmbedder(replace(self.embedder.spec, **{key: value}))
                with patch.object(embedder, "embed_query") as infer, self.assertRaises(ValueError):
                    search("events", self.index, self.artifact.sha256, embedder, RetrievalConfig())
                infer.assert_not_called()

    def test_k_changes_experiment_not_index(self):
        original_bytes = self.artifact.path.read_bytes()
        first = search("events", self.index, self.artifact.sha256, self.embedder, RetrievalConfig(top_k=1))
        second = search("events", self.index, self.artifact.sha256, self.embedder, RetrievalConfig(top_k=3))
        self.assertNotEqual(first.retrieval_config_id, second.retrieval_config_id)
        self.assertNotEqual(first.retrieval_experiment_id, second.retrieval_experiment_id)
        self.assertEqual(first.vector_index_id, second.vector_index_id)
        self.assertEqual(first.ranking, second.ranking)
        self.assertEqual(original_bytes, self.artifact.path.read_bytes())

    def test_query_and_artifact_checksum_are_experiment_dependencies(self):
        first = search("events", self.index, self.artifact.sha256, self.embedder, RetrievalConfig())
        changed = search("other", self.index, self.artifact.sha256, self.embedder, RetrievalConfig())
        changed_artifact = search("events", self.index, "0" * 64, self.embedder, RetrievalConfig())
        self.assertNotEqual(first.retrieval_experiment_id, changed.retrieval_experiment_id)
        self.assertNotEqual(first.retrieval_experiment_id, changed_artifact.retrieval_experiment_id)

    def test_empty_query_fails(self):
        with self.assertRaises(ValueError):
            search("  ", self.index, self.artifact.sha256, self.embedder, RetrievalConfig())

    def test_e5_prefixes_are_explicit_without_loading_model(self):
        embedder = object.__new__(LocalOnnxEmbedder)
        embedder.spec = replace(self.embedder.spec, settings={"input_prefix": "passage: "})
        with patch.object(embedder, "_embed_with_prefix", return_value=(1.0,)) as encode:
            embedder.embed("document")
            encode.assert_called_with("document", "passage: ")
            embedder.embed_query("question")
            encode.assert_called_with("question", "query: ")

    def test_retrieval_json_defaults_and_invalid_algorithm(self):
        source = Path(__file__).resolve().parents[1] / "config/retrieval.json"
        self.assertEqual(load_retrieval_config(source), RetrievalConfig())
        with self.assertRaises(ValueError):
            RetrievalConfig(algorithm="ann")

    def test_load_index_checks_checksum_identity_and_metadata(self):
        loaded, artifact = load_index(self.artifact.path)
        self.assertEqual(loaded, self.index)
        self.assertEqual(artifact.sha256, self.artifact.sha256)
        original = json.loads(self.artifact.path.read_bytes())
        for mutation in ("checksum", "identity", "format", "dimensions", "embedding", "metadata", "zero", "duplicate"):
            with self.subTest(mutation=mutation):
                data = json.loads(json.dumps(original))
                if mutation == "identity":
                    data["vector_index_id"] = "wrong"
                elif mutation == "format":
                    data["build_spec"]["index_format"] = "wrong"
                    data["vector_index_id"] = stable_id("vector-index", data["build_spec"])
                elif mutation == "dimensions":
                    data["records"][0]["vector"] = [1]
                elif mutation == "embedding":
                    data["records"][0]["embedding"]["revision"] = "different"
                elif mutation == "metadata":
                    data["records"][0]["metadata"]["text"] = "changed"
                elif mutation == "zero":
                    data["records"][0]["vector"] = [0] * 4
                elif mutation == "duplicate":
                    data["records"].append(data["records"][0])
                raw = json.dumps(data).encode("utf-8")
                self.artifact.path.write_bytes(raw)
                checksum = hashlib.sha256(raw).hexdigest() if mutation != "checksum" else "wrong"
                self.artifact.path.with_suffix(".sha256").write_text(checksum, encoding="ascii")
                with self.assertRaises(ValueError):
                    load_index(self.artifact.path)


if __name__ == "__main__":
    unittest.main()
