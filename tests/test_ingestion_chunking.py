from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from rag_lab.chunking import chunk_document
from rag_lab.config import ChunkingConfig, PipelineConfig, load_config
from rag_lab.dataset import build_chunk_dataset
from rag_lab.ingestion import corpus_id, load_corpus, load_document


class IngestionChunkingTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.path = self.root / "sample.txt"
        self.path.write_bytes(("0123456789" * 105).encode("utf-8"))
        self.config = PipelineConfig()

    def document(self):
        return load_document(self.path, self.root)

    def test_same_document_has_same_hash_and_id(self):
        first = self.document()
        self.assertEqual(first, self.document())
        self.assertEqual(first.source_sha256, hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_source_change_changes_document_and_corpus(self):
        first = self.document()
        self.path.write_bytes(b"different")
        second = self.document()
        self.assertNotEqual(first.document_id, second.document_id)
        self.assertNotEqual(first.source_sha256, second.source_sha256)
        self.assertNotEqual(corpus_id((first,)), corpus_id((second,)))

    def test_minimal_normalization_preserves_source_and_spaces(self):
        raw = "\ufeff  Hola\r\n\rMundo  \n".encode("utf-8")
        self.path.write_bytes(raw)
        document = self.document()
        self.assertEqual(document.source_text, raw.decode("utf-8"))
        self.assertEqual(document.text, "\ufeff  Hola\n\nMundo  \n")
        self.assertEqual(document.source_sha256, hashlib.sha256(raw).hexdigest())

    def test_invalid_utf8_fails_without_replacement(self):
        self.path.write_bytes(b"\xff")
        with self.assertRaises(UnicodeDecodeError):
            self.document()

    def test_discovery_is_recursive_sorted_and_txt_only(self):
        (self.root / "nested").mkdir()
        (self.root / "nested" / "a.TXT").write_bytes(b"hello")
        (self.root / "ignore.md").write_bytes(b"ignored")
        documents = load_corpus(self.root)
        self.assertEqual([doc.source_path for doc in documents], ["nested/a.TXT", "sample.txt"])

    def test_corpus_and_dataset_ignore_order_and_absolute_root(self):
        (self.root / "b.txt").write_bytes(b"second")
        documents = load_corpus(self.root)
        with tempfile.TemporaryDirectory() as other:
            for document in documents:
                (Path(other) / document.source_path).write_bytes(document.source_text.encode("utf-8"))
            copied = load_corpus(other)
        self.assertEqual(corpus_id(documents), corpus_id(tuple(reversed(copied))))
        self.assertEqual(build_chunk_dataset(documents, self.config),
                         build_chunk_dataset(tuple(reversed(copied)), self.config))

    def test_rename_changes_identity_and_duplicate_paths_fail(self):
        first = self.document()
        renamed = self.root / "renamed.txt"
        renamed.write_bytes(self.path.read_bytes())
        second = load_document(renamed, self.root)
        self.assertEqual(first.source_sha256, second.source_sha256)
        self.assertNotEqual(first.document_id, second.document_id)
        self.assertNotEqual(corpus_id((first,)), corpus_id((second,)))
        with self.assertRaises(ValueError):
            corpus_id((first, first))

    def test_500_50_offsets_overlap_and_partial_final_chunk(self):
        document = self.document()
        chunks = chunk_document(document, self.config.chunking)
        self.assertEqual([(c.start_char, c.end_char) for c in chunks],
                         [(0, 500), (450, 950), (900, 1050)])
        self.assertEqual([c.chunk_index for c in chunks], [0, 1, 2])
        self.assertEqual([len(c.text) for c in chunks], [500, 500, 150])
        for chunk in chunks:
            self.assertEqual(chunk.text, document.text[chunk.start_char:chunk.end_char])
            self.assertEqual(chunk.text_sha256, hashlib.sha256(chunk.text.encode("utf-8")).hexdigest())
        for previous, current in zip(chunks, chunks[1:]):
            self.assertEqual(previous.text[-50:], current.text[:50])

    def test_chunk_ids_repeat_but_distinguish_offsets(self):
        document = self.document()
        chunks = chunk_document(document, self.config.chunking)
        self.assertEqual(chunks, chunk_document(document, self.config.chunking))
        self.assertEqual(chunks[0].text_sha256, chunks[1].text_sha256)
        self.assertNotEqual(chunks[0].chunk_id, chunks[1].chunk_id)

    def test_small_empty_and_exact_boundary_documents(self):
        for length, ranges in ((0, []), (3, [(0, 3)]), (500, [(0, 500)]),
                               (950, [(0, 500), (450, 950)])):
            with self.subTest(length=length):
                self.path.write_bytes(b"a" * length)
                chunks = chunk_document(self.document(), self.config.chunking)
                self.assertEqual([(c.start_char, c.end_char) for c in chunks], ranges)

    def test_offsets_count_unicode_characters_not_utf8_bytes(self):
        self.path.write_bytes("á🐍x".encode("utf-8"))
        chunks = chunk_document(self.document(), ChunkingConfig(chunk_size=2, overlap=1))
        self.assertEqual([c.text for c in chunks], ["á🐍", "🐍x"])
        self.assertEqual([(c.start_char, c.end_char) for c in chunks], [(0, 2), (1, 3)])

    def test_dataset_is_reproducible_and_chunking_does_not_change_corpus(self):
        documents = load_corpus(self.root)
        first = build_chunk_dataset(documents, self.config)
        self.assertEqual(first, build_chunk_dataset(documents, self.config))
        for chunking in (ChunkingConfig(chunk_size=1000, overlap=100),
                         ChunkingConfig(chunk_size=1000, overlap=50),
                         ChunkingConfig(chunk_size=500, overlap=100)):
            with self.subTest(chunking=chunking):
                second = build_chunk_dataset(documents, replace(self.config, chunking=chunking))
                self.assertEqual(first.corpus_id, second.corpus_id)
                self.assertNotEqual(first.chunk_dataset_id, second.chunk_dataset_id)

    def test_dataset_tracks_upstream_versions_not_downstream(self):
        documents = load_corpus(self.root)
        first = build_chunk_dataset(documents, self.config)
        for name in ("ingestion", "chunking"):
            versions = replace(self.config.versions, **{name: "2"})
            changed = build_chunk_dataset(documents, replace(self.config, versions=versions))
            self.assertNotEqual(first.chunk_dataset_id, changed.chunk_dataset_id)
        versions = replace(self.config.versions, corpus="renamed-label", embedding="2", index="2",
                           retrieval="2", prompt="2", generation="2", eval_dataset="2")
        changed = build_chunk_dataset(documents, replace(self.config, rag_release="2.0.0", versions=versions))
        self.assertEqual(first, changed)

    def test_chunk_overlap_json_alias_and_validation(self):
        path = self.root / "config.json"
        for overlap, valid in ((0, True), (499, True), (-1, False), (500, False), (True, False)):
            with self.subTest(overlap=overlap):
                path.write_text(json.dumps({"chunking": {"chunk_overlap": overlap}}), encoding="utf-8")
                if valid:
                    self.assertEqual(load_config(path).chunking.overlap, overlap)
                else:
                    with self.assertRaises(ValueError):
                        load_config(path)
        path.write_text('{"chunking": {"overlap": 50, "chunk_overlap": 50}}', encoding="utf-8")
        with self.assertRaises(ValueError):
            load_config(path)

    def test_empty_corpus_has_stable_identity_and_no_chunks(self):
        self.assertEqual(corpus_id(()), corpus_id(()))
        self.assertEqual(build_chunk_dataset((), self.config).chunks, ())

    def test_cli_reports_synthetic_corpus_and_rejects_invalid_overlap(self):
        config_path = Path(__file__).resolve().parents[1] / "config" / "pipeline.json"
        command = [sys.executable, "-m", "rag_lab", str(self.root), "--config", str(config_path)]
        result = subprocess.run(command + ["--preview-chunks", "1"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Total chunks: 3", result.stdout)
        self.assertIn("chunk-000 chars [0, 500)", result.stdout)
        invalid = subprocess.run(command + ["--chunk-overlap", "500"], capture_output=True, text=True)
        self.assertEqual(invalid.returncode, 2)


if __name__ == "__main__":
    unittest.main()
