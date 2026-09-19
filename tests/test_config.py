import json
from pathlib import Path
import tempfile
import unittest

from rag_lab.config import ChunkingConfig, ModelConfig, PipelineConfig, load_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_initial_config_matches_defaults(self):
        self.assertEqual(load_config(ROOT / "config/pipeline.json"), PipelineConfig())

    def test_json_round_trip(self):
        config = PipelineConfig(chunking=ChunkingConfig(chunk_size=1000, overlap=100))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pipeline.json"
            path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
            self.assertEqual(load_config(path), config)

    def test_invalid_chunking(self):
        for values in ({"chunk_size": 0}, {"chunk_size": True}, {"overlap": -1},
                       {"overlap": 500}, {"overlap": 1.5}, {"strategy": "unknown"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ChunkingConfig(**values)

    def test_unselected_models_are_explicit(self):
        self.assertIsNone(PipelineConfig().embedding.model_id)
        with self.assertRaises(ValueError):
            ModelConfig(revision="v1")

    def test_invalid_json_configuration_is_rejected(self):
        for data in ({"unknown": 1}, {"chunking": {"unknown": 1}},
                     {"versions": {"corpus": 1}}, {"chunking": []}, []):
            with self.subTest(data=data), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "pipeline.json"
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises((TypeError, ValueError)):
                    load_config(path)

    def test_manifest_template_does_not_claim_an_artifact(self):
        manifest = json.loads((ROOT / "docs/manifest.example.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "planned")
        self.assertIsNone(manifest["index"]["artifact_path"])
        self.assertIsNone(manifest["index"]["artifact_sha256"])


if __name__ == "__main__":
    unittest.main()
