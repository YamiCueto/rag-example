"""E5 local: tokenización, inferencia, mean pooling y normalización visibles."""

from importlib.metadata import version
from pathlib import Path
import platform

from rag_lab.embeddings import EmbeddingSpec
from rag_lab.identity import sha256_bytes


class LocalOnnxEmbedder:
    query_prefix = "query: "

    def __init__(self, spec: EmbeddingSpec, model_directory: str | Path):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        supported = {
            "implementation": "e5-onnx-v1", "precision": "qint8",
            "input_prefix": "passage: ", "max_tokens": 512,
            "overflow": "reject", "pooling": "masked_mean", "normalization": "l2",
        }
        if (spec.provider != "onnxruntime-cpu" or spec.model != "intfloat/multilingual-e5-small"
                or spec.dimensions != 384
                or any(spec.settings.get(key) != value for key, value in supported.items())):
            raise ValueError("Este adaptador implementa únicamente la receta E5 ONNX documentada")
        root = Path(model_directory).resolve()
        files = {}
        for kind in ("model", "tokenizer"):
            path = (root / spec.settings[f"{kind}_file"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Asset fuera del directorio del modelo")
            if sha256_bytes(path.read_bytes()) != spec.settings[f"{kind}_sha256"]:
                raise ValueError(f"Checksum incorrecto para {kind}: {path}")
            files[kind] = path

        self.spec = spec
        self.np = np
        self.tokenizer = Tokenizer.from_file(str(files["tokenizer"]))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(files["model"]), sess_options=options, providers=["CPUExecutionProvider"],
        )
        self.provenance = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {name: version(name) for name in ("onnxruntime", "tokenizers", "numpy")},
            "execution_provider": "CPUExecutionProvider",
            "threads": 1,
        }

    def embed(self, text: str) -> tuple[float, ...]:
        return self._embed_with_prefix(text, self.spec.settings["input_prefix"])

    def embed_query(self, text: str) -> tuple[float, ...]:
        return self._embed_with_prefix(text, self.query_prefix)

    def _embed_with_prefix(self, text: str, prefix: str) -> tuple[float, ...]:
        np = self.np
        encoded = self.tokenizer.encode(prefix + text)
        if len(encoded.ids) > self.spec.settings["max_tokens"]:
            raise ValueError(f"El texto requiere {len(encoded.ids)} tokens; máximo 512, no se trunca")
        arrays = {
            "input_ids": np.array([encoded.ids], dtype=np.int64),
            "attention_mask": np.array([encoded.attention_mask], dtype=np.int64),
            "token_type_ids": np.array([encoded.type_ids], dtype=np.int64),
        }
        inputs = {entry.name: arrays[entry.name] for entry in self.session.get_inputs()}
        hidden = self.session.run(None, inputs)[0]
        if hidden.shape != (1, len(encoded.ids), self.spec.dimensions):
            raise ValueError(f"Forma inesperada de salida ONNX: {hidden.shape}")
        mask = arrays["attention_mask"][..., None].astype(np.float32)
        pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
        norm = np.linalg.norm(pooled, axis=1, keepdims=True)
        if not np.isfinite(pooled).all() or not np.isfinite(norm).all() or (norm == 0).any():
            raise ValueError("El modelo devolvió un vector no finito o nulo")
        vector = (pooled / norm)[0]
        return tuple(float(value) for value in vector)
