"""Identidades SHA-256 con una serialización explícita y compartida."""

import hashlib
import json


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def stable_id(kind: str, payload: object) -> str:
    encoded = json.dumps(
        {"kind": kind, "schema_version": 1, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return f"{kind}-{sha256_bytes(encoded)}"
