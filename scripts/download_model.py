"""Descarga explícita de dos assets públicos, con revisión y SHA-256 fijados."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen


def download_model(config_path: Path, destination: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for kind in ("model", "tokenizer"):
        relative = config["settings"][f"{kind}_file"]
        expected = config["settings"][f"{kind}_sha256"]
        target = destination / relative
        if not target.resolve().is_relative_to(destination.resolve()):
            raise ValueError("El asset debe quedar dentro del directorio del modelo")
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Checksum incorrecto; no se sobrescribe: {target}")
            print(f"Verified: {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{config['model']}/resolve/{config['revision']}/{relative}"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
                temporary = Path(output.name)
                digest = hashlib.sha256()
                with urlopen(url, timeout=60) as response:
                    while block := response.read(1024 * 1024):
                        digest.update(block)
                        output.write(block)
                output.flush()
                os.fsync(output.fileno())
            if digest.hexdigest() != expected:
                raise ValueError(f"El checksum descargado no coincide: {relative}")
            os.link(temporary, target)
            print(f"Downloaded and verified: {target}")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/embedding.json"))
    parser.add_argument("--destination", type=Path, default=Path(".models/multilingual-e5-small"))
    args = parser.parse_args()
    download_model(args.config, args.destination)
