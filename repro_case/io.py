"""Small, explicit I/O layer for content-addressed inputs and records."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import shutil
import struct
import sys
import tempfile
import urllib.request
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any


def repository_root() -> Path:
    override = os.environ.get("SRC_CASE_ROOT")
    if override:
        return Path(override).resolve()
    start = Path.cwd().resolve()
    candidates = (start, *start.parents, *Path(__file__).resolve().parents)
    for candidate in candidates:
        if (candidate / "case-study" / "specification.json").is_file() and (
            candidate / "main.tex"
        ).is_file():
            return candidate
    raise RuntimeError("cannot locate the case-study repository root")


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_digest(root: Path) -> tuple[str, list[str]]:
    patterns = (
        "pyproject.toml",
        "Dockerfile",
        ".github/workflows/*.yml",
        "main.tex",
        "manuscript.sty",
        "references.bib",
        "case-study/*.json",
        "case-study/*.md",
        "figures/**/*.tex",
        "tables/**/*.tex",
        "repro_case/**/*.py",
    )
    paths = sorted({path for pattern in patterns for path in root.glob(pattern) if path.is_file()})
    digest = hashlib.sha256()
    names: list[str] = []
    for path in paths:
        name = path.relative_to(root).as_posix()
        names.append(name)
        digest.update(name.encode() + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest(), names


def environment_record() -> dict[str, str]:
    return {
        "operating_system": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "onnxruntime": version("onnxruntime"),
        "pillow": version("pillow"),
        "container_marker": os.environ.get("SRC_CASE_CONTAINER", "none"),
        "container_image": os.environ.get("SRC_CASE_IMAGE", "none"),
        "runner_os": os.environ.get("SRC_CASE_RUNNER_OS", "unreported"),
        "runner_architecture": os.environ.get("SRC_CASE_RUNNER_ARCH", "unreported"),
    }


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "src-qmnist-case/0.1"})
    with (
        urllib.request.urlopen(request, timeout=90) as response,
        destination.open("wb") as output,
    ):
        shutil.copyfileobj(response, output)


def _idx_header(path: Path) -> tuple[int, tuple[int, ...], int]:
    with gzip.open(path, "rb") as stream:
        prefix = stream.read(4)
        if len(prefix) != 4:
            raise ValueError(f"truncated IDX header: {path}")
        magic = struct.unpack(">I", prefix)[0]
        type_code = (magic >> 8) & 255
        rank = magic & 255
        if magic >> 16 or type_code not in {8, 12} or not rank:
            raise ValueError(f"unsupported IDX header: {path}")
        raw_shape = stream.read(4 * rank)
        if len(raw_shape) != 4 * rank:
            raise ValueError(f"truncated IDX shape: {path}")
        shape = struct.unpack(f">{rank}I", raw_shape)
        bytes_per_value = {8: 1, 12: 4}[type_code]
        expected_payload = bytes_per_value
        for dimension in shape:
            expected_payload *= dimension
        actual_payload = 0
        while block := stream.read(1024 * 1024):
            actual_payload += len(block)
    if actual_payload != expected_payload:
        raise ValueError(f"IDX payload length mismatch: {path}")
    return type_code, shape, actual_payload


def validate_input(path: Path, item: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing input: {path}")
    if path.stat().st_size != item["bytes"]:
        raise ValueError(f"wrong byte length for {path.name}")
    digest = sha256_file(path)
    if digest != item["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {path.name}")
    record = {
        "role": item["role"],
        "name": item["name"],
        "path": str(path),
        "url": item["url"],
        "bytes": item["bytes"],
        "sha256": digest,
        "format": item["format"],
    }
    if item["format"] == "idx-gzip":
        type_code, shape, _ = _idx_header(path)
        expected_type = {"ubyte": 8, "int32": 12}[item["idx_type"]]
        if type_code != expected_type or list(shape) != item["idx_shape"]:
            raise ValueError(f"IDX structure mismatch for {path.name}")
        record["idx_shape"] = list(shape)
    elif item["format"] == "onnx":
        if path.read_bytes()[:4] == b"vers":
            raise ValueError(f"unresolved Git LFS pointer: {path.name}")
    else:
        raise ValueError(f"unsupported input format: {item['format']}")
    if "source_revision" in item:
        record["source_revision"] = item["source_revision"]
    return record


def fetch_input(item: dict[str, Any], cache: Path) -> dict[str, Any]:
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / item["filename"]
    try:
        return validate_input(destination, item)
    except ValueError:
        destination.unlink(missing_ok=True)
    _download(item["url"], destination)
    return validate_input(destination, item)


def fetch_inputs(specification: dict[str, Any], cache: Path) -> list[dict[str, Any]]:
    return [fetch_input(item, cache) for item in specification["inputs"]["files"]]


def read_idx(path: Path):
    import numpy as np

    type_code, shape, _ = _idx_header(path)
    dtype = {8: "u1", 12: ">i4"}[type_code]
    with gzip.open(path, "rb") as stream:
        stream.read(4 + 4 * len(shape))
        payload = stream.read()
    return np.frombuffer(payload, dtype=dtype).reshape(shape)


def load_evaluation(specification: dict[str, Any], cache: Path) -> dict[str, Any]:
    import numpy as np

    files = {item["role"]: item for item in specification["inputs"]["files"]}
    mnist_images = read_idx(cache / files["mnist_images"]["filename"])
    mnist_labels = read_idx(cache / files["mnist_labels"]["filename"]).astype("u1")
    qmnist_images = read_idx(cache / files["qmnist_images"]["filename"])
    qmnist_metadata = read_idx(cache / files["qmnist_labels"]["filename"])
    if not np.array_equal(mnist_labels, qmnist_metadata[:10000, 0].astype("u1")):
        raise ValueError("paired MNIST/QMNIST rows have different labels")
    return {
        "mnist_images": mnist_images,
        "mnist_labels": mnist_labels,
        "qmnist_images": qmnist_images,
        "model": cache / files["model"]["filename"],
    }


def artifact_records(directory: Path, names: list[str]) -> list[dict[str, Any]]:
    records = []
    for name in names:
        path = directory / name
        if not path.is_file():
            raise ValueError(f"expected output is missing: {name}")
        records.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return records


def verify_artifacts(directory: Path, records: list[dict[str, Any]]) -> list[str]:
    problems = []
    for record in records:
        path = directory / record["path"]
        if not path.is_file():
            problems.append(f"missing output: {record['path']}")
        elif path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            problems.append(f"changed output: {record['path']}")
    return problems


def print_json(value: Any) -> None:
    sys.stdout.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
