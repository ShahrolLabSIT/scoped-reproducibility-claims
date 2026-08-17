"""Build the four immutable-release assets from supported general assessments."""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from rocrate.model.contextentity import ContextEntity
from rocrate.rocrate import ROCrate

from .io import fetch_input, read_json, sha256_file, verify_artifacts, write_json
from .protocol import load_specification

RELEASE_ASSETS = ("main.pdf", "mnist-12.onnx", "reproduction.crate.zip", "SHA256SUMS")


def _expected_attempts(specification: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (mode, context["id"]): context
        for mode, claim in specification["claims"].items()
        for context in claim["contexts"]
    }


def _collect_attempts(
    attempt_root: Path, specification: dict[str, Any], image_reference: str
) -> list[tuple[Path, dict[str, Any]]]:
    expected = _expected_attempts(specification)
    found: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    for assessment_path in Path(attempt_root).rglob("assessment.json"):
        assessment = read_json(assessment_path)
        key = (assessment.get("mode", ""), assessment.get("context", ""))
        if key not in expected:
            continue
        if key in found:
            raise ValueError(f"duplicate registered attempt: {key[0]}:{key[1]}")
        directory = assessment_path.parent
        provenance = read_json(directory / "provenance.json")
        if verify_artifacts(directory, provenance["outputs"]):
            raise ValueError(f"attempt outputs changed after assessment: {key[0]}:{key[1]}")
        if not (directory / "ro-crate-metadata.json").is_file():
            raise ValueError(f"attempt is not an RO-Crate: {key[0]}:{key[1]}")
        if assessment.get("classification") != "admissible" or assessment.get("decision") != (
            "reproduced"
        ):
            raise ValueError(f"attempt is not reproduced: {key[0]}:{key[1]}")
        if (
            key[0] == "container"
            and assessment.get("environment", {}).get("container_image") != image_reference
        ):
            raise ValueError(f"container image does not match the release digest: {key[1]}")
        found[key] = (directory, assessment)
    missing = sorted(set(expected) - set(found))
    if missing:
        labels = ", ".join(f"{mode}:{context}" for mode, context in missing)
        raise ValueError(f"missing registered attempts: {labels}")
    return [found[key] for key in sorted(found)]


def _copy_attempt(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    for path in sorted(source.iterdir()):
        if path.is_file() and not path.is_symlink():
            shutil.copyfile(path, destination / path.name)


def _add_tree(crate: ROCrate, root: Path) -> None:
    media_types = {
        ".csv": "text/csv",
        ".json": "application/json",
        ".md": "text/markdown",
        ".onnx": "application/octet-stream",
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".tex": "application/x-tex",
        ".txt": "text/plain",
    }
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        crate.add_file(
            path,
            relative,
            properties={
                "name": path.name,
                "contentSize": str(path.stat().st_size),
                "encodingFormat": media_types.get(path.suffix, "application/octet-stream"),
            },
        )


def package_release(
    *,
    attempt_root: Path,
    general_assessment: Path,
    cache: Path,
    output: Path,
    commit: str,
    tag: str,
    repository: str,
    image: str,
    image_digest: str,
    root: Path,
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("commit must be a full lowercase Git SHA")
    if tag != f"run-{commit}":
        raise ValueError("release tag must be run-<full commit SHA>")
    if not re.fullmatch(r"[^/\s]+/[^/\s]+", repository):
        raise ValueError("repository must have the form owner/name")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_digest):
        raise ValueError("image digest must be sha256:<64 lowercase hex characters>")

    specification, specification_path = load_specification(root)
    assessment_path = Path(general_assessment) / "general-assessments.json"
    assessment_record = read_json(assessment_path)
    if assessment_record.get("specification_sha256") != sha256_file(specification_path):
        raise ValueError("general assessment and release use different specifications")
    assessments = assessment_record.get("general_assessments", {})
    if {mode: item.get("decision") for mode, item in assessments.items()} != {
        "container": "supported",
        "native": "supported",
    }:
        raise ValueError("only supported general assessments can be released")

    image_reference = f"{image}@{image_digest}"
    attempts = _collect_attempts(attempt_root, specification, image_reference)
    pdf_digests = assessments["container"]["exact_output_digests"].get("main.pdf")
    if not isinstance(pdf_digests, list) or len(pdf_digests) != 1:
        raise ValueError("the container general assessment did not establish one manuscript digest")
    canonical_pdf = next(
        directory / "main.pdf"
        for directory, assessment in attempts
        if assessment["mode"] == "container"
    )
    if sha256_file(canonical_pdf) != pdf_digests[0]:
        raise ValueError("canonical manuscript does not match the general assessment")

    model = next(item for item in specification["inputs"]["files"] if item["role"] == "model")
    model_record = fetch_input(model, Path(cache))
    source_records = [item for item in specification["inputs"]["files"] if item["role"] != "model"]

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in RELEASE_ASSETS:
        (output / name).unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="release-crate-", dir=output.parent) as temporary:
        stage = Path(temporary)
        (stage / "inputs").mkdir()
        (stage / "general-assessment").mkdir()
        (stage / "attempts").mkdir()
        (stage / "publication").mkdir()
        shutil.copyfile(specification_path, stage / "specification.json")
        shutil.copyfile(Path(cache) / model["filename"], stage / "inputs" / model["filename"])
        write_json(stage / "inputs" / "sources.json", {"files": source_records})
        shutil.copyfile(assessment_path, stage / "general-assessment" / "general-assessments.json")
        shutil.copyfile(
            Path(general_assessment) / "general-report.md",
            stage / "general-assessment" / "general-report.md",
        )
        shutil.copyfile(canonical_pdf, stage / "publication" / "main.pdf")

        attempt_records = []
        for directory, assessment in attempts:
            context = assessment["context"]
            target = stage / "attempts" / context
            _copy_attempt(directory, target)
            attempt_records.append(
                {
                    "mode": assessment["mode"],
                    "context": context,
                    "assessment_sha256": sha256_file(target / "assessment.json"),
                    "ro_crate_metadata_sha256": sha256_file(target / "ro-crate-metadata.json"),
                }
            )

        manifest = {
            "schema_version": "1.0",
            "release": {
                "repository": repository,
                "commit": commit,
                "tag": tag,
                "source": f"https://github.com/{repository}/tree/{commit}",
            },
            "container": {
                "image": image,
                "digest": image_digest,
                "reference": image_reference,
            },
            "model": {
                key: model_record[key]
                for key in ("name", "url", "bytes", "sha256", "format", "source_revision")
            }
            | {"path": f"inputs/{model['filename']}"},
            "data_inputs": source_records,
            "general_assessments": {
                mode: assessments[mode]["decision"] for mode in ("container", "native")
            },
            "attempts": attempt_records,
            "assets": {
                "manuscript": "main.pdf",
                "model": model["filename"],
                "evidence": "reproduction.crate.zip",
                "checksums": "SHA256SUMS",
            },
        }
        write_json(stage / "release-manifest.json", manifest)
        (stage / "README.md").write_text(
            "# General reproducibility assessment\n\n"
            f"Evidence for `{repository}@{commit}` and `{image_reference}`.\n\n"
            "Each directory below `attempts/` is a Workflow Run RO-Crate. "
            "The outer archive is an RO-Crate that collects those attempts, the "
            "general assessment, the fixed model, and the canonical manuscript. "
            "MNIST and QMNIST remain external content-addressed inputs recorded in "
            "`inputs/sources.json`.\n",
            encoding="utf-8",
        )

        crate = ROCrate(version="1.1")
        _add_tree(crate, stage)
        source = crate.add(
            ContextEntity(
                crate,
                f"https://github.com/{repository}/tree/{commit}",
                properties={
                    "@type": "SoftwareSourceCode",
                    "name": repository,
                    "identifier": commit,
                    "codeRepository": f"https://github.com/{repository}",
                },
            )
        )
        container = crate.add(
            ContextEntity(
                crate,
                f"docker://{image_reference}",
                properties={
                    "@type": "SoftwareApplication",
                    "name": image,
                    "identifier": image_digest,
                },
            )
        )
        crate.root_dataset["name"] = f"General reproducibility assessment {tag}"
        crate.root_dataset["description"] = (
            "Six registered Workflow Run RO-Crates and their general assessment"
        )
        crate.root_dataset["datePublished"] = assessment_record["created_at"]
        crate.root_dataset["license"] = {"@id": "https://spdx.org/licenses/Apache-2.0"}
        crate.root_dataset["isBasedOn"] = source
        crate.root_dataset["mentions"] = container
        crate.write_zip(output / "reproduction.crate.zip")

    shutil.copyfile(canonical_pdf, output / "main.pdf")
    shutil.copyfile(Path(cache) / model["filename"], output / model["filename"])
    checksums = [
        f"{sha256_file(output / name)}  {name}"
        for name in ("main.pdf", model["filename"], "reproduction.crate.zip")
    ]
    (output / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="ascii")
    return manifest
