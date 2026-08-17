"""Concrete binding, registration, verification, provenance, and inference."""

from __future__ import annotations

import os
import shutil
import subprocess
import traceback
from pathlib import Path
from typing import Any

from PIL import Image

from .analysis import FIGURE_OUTPUTS, run_analysis
from .crate import write_workflow_run_crate
from .io import (
    artifact_records,
    environment_record,
    fetch_inputs,
    load_evaluation,
    now,
    read_json,
    repository_root,
    sha256_file,
    source_digest,
    verify_artifacts,
    write_json,
)

ANALYSIS_OUTPUTS = [
    "predictions.csv",
    "metrics.json",
    "figure-data.json",
    *FIGURE_OUTPUTS,
]

ATTEMPT_RECORDS = [
    "artifacts.json",
    "assessment.json",
    "binding.json",
    "latex.log",
    "latexmk.log",
    "main.pdf",
    "provenance.json",
    "registration.json",
    "ro-crate-metadata.json",
    "workflow.json",
    "confusion.csv",
    "values.tex",
]


def load_specification(root: Path) -> tuple[dict[str, Any], Path]:
    path = root / "case-study" / "specification.json"
    specification = read_json(path)
    required = {
        "schema_version",
        "id",
        "inputs",
        "criterion",
        "provenance",
        "claims",
        "inference_rule",
    }
    if specification.get("schema_version") != "1.0" or not required.issubset(specification):
        raise ValueError("unsupported or incomplete case-study specification")
    if set(specification["claims"]) != {"container", "native"}:
        raise ValueError("specification must define exactly the container and native claims")
    return specification, path


def bind(root: Path, cache: Path, output: Path) -> dict[str, Any] | None:
    started = now()
    try:
        specification, specification_path = load_specification(root)
        inputs = fetch_inputs(specification, cache)
        binding = {
            "schema_version": "1.0",
            "complete": True,
            "started_at": started,
            "completed_at": now(),
            "specification": {
                "id": specification["id"],
                "path": specification_path.relative_to(root).as_posix(),
                "sha256": sha256_file(specification_path),
            },
            "external_inputs": inputs,
        }
        write_json(output / "binding.json", binding)
        return {"specification": specification, "binding": binding}
    except Exception as error:
        write_json(
            output / "binding.json",
            {
                "schema_version": "1.0",
                "complete": False,
                "started_at": started,
                "completed_at": now(),
                "error": {"type": type(error).__name__, "message": str(error)},
            },
        )
        return None


def _registered_context(claim: dict[str, Any], context: str) -> bool:
    return context in {item["id"] for item in claim["contexts"]}


def _context_contract(claim: dict[str, Any], context: str) -> dict[str, Any] | None:
    return next((item for item in claim["contexts"] if item["id"] == context), None)


def _compile_paper(root: Path, output: Path) -> str:
    stage = output / "paper-source"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    shutil.copyfile(root / "main.tex", stage / "main.tex")
    shutil.copyfile(root / "manuscript.sty", stage / "manuscript.sty")
    shutil.copyfile(root / "references.bib", stage / "references.bib")
    shutil.copytree(root / "figures", stage / "figures")
    shutil.copytree(root / "tables", stage / "tables")
    generated = stage / "build" / "case-study"
    generated.mkdir(parents=True, exist_ok=True)
    for name in FIGURE_OUTPUTS:
        shutil.copyfile(output / name, generated / name)
    latex_output = output / "latex"
    if latex_output.exists():
        shutil.rmtree(latex_output)
    latex_output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({"SOURCE_DATE_EPOCH": "946684800", "TZ": "UTC"})
    command = [
        "latexmk",
        "-pdf",
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-outdir={latex_output.resolve()}",
        "main.tex",
    ]
    result = subprocess.run(
        command,
        cwd=stage,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )
    (output / "latexmk.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"latexmk failed with exit status {result.returncode}; see latexmk.log")
    pdf = latex_output / "main.pdf"
    if not pdf.is_file() or not pdf.read_bytes().startswith(b"%PDF-"):
        raise RuntimeError("LaTeX did not produce a valid PDF header")
    final_log = (latex_output / "main.log").read_text(encoding="utf-8", errors="replace")
    unresolved = ("undefined references", "undefined citations", "! LaTeX Error")
    if any(marker in final_log for marker in unresolved):
        raise RuntimeError("final LaTeX pass contains unresolved references, citations, or errors")
    (output / "latex.log").write_text(final_log, encoding="utf-8")
    shutil.copyfile(pdf, output / "main.pdf")
    return "main.pdf"


def _semantic_output_problems(output: Path) -> list[str]:
    problems: list[str] = []
    figure_data = output / "figure-data.json"
    try:
        if figure_data.is_file():
            record = read_json(figure_data)
            expected = {"schema_version", "paired_reconstruction", "changed_pairs"}
            if set(record) != expected:
                problems.append("invalid semantic output: figure-data.json")
        values_path = output / "case-study-values.tex"
        if values_path.is_file():
            values = values_path.read_text(encoding="utf-8")
            required_macros = (
                "\\QMNISTChangedCount",
                "\\QMNISTAgreementPercent",
                "\\QMNISTPairOneRow",
                "\\QMNISTPairThreeTransition",
            )
            if not all(macro in values for macro in required_macros):
                problems.append("invalid semantic output: case-study-values.tex")
        for name in FIGURE_OUTPUTS:
            if not name.endswith(".png") or not (output / name).is_file():
                continue
            with Image.open(output / name) as image:
                image.load()
                if image.format != "PNG" or image.mode != "RGB" or image.size != (28, 28):
                    problems.append(f"invalid semantic output: {name}")
    except (OSError, ValueError):
        problems.append("unreadable semantic output: generated QMNIST figure data")
    return problems


def _assessment(
    mode: str,
    context: str,
    claim: dict[str, Any],
    provenance: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    reasons: list[str] = []
    environment = provenance["environment"]
    conformance = True
    if environment["python"] != "3.12.10":
        conformance = False
        reasons.append("the boundary requires Python 3.12.10")
    if mode == "container" and not provenance["paper_compilation_requested"]:
        conformance = False
        reasons.append("the container boundary requires manuscript compilation")
    if mode == "container" and environment["container_marker"] != "linux-amd64":
        conformance = False
        reasons.append("the container boundary requires the fixed linux/amd64 image")
    if mode == "native" and provenance["paper_compilation_requested"]:
        conformance = False
        reasons.append("the native boundary excludes manuscript compilation")
    if mode == "native" and environment["container_marker"] != "none":
        conformance = False
        reasons.append("the native boundary excludes container execution")
    context_contract = _context_contract(claim, context)
    if context_contract is not None:
        if environment["runner_os"] != context_contract["runner_os"]:
            conformance = False
            reasons.append("the observed runner operating system does not match registration")
        if environment["runner_architecture"] != context_contract["runner_architecture"]:
            conformance = False
            reasons.append("the observed runner architecture does not match registration")

    manifest_problems = verify_artifacts(output, provenance["outputs"])
    semantic_problems = _semantic_output_problems(output)
    eligible = not manifest_problems and not semantic_problems
    reasons.extend(manifest_problems)
    reasons.extend(semantic_problems)
    if provenance["termination"]["status"] not in {"completed", "failed"}:
        eligible = False
        reasons.append("termination was not recorded")

    if not conformance:
        classification = "inadmissible"
        decision: str | None = None
    elif not eligible:
        classification = "unassessable"
        decision = None
    else:
        classification = "admissible"
        expected = set(claim["exact_outputs"] + claim.get("semantic_outputs", []))
        observed = {item["path"] for item in provenance["outputs"]}
        criterion_met = False
        metrics_path = output / "metrics.json"
        if metrics_path.is_file():
            criterion_met = bool(read_json(metrics_path)["criterion"]["met"])
        reproduced = (
            provenance["termination"]["status"] == "completed"
            and expected.issubset(observed)
            and criterion_met
        )
        decision = "reproduced" if reproduced else "not_reproduced"
        if not reproduced and not reasons:
            reasons.append(
                "the execution failed, an output is absent, or the registered "
                "analysis rule was not met"
            )
    return {
        "schema_version": "1.0",
        "attempt_id": provenance["attempt_id"],
        "src_id": claim["src_id"],
        "mode": mode,
        "context": context,
        "registered_for_inference": _registered_context(claim, context),
        "environment": provenance["environment"],
        "boundary_conformance": "conformant" if conformance else "nonconformant",
        "evidence_eligibility": "eligible" if eligible else "unassessable",
        "classification": classification,
        "decision": decision,
        "reasons": reasons,
        "output_hashes": {item["path"]: item["sha256"] for item in provenance["outputs"]},
    }


def run_attempt(
    *,
    mode: str,
    context: str,
    output: Path,
    cache: Path | None = None,
    compile_paper: bool = False,
    root: Path | None = None,
) -> tuple[dict[str, Any], int]:
    root = (root or repository_root()).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ANALYSIS_OUTPUTS + ATTEMPT_RECORDS + ["case-study.png"]:
        (output / name).unlink(missing_ok=True)
    for directory in ("inputs", "latex", "paper-source"):
        shutil.rmtree(output / directory, ignore_errors=True)
    cache = Path(cache or root / "build" / "inputs").resolve()
    bound = bind(root, cache, output)
    if bound is None:
        return {"attempt": None, "binding": "incomplete", "path": str(output)}, 2

    specification = bound["specification"]
    claim = specification["claims"][mode]
    attempt_id = f"{mode}:{context}"
    registration = {
        "schema_version": "1.0",
        "attempt_id": attempt_id,
        "src_id": claim["src_id"],
        "context": context,
        "source": "direct execution",
        "selection": "registered inference context"
        if _registered_context(claim, context)
        else "attempt-level local context",
        "conforms_to_inference_sampling": _registered_context(claim, context),
        "registered_at": now(),
        "registered_before_decision": True,
    }
    write_json(output / "registration.json", registration)
    source_hash, source_files = source_digest(root)
    write_json(
        output / "workflow.json",
        {
            "schema_version": "1.0",
            "name": "src-case run",
            "entrypoint": "repro_case.protocol.run_attempt",
            "source_sha256": source_hash,
            "specification_id": specification["id"],
        },
    )
    started = now()
    output_names: list[str] = []
    termination: dict[str, Any]
    try:
        evaluation = load_evaluation(specification, cache)
        output_names = run_analysis(
            evaluation,
            output,
            specification["criterion"]["paired_prediction_agreement_minimum"],
        )
        if compile_paper:
            output_names.append(_compile_paper(root, output))
        termination = {"status": "completed", "exit_code": 0}
    except Exception as error:
        termination = {
            "status": "failed",
            "exit_code": 1,
            "error": {"type": type(error).__name__, "message": str(error)},
            "traceback": traceback.format_exc(),
        }
        output_names = [
            name for name in ANALYSIS_OUTPUTS + ["main.pdf"] if (output / name).is_file()
        ]

    records = artifact_records(output, output_names)
    write_json(
        output / "artifacts.json",
        {"schema_version": "1.0", "algorithm": "sha256", "files": records},
    )
    provenance = {
        "schema_version": "1.0",
        "attempt_id": attempt_id,
        "registration": "registration.json",
        "specification_id": specification["id"],
        "src_id": claim["src_id"],
        "mode": mode,
        "context": context,
        "started_at": started,
        "ended_at": now(),
        "source": {"sha256": source_hash, "files": source_files},
        "environment": environment_record(),
        "command": f"src-case run --mode {mode} --context {context}",
        "paper_compilation_requested": compile_paper,
        "termination": termination,
        "external_inputs": bound["binding"]["external_inputs"],
        "outputs": records,
        "derivations": [
            {
                "output": item["path"],
                "used": [
                    "MNIST test10k",
                    "QMNIST test rows 0:10000",
                    "ONNX Model Zoo MNIST-12",
                ],
            }
            for item in records
        ],
        "boundary": {
            "includes": claim["boundary_includes"],
            "excludes": claim["boundary_excludes"],
        },
    }
    write_json(output / "provenance.json", provenance)
    assessment = _assessment(mode, context, claim, provenance, output)
    write_json(output / "assessment.json", assessment)
    write_workflow_run_crate(output, root, provenance, assessment)
    exit_code = 0 if assessment["decision"] == "reproduced" else 3
    if assessment["classification"] in {"inadmissible", "unassessable"}:
        exit_code = 4
    return {"attempt": attempt_id, "assessment": assessment, "path": str(output)}, exit_code


def _infer_general_assessment(
    mode: str,
    claim: dict[str, Any],
    assessments: list[tuple[Path, dict[str, Any]]],
) -> dict[str, Any]:
    collected = []
    exact_hashes = {name: set() for name in claim["exact_outputs"]}
    has_failed_decision = False
    has_unresolved = False
    container_images: set[str] = set()
    for context_record in claim["contexts"]:
        context = context_record["id"]
        matches = [
            (path, item)
            for path, item in assessments
            if item.get("mode") == mode and item.get("context") == context
        ]
        if len(matches) != 1:
            classification = "missing" if not matches else "unassessable"
            collected.append(
                {"context": context, "classification": classification, "decision": None}
            )
            has_unresolved = True
            continue
        path, item = matches[0]
        collected.append(
            {
                "context": context,
                "classification": item["classification"],
                "decision": item["decision"],
                "assessment": str(path),
            }
        )
        if item["classification"] != "admissible":
            has_unresolved = True
        elif item["decision"] != "reproduced":
            has_failed_decision = True
        if mode == "container":
            container_images.add(item.get("environment", {}).get("container_image", "none"))
        for name in exact_hashes:
            if name in item.get("output_hashes", {}):
                exact_hashes[name].add(item["output_hashes"][name])
            else:
                has_failed_decision = True
    disagreements = {
        name: sorted(values) for name, values in exact_hashes.items() if len(values) > 1
    }
    incomplete_exact = [name for name, values in exact_hashes.items() if len(values) != 1]
    if mode == "container" and (len(container_images) != 1 or "none" in container_images):
        has_unresolved = True
    if has_failed_decision or disagreements:
        decision = "not_supported"
    elif has_unresolved or incomplete_exact:
        decision = "not_established"
    else:
        decision = "supported"
    return {
        "src_id": claim["src_id"],
        "decision": decision,
        "attempts": collected,
        "exact_output_digests": {name: sorted(values) for name, values in exact_hashes.items()},
        "disagreements": disagreements,
        "missing_exact_outputs": incomplete_exact,
        "container_images": sorted(container_images) if mode == "container" else [],
    }


def infer_general_assessments(
    attempt_root: Path, output: Path, root: Path | None = None
) -> tuple[dict[str, Any], int]:
    root = (root or repository_root()).resolve()
    specification, specification_path = load_specification(root)
    assessments: list[tuple[Path, dict[str, Any]]] = []
    for path in Path(attempt_root).rglob("assessment.json"):
        try:
            assessments.append((path, read_json(path)))
        except (OSError, ValueError):
            continue
    general_assessments = {
        mode: _infer_general_assessment(mode, claim, assessments)
        for mode, claim in specification["claims"].items()
    }
    record = {
        "schema_version": "1.0",
        "specification_id": specification["id"],
        "created_at": now(),
        "specification_sha256": sha256_file(specification_path),
        "rule": specification["inference_rule"],
        "general_assessments": general_assessments,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "general-assessments.json", record)
    lines = ["# QMNIST SRC general report", ""]
    for mode, claim in general_assessments.items():
        lines.extend([f"- `{mode}` / `{claim['src_id']}`: **{claim['decision']}**"])
    (output / "general-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    decisions = {claim["decision"] for claim in general_assessments.values()}
    if decisions == {"supported"}:
        exit_code = 0
    elif "not_supported" in decisions:
        exit_code = 3
    else:
        exit_code = 4
    return record, exit_code
