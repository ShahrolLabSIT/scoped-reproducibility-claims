from __future__ import annotations

import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path

from rocrate.rocrate import ROCrate

from repro_case.analysis import FIGURE_OUTPUTS, compute_analysis, predict, render_figure_assets
from repro_case.crate import write_workflow_run_crate
from repro_case.io import read_json, sha256_file, verify_artifacts, write_json
from repro_case.protocol import infer_general_assessments, load_specification
from repro_case.release import RELEASE_ASSETS, package_release

ROOT = Path(__file__).resolve().parents[1]


class AnalysisTests(unittest.TestCase):
    def test_specification_binds_the_published_model_by_digest(self) -> None:
        specification, _ = load_specification(ROOT)
        model = next(item for item in specification["inputs"]["files"] if item["role"] == "model")
        self.assertEqual(model["format"], "onnx")
        self.assertEqual(model["source_revision"], "4c46cd00fbdb7cd30b6c1c17ab54f2e1f4f7b177")
        self.assertEqual(
            model["sha256"],
            "5c688690f8bacf667d4c2074af5ad0646ca328d7ab03eccf944a65b320171bdd",
        )
        self.assertEqual(
            specification["criterion"]["paired_prediction_agreement_minimum"],
            {"numerator": 99, "denominator": 100},
        )

    def test_registered_paired_threshold_is_exact(self) -> None:
        import numpy as np

        evaluation = {"mnist_labels": np.zeros(10000, dtype=np.uint8)}
        mnist_predictions = np.zeros(10000, dtype=np.uint8)
        qmnist_predictions = mnist_predictions.copy()
        qmnist_predictions[:100] = 1
        metrics, _ = compute_analysis(
            evaluation,
            mnist_predictions,
            qmnist_predictions,
            {"numerator": 99, "denominator": 100},
        )
        self.assertTrue(metrics["criterion"]["met"])
        self.assertEqual(metrics["paired_reconstruction"]["maximum_disagreements"], 100)

        qmnist_predictions[100] = 1
        metrics, _ = compute_analysis(
            evaluation,
            mnist_predictions,
            qmnist_predictions,
            {"numerator": 99, "denominator": 100},
        )
        self.assertFalse(metrics["criterion"]["met"])

    def test_predict_applies_the_registered_image_scaling(self) -> None:
        import numpy as np

        class Tensor:
            name = "input"
            shape = [1, 1, 28, 28]

        class Session:
            def get_inputs(self):
                return [Tensor()]

            def get_outputs(self):
                output = Tensor()
                output.name = "output"
                return [output]

            def run(self, _outputs, feed):
                value = float(feed["input"].max())
                logits = np.zeros((1, 10), dtype=np.float32)
                logits[0, 7 if value == 1.0 else 3] = 1
                return [logits]

        images = np.stack(
            [np.zeros((28, 28), dtype=np.uint8), np.full((28, 28), 255, dtype=np.uint8)]
        )
        np.testing.assert_array_equal(predict(Session(), images), np.array([3, 7]))

    def test_figure_encoding_is_repeatable(self) -> None:
        import numpy as np

        images = np.stack([np.full((28, 28), index * 80, dtype=np.uint8) for index in range(3)])
        figure_data = {
            "paired_reconstruction": {
                "agreement": "0.970000",
                "agreements": 97,
                "disagreements": 3,
                "total": 100,
                "agreement_threshold": "0.900000",
                "maximum_disagreements": 10,
            },
            "changed_pairs": [
                {
                    "row": index,
                    "true": index,
                    "mnist_prediction": (index + 1) % 10,
                    "qmnist_prediction": index,
                }
                for index in range(3)
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            render_figure_assets(first, figure_data, images, images)
            render_figure_assets(second, figure_data, images, images)
            for name in FIGURE_OUTPUTS:
                self.assertEqual(
                    hashlib.sha256((first / name).read_bytes()).digest(),
                    hashlib.sha256((second / name).read_bytes()).digest(),
                )


class RecordTests(unittest.TestCase):
    def test_rocrate_library_builds_a_workflow_run_crate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "attempt"
            output.mkdir()
            write_json(output / "workflow.json", {"entrypoint": "example.run"})
            (output / "result.txt").write_text("result", encoding="utf-8")
            provenance = {
                "attempt_id": "native:test",
                "started_at": "2026-01-01T00:00:00Z",
                "ended_at": "2026-01-01T00:00:01Z",
                "environment": {"python": "3.12.10"},
                "termination": {"status": "completed"},
                "external_inputs": [
                    {
                        "role": "model",
                        "name": "Published model",
                        "url": "https://example.test/images.gz",
                        "bytes": 1,
                        "sha256": "b" * 64,
                    },
                ],
                "outputs": [{"path": "result.txt"}],
            }
            write_workflow_run_crate(
                output,
                root,
                provenance,
                {"decision": "reproduced", "classification": "admissible"},
            )

            crate = ROCrate(output)
            self.assertEqual(crate.version, "1.1")
            self.assertIn("ComputationalWorkflow", crate.mainEntity.type)
            self.assertEqual(crate.mainEntity.id, "workflow.json")
            self.assertEqual(len(crate.get_by_type("CreateAction")), 1)

    def test_manifest_detects_changed_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "result.txt"
            path.write_text("first", encoding="utf-8")
            record = [
                {"path": "result.txt", "bytes": 5, "sha256": hashlib.sha256(b"first").hexdigest()}
            ]
            self.assertEqual(verify_artifacts(directory, record), [])
            path.write_text("later", encoding="utf-8")
            self.assertEqual(verify_artifacts(directory, record), ["changed output: result.txt"])

    def test_inference_retains_missing_registered_context(self) -> None:
        specification, _ = load_specification(ROOT)
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            attempt_root = base / "attempts"
            for mode, claim in specification["claims"].items():
                for context in claim["contexts"]:
                    directory = attempt_root / context["id"]
                    hashes = {name: "a" * 64 for name in claim["exact_outputs"]}
                    write_json(
                        directory / "assessment.json",
                        {
                            "mode": mode,
                            "context": context["id"],
                            "classification": "admissible",
                            "decision": "reproduced",
                            "environment": {
                                "container_image": "ghcr.io/example/case@sha256:" + "b" * 64
                                if mode == "container"
                                else "none"
                            },
                            "output_hashes": hashes,
                        },
                    )
            record, exit_code = infer_general_assessments(attempt_root, base / "complete", ROOT)
            self.assertEqual(exit_code, 0)
            self.assertEqual(
                {item["decision"] for item in record["general_assessments"].values()},
                {"supported"},
            )

            missing = (
                attempt_root
                / specification["claims"]["native"]["contexts"][0]["id"]
                / "assessment.json"
            )
            missing.unlink()
            record, exit_code = infer_general_assessments(attempt_root, base / "missing", ROOT)
            self.assertEqual(exit_code, 4)
            self.assertEqual(record["general_assessments"]["native"]["decision"], "not_established")
            rows = record["general_assessments"]["native"]["attempts"]
            self.assertIn("missing", {row["classification"] for row in rows})
            self.assertEqual(
                read_json(base / "missing" / "general-assessments.json")["schema_version"],
                "1.0",
            )


class ReleaseTests(unittest.TestCase):
    COMMIT = "a" * 40
    DIGEST = "sha256:" + "b" * 64
    IMAGE = "ghcr.io/example/project/case-study"

    def _general_assessment(self, base: Path, *, supported: bool = True) -> dict[str, Path]:
        root = base / "repository"
        case_study = root / "case-study"
        case_study.mkdir(parents=True)
        model_payload = b"small onnx model"
        model_digest = hashlib.sha256(model_payload).hexdigest()
        contexts = {
            "container": [f"container-{index}" for index in range(3)],
            "native": [f"native-{index}" for index in range(3)],
        }
        specification = {
            "schema_version": "1.0",
            "id": "test-general-assessment",
            "inputs": {
                "files": [
                    {
                        "role": "data",
                        "name": "External data",
                        "filename": "data.gz",
                        "url": "https://example.test/data.gz",
                        "sha256": "c" * 64,
                        "bytes": 10,
                        "format": "idx-gzip",
                    },
                    {
                        "role": "model",
                        "name": "Fixed model",
                        "filename": "mnist-12.onnx",
                        "url": "https://example.test/mnist-12.onnx",
                        "source_revision": "model-revision",
                        "sha256": model_digest,
                        "bytes": len(model_payload),
                        "format": "onnx",
                    },
                ]
            },
            "criterion": {},
            "provenance": {},
            "claims": {
                mode: {
                    "src_id": f"SRC-{mode}",
                    "contexts": [{"id": context} for context in mode_contexts],
                    "exact_outputs": ["main.pdf"] if mode == "container" else ["result.txt"],
                }
                for mode, mode_contexts in contexts.items()
            },
            "inference_rule": {},
        }
        specification_path = case_study / "specification.json"
        write_json(specification_path, specification)
        cache = base / "cache"
        cache.mkdir()
        (cache / "mnist-12.onnx").write_bytes(model_payload)

        attempts = base / "attempts"
        image_reference = f"{self.IMAGE}@{self.DIGEST}"
        pdf_payload = b"%PDF-1.4\ntest\n"
        pdf_digest = hashlib.sha256(pdf_payload).hexdigest()
        for mode, mode_contexts in contexts.items():
            for context in mode_contexts:
                directory = attempts / f"attempt-{context}"
                directory.mkdir(parents=True)
                output_name = "main.pdf" if mode == "container" else "result.txt"
                output_payload = pdf_payload if mode == "container" else b"result"
                (directory / output_name).write_bytes(output_payload)
                output_record = {
                    "path": output_name,
                    "bytes": len(output_payload),
                    "sha256": hashlib.sha256(output_payload).hexdigest(),
                }
                write_json(directory / "provenance.json", {"outputs": [output_record]})
                write_json(
                    directory / "assessment.json",
                    {
                        "mode": mode,
                        "context": context,
                        "classification": "admissible",
                        "decision": "reproduced",
                        "environment": {
                            "container_image": image_reference if mode == "container" else "none"
                        },
                        "output_hashes": {output_name: output_record["sha256"]},
                    },
                )
                write_json(directory / "ro-crate-metadata.json", {"@graph": []})

        general_assessment = base / "general-assessment"
        general_assessment.mkdir()
        decision = "supported" if supported else "not_established"
        write_json(
            general_assessment / "general-assessments.json",
            {
                "schema_version": "1.0",
                "created_at": "2026-01-01T00:00:00Z",
                "specification_sha256": sha256_file(specification_path),
                "general_assessments": {
                    "container": {
                        "decision": decision,
                        "exact_output_digests": {"main.pdf": [pdf_digest]},
                    },
                    "native": {
                        "decision": decision,
                        "exact_output_digests": {
                            "result.txt": [hashlib.sha256(b"result").hexdigest()]
                        },
                    },
                },
            },
        )
        (general_assessment / "general-report.md").write_text(
            "# General report\n", encoding="utf-8"
        )
        return {
            "root": root,
            "attempts": attempts,
            "general_assessment": general_assessment,
            "cache": cache,
        }

    def test_package_release_builds_exactly_four_assets_and_nested_crates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._general_assessment(Path(temporary))
            output = Path(temporary) / "release"
            package_release(
                attempt_root=Path(paths["attempts"]),
                general_assessment=Path(paths["general_assessment"]),
                cache=Path(paths["cache"]),
                output=output,
                commit=self.COMMIT,
                tag=f"run-{self.COMMIT}",
                repository="example/project",
                image=self.IMAGE,
                image_digest=self.DIGEST,
                root=Path(paths["root"]),
            )
            self.assertEqual({path.name for path in output.iterdir()}, set(RELEASE_ASSETS))
            crate = ROCrate(output / "reproduction.crate.zip")
            self.assertEqual(crate.version, "1.1")
            self.assertEqual(
                crate.root_dataset["isBasedOn"]["@id"],
                f"https://github.com/example/project/tree/{self.COMMIT}",
            )
            with zipfile.ZipFile(output / "reproduction.crate.zip") as archive:
                names = set(archive.namelist())
            self.assertIn("ro-crate-metadata.json", names)
            self.assertIn("release-manifest.json", names)
            self.assertIn("inputs/mnist-12.onnx", names)
            self.assertIn("inputs/sources.json", names)
            self.assertIn("general-assessment/general-assessments.json", names)
            self.assertIn("general-assessment/general-report.md", names)
            self.assertIn("publication/main.pdf", names)
            for index in range(3):
                self.assertIn(f"attempts/container-{index}/ro-crate-metadata.json", names)
                self.assertIn(f"attempts/native-{index}/ro-crate-metadata.json", names)

            checksum_lines = (output / "SHA256SUMS").read_text(encoding="ascii").splitlines()
            self.assertEqual(len(checksum_lines), 3)
            for line in checksum_lines:
                digest, filename = line.split("  ")
                self.assertEqual(digest, sha256_file(output / filename))

    def test_package_release_rejects_unsupported_general_assessments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._general_assessment(Path(temporary), supported=False)
            with self.assertRaisesRegex(ValueError, "supported general assessments"):
                package_release(
                    attempt_root=Path(paths["attempts"]),
                    general_assessment=Path(paths["general_assessment"]),
                    cache=Path(paths["cache"]),
                    output=Path(temporary) / "release",
                    commit=self.COMMIT,
                    tag=f"run-{self.COMMIT}",
                    repository="example/project",
                    image=self.IMAGE,
                    image_digest=self.DIGEST,
                    root=Path(paths["root"]),
                )


if __name__ == "__main__":
    unittest.main()
