"""Fixed-model paired inference and deterministic rendering."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageOps

from .io import write_json

FIGURE_OUTPUTS = [
    "case-study-values.tex",
    *[
        f"case-study-{kind}-{slot}.png"
        for slot in range(1, 4)
        for kind in ("mnist", "difference", "qmnist")
    ],
]


def load_model(path: Path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    model_input = session.get_inputs()
    if len(model_input) != 1 or model_input[0].shape != [1, 1, 28, 28]:
        raise ValueError("unexpected ONNX model input contract")
    if len(session.get_outputs()) != 1:
        raise ValueError("unexpected ONNX model output contract")
    return session


def predict(session, images):
    import numpy as np

    values = np.asarray(images, dtype=np.float32).reshape(-1, 1, 28, 28) / np.float32(255)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    predictions = np.empty(len(values), dtype=np.uint8)
    for index, image in enumerate(values):
        logits = session.run([output_name], {input_name: image[None, ...]})[0]
        predictions[index] = np.argmax(logits[0])
    return predictions


def _ratio(numerator: int, denominator: int) -> str:
    return f"{numerator / denominator:.6f}" if denominator else "0.000000"


def compute_analysis(
    evaluation: dict[str, Any],
    mnist_predictions,
    qmnist_predictions,
    paired_agreement_minimum: dict[str, int],
):
    import numpy as np

    mnist_labels = np.asarray(evaluation["mnist_labels"], dtype=np.uint8)
    mnist_predictions = np.asarray(mnist_predictions, dtype=np.uint8)
    qmnist_predictions = np.asarray(qmnist_predictions, dtype=np.uint8)
    if len(mnist_predictions) != 10000 or len(qmnist_predictions) != 10000:
        raise ValueError("prediction count does not match the 10,000 registered pairs")

    threshold_numerator = int(paired_agreement_minimum["numerator"])
    threshold_denominator = int(paired_agreement_minimum["denominator"])
    if not 0 <= threshold_numerator <= threshold_denominator or threshold_denominator <= 0:
        raise ValueError("paired agreement threshold must be a fraction between zero and one")
    paired_total = 10000
    minimum_agreements = (
        paired_total * threshold_numerator + threshold_denominator - 1
    ) // threshold_denominator
    maximum_disagreements = paired_total - minimum_agreements
    agreements = int((mnist_predictions == qmnist_predictions).sum())
    paired = {
        "agreements": agreements,
        "disagreements": paired_total - agreements,
        "total": paired_total,
        "agreement": _ratio(agreements, paired_total),
        "agreement_threshold": _ratio(threshold_numerator, threshold_denominator),
        "maximum_disagreements": maximum_disagreements,
        "mnist_correct_qmnist_wrong": int(
            ((mnist_predictions == mnist_labels) & (qmnist_predictions != mnist_labels)).sum()
        ),
        "mnist_wrong_qmnist_correct": int(
            ((mnist_predictions != mnist_labels) & (qmnist_predictions == mnist_labels)).sum()
        ),
        "both_wrong_different": int(
            (
                (mnist_predictions != qmnist_predictions)
                & (mnist_predictions != mnist_labels)
                & (qmnist_predictions != mnist_labels)
            ).sum()
        ),
    }
    metrics = {
        "schema_version": "1.0",
        "model": "ONNX Model Zoo MNIST-12 CNN",
        "paired_reconstruction": paired,
        "criterion": {
            "rule": "paired_prediction_agreement >= registered minimum",
            "met": agreements >= minimum_agreements,
        },
    }

    changed = np.flatnonzero(mnist_predictions != qmnist_predictions)
    changed_pairs = [
        {
            "row": int(index),
            "true": int(mnist_labels[index]),
            "mnist_prediction": int(mnist_predictions[index]),
            "qmnist_prediction": int(qmnist_predictions[index]),
        }
        for index in changed
    ]
    figure_data = {
        "schema_version": "1.0",
        "paired_reconstruction": paired,
        "changed_pairs": changed_pairs,
    }
    return metrics, figure_data


def render_figure_assets(
    destination: Path, figure_data: dict[str, Any], mnist_images, qmnist_images
) -> list[str]:
    """Write only the raster data tiles used by the native TikZ figure."""
    destination.mkdir(parents=True, exist_ok=True)
    changed_pairs = figure_data["changed_pairs"]
    displayed_pairs = changed_pairs[:3]
    suffixes = ("One", "Two", "Three")
    values: list[str] = ["% Generated by repro_case.analysis; do not edit."]
    paired = figure_data["paired_reconstruction"]
    values.extend(
        [
            rf"\def\QMNISTChangedCount{{{int(paired['disagreements'])}}}",
            rf"\def\QMNISTAgreementPercent{{{100 * float(paired['agreement']):.2f}}}",
            rf"\def\QMNISTChangeLimit{{{int(paired['maximum_disagreements'])}}}",
            rf"\def\QMNISTDisplayedCount{{{len(displayed_pairs)}}}",
        ]
    )

    q_brighter_colour = "#0072b2"
    q_darker_colour = "#d55e00"
    blank = Image.new("RGB", (28, 28), "white")
    for slot, suffix in enumerate(suffixes, start=1):
        item = displayed_pairs[slot - 1] if slot <= len(displayed_pairs) else None
        if item is None:
            row = truth = mnist_prediction = qmnist_prediction = 0
            visible = footnote = 0
            transition = ""
            mnist = qmnist = difference = blank
        else:
            row = int(item["row"])
            truth = int(item["true"])
            mnist_prediction = int(item["mnist_prediction"])
            qmnist_prediction = int(item["qmnist_prediction"])
            visible = 1
            mnist_source = Image.fromarray(mnist_images[row].astype("uint8"))
            qmnist_source = Image.fromarray(qmnist_images[row].astype("uint8"))
            q_brighter = ImageChops.subtract(qmnist_source, mnist_source)
            q_darker = ImageChops.subtract(mnist_source, qmnist_source)
            difference_peak = max(q_brighter.getextrema()[1], q_darker.getextrema()[1])
            footnote = int(difference_peak == 1)
            if difference_peak:
                q_brighter = q_brighter.point(
                    lambda value, peak=difference_peak: round(value * 255 / peak)
                )
                q_darker = q_darker.point(
                    lambda value, peak=difference_peak: round(value * 255 / peak)
                )
            difference = Image.new("RGB", mnist_source.size, "white")
            difference = Image.composite(
                Image.new("RGB", mnist_source.size, q_brighter_colour),
                difference,
                q_brighter,
            )
            difference = Image.composite(
                Image.new("RGB", mnist_source.size, q_darker_colour),
                difference,
                q_darker,
            )
            mnist = ImageOps.invert(mnist_source).convert("RGB")
            qmnist = ImageOps.invert(qmnist_source).convert("RGB")
            if mnist_prediction == truth and qmnist_prediction != truth:
                transition = "correct to wrong"
            elif mnist_prediction != truth and qmnist_prediction == truth:
                transition = "wrong to correct"
            else:
                transition = "wrong to wrong"

        for kind, image in (
            ("mnist", mnist),
            ("difference", difference),
            ("qmnist", qmnist),
        ):
            image.save(
                destination / f"case-study-{kind}-{slot}.png",
                format="PNG",
                optimize=False,
                compress_level=9,
            )
        values.extend(
            [
                rf"\def\QMNISTPair{suffix}Visible{{{visible}}}",
                rf"\def\QMNISTPair{suffix}Row{{{row}}}",
                rf"\def\QMNISTPair{suffix}Truth{{{truth}}}",
                rf"\def\QMNISTPair{suffix}MNISTPrediction{{{mnist_prediction}}}",
                rf"\def\QMNISTPair{suffix}QMNISTPrediction{{{qmnist_prediction}}}",
                rf"\def\QMNISTPair{suffix}Transition{{{transition}}}",
                rf"\def\QMNISTPair{suffix}Footnote{{{footnote}}}",
            ]
        )

    (destination / "case-study-values.tex").write_text(
        "\n".join(values) + "\n", encoding="utf-8", newline="\n"
    )
    return FIGURE_OUTPUTS


def _write_csv(path: Path, rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)


def run_analysis(
    evaluation: dict[str, Any], output: Path, paired_agreement_minimum: dict[str, int]
) -> list[str]:
    session = load_model(evaluation["model"])
    mnist_predictions = predict(session, evaluation["mnist_images"])
    qmnist_images = evaluation["qmnist_images"][:10000]
    qmnist_predictions = predict(session, qmnist_images)
    metrics, figure_data = compute_analysis(
        evaluation, mnist_predictions, qmnist_predictions, paired_agreement_minimum
    )
    write_json(output / "metrics.json", metrics)
    write_json(output / "figure-data.json", figure_data)

    prediction_rows: list[list[Any]] = [
        [
            "row",
            "true",
            "mnist_prediction",
            "qmnist_prediction",
            "same_prediction",
            "mnist_correct",
            "qmnist_correct",
        ]
    ]
    for row, (truth, mnist_prediction, qmnist_prediction) in enumerate(
        zip(
            evaluation["mnist_labels"],
            mnist_predictions,
            qmnist_predictions,
            strict=True,
        )
    ):
        prediction_rows.append(
            [
                row,
                int(truth),
                int(mnist_prediction),
                int(qmnist_prediction),
                int(mnist_prediction == qmnist_prediction),
                int(mnist_prediction == truth),
                int(qmnist_prediction == truth),
            ]
        )
    _write_csv(output / "predictions.csv", prediction_rows)
    figure_outputs = render_figure_assets(
        output, figure_data, evaluation["mnist_images"], qmnist_images
    )
    return [
        "predictions.csv",
        "metrics.json",
        "figure-data.json",
        *figure_outputs,
    ]
