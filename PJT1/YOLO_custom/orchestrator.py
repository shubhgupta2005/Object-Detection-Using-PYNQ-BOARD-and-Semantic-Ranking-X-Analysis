#!/usr/bin/env python
"""
Full pipeline orchestrator.

Takes an image path and text-based task input, then runs:
  Model-1 Tiny YOLOv2 detection -> Model-2 semantic task-aware ranking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_model2_with_image_output import main as run_yolo_and_model2


def _normalize_cli_input(value: str) -> str:
    """Trim spaces and remove one layer of wrapping quotes."""
    cleaned = (value or "").strip()
    if len(cleaned) >= 2 and (
        (cleaned[0] == '"' and cleaned[-1] == '"')
        or (cleaned[0] == "'" and cleaned[-1] == "'")
    ):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _normalize_label(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


def _compute_accuracy(y_true: list[str], y_pred: list[str]) -> float:
    if not y_true:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if _normalize_label(t) == _normalize_label(p))
    return correct / len(y_true)


def _compute_confusion_matrix(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    labels = sorted({_normalize_label(v) for v in y_true + y_pred})
    index = {label: i for i, label in enumerate(labels)}

    matrix = [[0 for _ in labels] for _ in labels]
    for t, p in zip(y_true, y_pred):
        ti = index[_normalize_label(t)]
        pi = index[_normalize_label(p)]
        matrix[ti][pi] += 1

    return {
        "labels": labels,
        "matrix": matrix,
        "rows": "actual",
        "cols": "predicted",
    }


def run_full_pipeline(
    image_path: str,
    text_input: str,
    ground_truth: str | None = None,
    custom_model_dir: str = "Classifier_Custom",
    detections_json: str = "YOLO_custom/results/detections.json",
    output_json: str = "YOLO_custom/results/final_model2_output.json",
    output_image: str = "YOLO_custom/results/final_model2_output.jpg",
) -> dict:
    """
    Run end-to-end pipeline with a single call.

    Parameters
    ----------
    image_path : str
        Input image path.
    text_input : str
        Natural-language task prompt (text-based input).
    """
    image_file = Path(image_path)
    if not image_file.exists():
        raise FileNotFoundError(f"Input image not found: {image_file}")

    repo_root = Path(__file__).resolve().parent.parent

    def _resolve_repo_path(path_str: str) -> str:
        p = Path(path_str)
        if p.is_absolute():
            return str(p)
        return str((repo_root / p).resolve())

    model_dir = Path(custom_model_dir)
    if not model_dir.exists():
        # Fallback for invocation from inside YOLO_custom.
        model_dir = (repo_root / custom_model_dir).resolve()

    detections_json = _resolve_repo_path(detections_json)
    output_json = _resolve_repo_path(output_json)
    output_image = _resolve_repo_path(output_image)

    payload = run_yolo_and_model2(
        image_path=str(image_file),
        task_text=text_input,
        custom_model_dir=str(model_dir),
        detections_json=detections_json,
        output_json=output_json,
        output_image=output_image,
    )

    if ground_truth:
        selected = str(payload.get("model2", {}).get("selected_object", {}).get("class_name", ""))
        y_true = [ground_truth]
        y_pred = [selected]
        payload["evaluation"] = {
            "ground_truth": ground_truth,
            "prediction": selected,
            "accuracy": _compute_accuracy(y_true, y_pred),
            "confusion_matrix": _compute_confusion_matrix(y_true, y_pred),
        }

        # Persist evaluation in the same final output JSON for traceability.
        output_file = Path(output_json)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run full pipeline: Tiny YOLOv2 detection + semantic task-aware object selection"
        )
    )
    parser.add_argument("--image", type=str, required=False, help="Input image path")
    parser.add_argument(
        "--text-input",
        type=str,
        required=False,
        help="Text-based task input (example: 'object used to serve wine')",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Alias for --text-input (if both are provided, --text-input is used)",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=None,
        help="Optional expected class label for accuracy/confusion-matrix evaluation",
    )
    parser.add_argument(
        "--custom-model-dir",
        type=str,
        default="Classifier_Custom",
        help="Directory containing task_aware_selector.py",
    )
    parser.add_argument(
        "--detections-json",
        type=str,
        default="YOLO_custom/results/detections.json",
        help="Intermediate detections JSON output",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="YOLO_custom/results/final_model2_output.json",
        help="Final combined output JSON",
    )
    parser.add_argument(
        "--output-image",
        type=str,
        default="YOLO_custom/results/final_model2_output.jpg",
        help="Final annotated output image",
    )
    return parser


def _prompt_if_missing(value: str | None, prompt_text: str) -> str:
    if value and str(value).strip():
        return _normalize_cli_input(str(value))
    entered = _normalize_cli_input(input(prompt_text))
    if not entered:
        raise ValueError(f"Missing required input: {prompt_text}")
    return entered


if __name__ == "__main__":
    args = _build_parser().parse_args()
    # Interactive mode: if arguments are omitted, ask for them.
    image_arg = _prompt_if_missing(args.image, "Enter image path: ")
    final_text = args.text_input if args.text_input else args.task
    final_text = _prompt_if_missing(
        final_text,
        "Enter text-based task input (example: object used to serve wine): ",
    )
    gt_label = _normalize_cli_input(args.ground_truth) if args.ground_truth else None

    result = run_full_pipeline(
        image_path=image_arg,
        text_input=final_text,
        ground_truth=gt_label,
        custom_model_dir=args.custom_model_dir,
        detections_json=args.detections_json,
        output_json=args.output_json,
        output_image=args.output_image,
    )

    model2 = result.get("model2", {})
    selected = model2.get("selected_object", {})
    summary = {
        "task": result.get("task"),
        "selected_object": selected.get("class_name"),
        "confidence": selected.get("confidence"),
        "score": selected.get("score"),
        "output_json": args.output_json,
        "output_image": result.get("final_output_image", args.output_image),
    }
    if result.get("evaluation"):
        summary["evaluation"] = result["evaluation"]
    print(json.dumps(summary, indent=2))
