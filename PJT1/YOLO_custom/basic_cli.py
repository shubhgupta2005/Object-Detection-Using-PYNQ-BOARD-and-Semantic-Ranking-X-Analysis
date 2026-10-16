#!/usr/bin/env python
"""
Very basic CLI for running YOLO + model2 pipeline.

It lets the user provide:
1) image path
2) task text

If arguments are not passed, it prompts interactively.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from run_model2_with_image_output import main as run_pipeline


def _normalize_user_text(value: str) -> str:
    """Trim whitespace and remove one layer of matching wrapping quotes."""
    cleaned = value.strip()
    if len(cleaned) >= 2 and ((cleaned[0] == '"' and cleaned[-1] == '"') or (cleaned[0] == "'" and cleaned[-1] == "'")):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _prompt_if_missing(value: str | None, label: str) -> str:
    """Prompt user only when a required value was not provided by args."""
    if value and value.strip():
        return _normalize_user_text(value)
    return _normalize_user_text(input(f"Enter {label}: "))


def main() -> None:
    parser = argparse.ArgumentParser(description="Basic CLI: input image path and task")
    parser.add_argument("--image", type=str, default=None, help="Input image path")
    parser.add_argument("--task", type=str, default=None, help="Task text")
    parser.add_argument(
        "--custom-model-dir",
        type=str,
        default=str((Path(__file__).resolve().parent.parent / "Classifier_Custom").resolve()),
        help="Path to folder containing task_aware_selector.py",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(Path("results") / "final_model2_output.json"),
        help="Output JSON file",
    )
    parser.add_argument(
        "--output-image",
        type=str,
        default=str(Path("results") / "final_model2_output.jpg"),
        help="Output annotated image",
    )
    parser.add_argument(
        "--detections-json",
        type=str,
        default=str(Path("results") / "detections.json"),
        help="Intermediate detections JSON",
    )

    args = parser.parse_args()

    image_path = _prompt_if_missing(args.image, "image path")
    task_text = _prompt_if_missing(args.task, "task text")

    if not image_path:
        raise ValueError("Image path is required.")
    if not task_text:
        raise ValueError("Task text is required.")

    payload = run_pipeline(
        image_path=image_path,
        task_text=task_text,
        custom_model_dir=args.custom_model_dir,
        detections_json=args.detections_json,
        output_json=args.output_json,
        output_image=args.output_image,
    )

    model2 = payload.get("model2", {})
    selected = model2.get("selected_object", {})
    print("\n=== Done ===")
    print(f"Task: {payload.get('task')}")
    print(f"Selected object: {selected.get('class_name')}")
    print(f"Output JSON: {args.output_json}")
    print(f"Output image: {args.output_image}")


if __name__ == "__main__":
    main()
