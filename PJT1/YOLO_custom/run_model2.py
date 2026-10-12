#!/usr/bin/env python
"""
Run model 2 (task-aware ranker) on detections JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from model2_task_ranker import TaskAwareRankerModel


def run_model2(input_json: str, output_json: str) -> dict:
    input_path = Path(input_json)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    task = str(payload.get("task", ""))
    detections = payload.get("detections", [])

    model2 = TaskAwareRankerModel()
    result = model2.forward(task=task, detections=detections)

    output_payload = {
        "task": task,
        "num_candidates": len(detections),
        **result,
    }

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    return output_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run task-aware model2 ranking on YOLO detections")
    parser.add_argument("--input", type=str, default="results/task_aware_input.json", help="Input JSON containing task + detections")
    parser.add_argument("--output", type=str, default="results/model2_output.json", help="Output JSON for model2 result")
    args = parser.parse_args()

    out = run_model2(args.input, args.output)
    print(json.dumps(out, indent=2))
