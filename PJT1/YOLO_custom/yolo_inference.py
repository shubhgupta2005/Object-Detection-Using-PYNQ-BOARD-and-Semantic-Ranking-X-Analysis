#!/usr/bin/env python
"""
YOLO11n single-image inference module (Model 1).

This module detects objects in an image using the upgraded YOLO11n architecture
and exports structured JSON output compatible with downstream Model 2 (Task-Aware Ranker).
"""

from __future__ import annotations

import argparse
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from PIL import Image
from ultralytics import YOLO, YOLOWorld


def _normalize_path_text(value: str) -> str:
    """Trim whitespace and remove matching quotes."""
    cleaned = value.strip()
    if len(cleaned) >= 2 and ((cleaned[0] == '"' and cleaned[-1] == '"') or (cleaned[0] == "'" and cleaned[-1] == "'")):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _resolve_local_yolo_weights() -> Path:
    """Resolve local YOLO11n weights path."""
    env_value = os.environ.get("YOLO_WEIGHTS_PATH") or os.environ.get("YOLO11_WEIGHTS_PATH")
    here = Path(__file__).resolve().parent
    repo_root = here.parent

    candidates: Sequence[Path] = []
    if env_value:
        candidates = [Path(_normalize_path_text(env_value))]
    else:
        candidates = [
            here / "yolo11n.pt",
            repo_root / "yolo11n.pt",
            Path.cwd() / "yolo11n.pt",
        ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    # Default location in YOLO_custom folder
    return (here / "yolo11n.pt").resolve()


@lru_cache(maxsize=4)
def load_yolo_model(weights_path: str | Path | None = None) -> Dict[str, Any]:
    """
    Load the YOLO11n model once and cache it.

    Returns:
        A dictionary containing model object and backend metadata.
    """
    resolved_path = Path(weights_path).resolve() if weights_path else _resolve_local_yolo_weights()
    model = YOLO(str(resolved_path))

    return {
        "backend": "tinyyolov2",
        "model": model,
        "weights_path": str(resolved_path),
    }


@lru_cache(maxsize=1)
def load_world_model(weights_path: str = "yolov8s-worldv2.pt") -> YOLOWorld:
    """Load the open-vocabulary YOLO-World model once and cache it."""
    from ultralytics.nn import text_model

    here = Path(__file__).resolve().parent
    # Ultralytics' default weights_dir is cwd-relative; pin the CLIP text-encoder
    # download to this folder so launching from another directory reuses it.
    text_model.WEIGHTS_DIR = here / "weights"
    local = here / weights_path
    return YOLOWorld(str(local) if local.exists() else weights_path)


_world_classes: Tuple[str, ...] = ()


def run_world_inference(
    image_path: str,
    classes: Sequence[str],
    conf_threshold: float = 0.05,
    iou_threshold: float = 0.45,
) -> List[Dict[str, Any]]:
    """
    Run YOLO-World detection for an arbitrary list of class names.

    Returns detections in the same structure as run_inference().
    """
    global _world_classes

    img_file = Path(_normalize_path_text(image_path))
    if not img_file.exists():
        raise FileNotFoundError(f"Image not found: {img_file}")

    model = load_world_model()
    if tuple(classes) != _world_classes:
        # Same as YOLOWorld.set_classes(), but keeps the CLIP text encoder in memory;
        # the default reloads the 338 MB checkpoint on every call (~9 s on CPU).
        model.model.set_classes(list(classes), cache_clip_model=True)
        model.model.names = list(classes)
        model.predictor = None
        _world_classes = tuple(classes)

    results = model.predict(str(img_file), conf=conf_threshold, iou=iou_threshold, verbose=False)

    detections: List[Dict[str, Any]] = []
    if results and results[0].boxes is not None:
        names = results[0].names
        for box in results[0].boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            detections.append(
                {
                    "class_name": names[int(box.cls[0].item())],
                    "confidence": round(float(box.conf[0].item()), 4),
                    "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                }
            )

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


# YOLO11n fine-tuned on classroom / room objects by finetune/train_classroom.py.
CLASSROOM_WEIGHTS = Path(__file__).resolve().parent / "classroom_yolo11n.pt"


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert value to float safely."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def run_inference(
    model_bundle: Dict[str, Any],
    image_path: str,
    conf_threshold: float = 0.20,
    iou_threshold: float = 0.45,
) -> List[Dict[str, Any]]:
    """
    Run YOLO11n object detection on one image.

    Args:
        model_bundle: Output from load_yolo_model().
        image_path: Path to input image.
        conf_threshold: Minimum confidence threshold.
        iou_threshold: NMS IoU threshold.

    Returns:
        Detection list with normalized structure:
        [{"class_name": str, "confidence": float, "bbox": [x1, y1, x2, y2]}, ...]
    """
    img_file = Path(_normalize_path_text(image_path))
    if not img_file.exists():
        raise FileNotFoundError(f"Image not found: {img_file}")

    model: YOLO = model_bundle["model"]

    # Run inference with YOLO11n
    results = model(
        str(img_file),
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
    )

    detections: List[Dict[str, Any]] = []
    if results and len(results) > 0:
        result = results[0]
        names = model.names
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                class_name = names.get(cls_id, f"class_{cls_id}")

                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = [float(v) for v in xyxy]

                detections.append(
                    {
                        "class_name": class_name,
                        "confidence": round(confidence, 4),
                        "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    }
                )

    # If no detections found at threshold, retry with lower threshold (0.05) to provide candidates
    if not detections and conf_threshold > 0.05:
        results_fallback = model(
            str(img_file),
            conf=0.05,
            iou=iou_threshold,
            verbose=False,
        )
        if results_fallback and len(results_fallback) > 0:
            boxes = results_fallback[0].boxes
            names = model.names
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    class_name = names.get(cls_id, f"class_{cls_id}")
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = [float(v) for v in xyxy]
                    detections.append(
                        {
                            "class_name": class_name,
                            "confidence": round(confidence, 4),
                            "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                        }
                    )

    # Sort descending by confidence
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def filter_detections(detections: List[Dict[str, Any]], confidence_threshold: float = 0.001) -> List[Dict[str, Any]]:
    """
    Filter detection list by confidence and geometric validity.
    """
    filtered: List[Dict[str, Any]] = []

    for det in detections:
        class_name = det.get("class_name")
        confidence = _safe_float(det.get("confidence"), default=-1.0)
        bbox = det.get("bbox")

        if not isinstance(class_name, str) or not class_name.strip():
            continue
        if confidence < confidence_threshold:
            continue
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = (_safe_float(v) for v in bbox)
        if x2 <= x1 or y2 <= y1:
            continue

        filtered.append(
            {
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            }
        )

    return filtered


def save_detections(detections: List[Dict[str, Any]], output_path: str = "detections.json") -> Dict[str, Any]:
    """
    Save detections in structured JSON format.
    """
    payload = {"detections": detections}
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return payload


def main(
    image_path: str,
    output_path: str = "detections.json",
    conf_threshold: float = 0.20,
    detector: str = "yolo11n",
    classes: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """
    End-to-end Model 1 pipeline:
    1) Load YOLO11n, YOLO-World (detector="world") or the fine-tuned
       classroom YOLO11n (detector="classroom")
    2) Run inference on image
    3) Filter detections
    4) Save JSON
    """
    if detector == "classroom":
        if not CLASSROOM_WEIGHTS.exists():
            raise FileNotFoundError(f"Fine-tuned classroom model not found: {CLASSROOM_WEIGHTS}")
        model_bundle = load_yolo_model(str(CLASSROOM_WEIGHTS))
        raw_detections = run_inference(model_bundle, image_path=image_path, conf_threshold=conf_threshold)
    elif detector == "world":
        if not classes:
            raise ValueError("detector='world' requires a non-empty class list")
        conf_threshold = min(conf_threshold, 0.05)
        raw_detections = run_world_inference(image_path, classes, conf_threshold=conf_threshold)
    else:
        model_bundle = load_yolo_model()
        raw_detections = run_inference(model_bundle, image_path=image_path, conf_threshold=conf_threshold)
    filtered = filter_detections(raw_detections, confidence_threshold=conf_threshold)
    if not filtered and raw_detections:
        filtered = raw_detections
    return save_detections(filtered, output_path=output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO11n inference on a single image and save detections JSON.")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("--output", type=str, default="detections.json", help="Output JSON file path")
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold")
    parser.add_argument("--detector", choices=["yolo11n", "world", "classroom"], default="yolo11n", help="Detector backend")
    parser.add_argument("--classes", type=str, default=None, help="Comma-separated class names (YOLO-World only)")

    args = parser.parse_args()
    result_payload = main(
        image_path=args.image_path,
        output_path=args.output,
        conf_threshold=args.conf,
        detector=args.detector,
        classes=[c.strip() for c in args.classes.split(",")] if args.classes else None,
    )
    print(json.dumps(result_payload, indent=2))
