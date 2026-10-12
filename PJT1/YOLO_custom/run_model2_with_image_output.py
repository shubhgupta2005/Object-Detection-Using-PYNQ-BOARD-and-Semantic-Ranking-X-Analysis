#!/usr/bin/env python

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw

import yolo_inference


def _load_custom_selector_module(custom_model_dir: str):
    model_dir = Path(custom_model_dir)
    module_path = model_dir / "task_aware_selector.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Custom selector module not found: {module_path}")

    # Ensure sibling imports inside the custom selector module work,
    # e.g. `from semantic_parser import ...` in Classifier_Custom.
    model_dir_str = str(model_dir.resolve())
    if model_dir_str not in sys.path:
        sys.path.insert(0, model_dir_str)

    spec = importlib.util.spec_from_file_location("task_aware_selector", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def _compute_crop_task_similarity(
    custom_module: Any,
    image_path: str,
    detections: List[Dict[str, Any]],
    clip_model: Any,
    preprocess: Any,
    task_emb: Any,
    device: str,
) -> List[float]:
    """
    Compute CLIP cosine similarity between task text embedding and each detected image crop.
    This explicitly passes image content to model 2.
    """
    image = Image.open(image_path).convert("RGB")
    sims: List[float] = []

    with custom_module.torch.no_grad():
        for det in detections:
            bbox = det.get("bbox", [0, 0, 0, 0])
            if not isinstance(bbox, list) or len(bbox) != 4:
                sims.append(0.0)
                continue

            x1, y1, x2, y2 = [float(v) for v in bbox]
            x1 = max(0, int(round(x1)))
            y1 = max(0, int(round(y1)))
            x2 = max(x1 + 1, int(round(x2)))
            y2 = max(y1 + 1, int(round(y2)))

            crop = image.crop((x1, y1, x2, y2))
            crop_tensor = preprocess(crop).unsqueeze(0).to(device)

            img_emb = clip_model.encode_image(crop_tensor)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

            sim = float((task_emb @ img_emb.T).squeeze().item())
            sims.append(sim)

    return sims


def _run_custom_model2_with_bbox(
    custom_module: Any,
    detections: List[Dict[str, Any]],
    task_text: str,
    image_path: str,
) -> Dict[str, Any]:
    filtered = custom_module._filter_detections(detections)
    if not filtered:
        return {
            "task": task_text,
            "classified_task": None,
            "task_index": -1,
            "task_class_similarity": 0.0,
            "selected_object": {},
            "ranked_objects": [],
            "status": "no_valid_detections",
            "message": "No valid detections for model 2 after filtering.",
            "reasoning_trace": {
                "task": task_text,
                "stages": [
                    {
                        "stage": "filter_detections",
                        "output": {
                            "input_count": len(detections),
                            "valid_count": 0,
                            "detections": [],
                        },
                    },
                    {
                        "stage": "selection",
                        "output": {
                            "selected_object": None,
                            "reason": "No valid detections available.",
                        },
                    },
                ],
            },
        }

    device = "cuda" if custom_module.torch.cuda.is_available() else "cpu"
    clip_model, preprocess, tokenizer, device = custom_module._load_clip(device)

    classifier = custom_module.ClipTaskClassifier(clip_model, tokenizer, device)
    classified_task, task_idx, task_class_similarity, task_emb = classifier.classify(task_text)

    labels = [d["class_name"] for d in filtered]
    confidences = [d["confidence"] for d in filtered]

    obj_embs = custom_module.encode_object_labels(labels, clip_model, tokenizer, device)
    scores, cos_sim = custom_module.compute_similarity(task_emb, obj_embs, confidences)

    # Additional image-aware score: task text vs CLIP embedding of each detected image crop.
    crop_cos_sims = _compute_crop_task_similarity(
        custom_module=custom_module,
        image_path=image_path,
        detections=filtered,
        clip_model=clip_model,
        preprocess=preprocess,
        task_emb=task_emb,
        device=device,
    )

    # Fuse base score with image-crop similarity to make model 2 image-aware.
    fused_ranked: List[Dict[str, Any]] = []
    for idx, det in enumerate(filtered):
        base_score = float(scores[idx].item())
        label_cos_sim = float(cos_sim[idx].item())
        crop_cos_sim = float(crop_cos_sims[idx])
        crop_cos_norm = (crop_cos_sim + 1.0) / 2.0

        fused_score = 0.65 * base_score + 0.35 * crop_cos_norm

        fused_ranked.append(
            {
                **det,
                "score": fused_score,
                "base_score": base_score,
                "cos_sim": label_cos_sim,
                "image_cos_sim": crop_cos_sim,
            }
        )

    fused_ranked.sort(key=lambda item: item["score"], reverse=True)
    best = fused_ranked[0]

    result = {
        "task": task_text,
        "classified_task": classified_task,
        "task_index": int(task_idx),
        "task_class_similarity": float(task_class_similarity),
        "selected_object": {
            "class_name": best.get("class_name"),
            "confidence": float(best.get("confidence", 0.0)),
            "bbox": best.get("bbox"),
            "score": float(best.get("score", 0.0)),
            "base_score": float(best.get("base_score", 0.0)),
            "cos_sim": float(best.get("cos_sim", 0.0)),
            "image_cos_sim": float(best.get("image_cos_sim", 0.0)),
        },
        "ranked_objects": fused_ranked,
    }

    result["reasoning_trace"] = {
        "task": task_text,
        "stages": [
            {
                "stage": "filter_detections",
                "output": {
                    "input_count": len(detections),
                    "valid_count": len(filtered),
                    "detections": filtered,
                },
            },
            {
                "stage": "task_classification",
                "output": {
                    "task_text": task_text,
                    "classified_task": classified_task,
                    "task_index": int(task_idx),
                    "task_class_similarity": float(task_class_similarity),
                },
            },
            {
                "stage": "object_encoding",
                "output": {
                    "labels": labels,
                    "confidence_values": [float(v) for v in confidences],
                    "count": len(labels),
                },
            },
            {
                "stage": "similarity_scoring",
                "output": {
                    "base_scores": [float(v.item()) for v in scores],
                    "label_cosine_similarity": [float(v.item()) for v in cos_sim],
                    "crop_cosine_similarity": [float(v) for v in crop_cos_sims],
                },
            },
            {
                "stage": "fusion_and_ranking",
                "output": {
                    "ranked_candidates": fused_ranked,
                },
            },
            {
                "stage": "selection",
                "output": {
                    "selected_object": result.get("selected_object"),
                },
            },
        ],
    }
    return result


def _draw_selected_box(image_path: str, selected_object: Dict[str, Any], output_image_path: str) -> None:
    bbox = selected_object.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("Selected object does not contain a valid bbox.")

    x1, y1, x2, y2 = [float(v) for v in bbox]
    class_name = str(selected_object.get("class_name", "object"))
    conf = float(selected_object.get("confidence", 0.0))
    score = float(selected_object.get("score", 0.0))

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=4)
    label = f"{class_name} | conf={conf:.3f} | score={score:.3f}"

    text_x = max(0, int(x1))
    text_y = max(0, int(y1) - 20)
    draw.rectangle([text_x, text_y, text_x + max(180, len(label) * 7), text_y + 18], fill=(0, 255, 0))
    draw.text((text_x + 4, text_y + 2), label, fill=(0, 0, 0))

    out_path = Path(output_image_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def main(
    image_path: str,
    task_text: str,
    custom_model_dir: str,
    yolo_model: str = "tiny_yolo_v2.pt",
    detections_json: str = "results/detections.json",
    output_json: str = "results/final_model2_output.json",
    output_image: str = "results/final_model2_output.jpg",
):
    repo_root = Path(__file__).resolve().parent.parent

    def _resolve_repo_path(path_str: str) -> str:
        p = Path(path_str)
        if p.is_absolute():
            return str(p)
        return str((repo_root / p).resolve())

    detections_json = _resolve_repo_path(detections_json)
    output_json = _resolve_repo_path(output_json)
    output_image = _resolve_repo_path(output_image)

    detections_payload = yolo_inference.main(
        image_path=image_path,
        output_path=detections_json,
    )

    detections = detections_payload.get("detections", [])

    # Persist per-input metadata in detections JSON so downstream model-2 runs
    # can always recover the correct image path and task context.
    detections_payload_with_meta = {
        "input_image_path": image_path,
        "image_path": image_path,
        "task": task_text,
        "detections": detections,
    }
    detections_json_path = Path(detections_json)
    detections_json_path.parent.mkdir(parents=True, exist_ok=True)
    with detections_json_path.open("w", encoding="utf-8") as f:
        json.dump(detections_payload_with_meta, f, indent=2)

    custom_module = _load_custom_selector_module(custom_model_dir)
    model2_result = _run_custom_model2_with_bbox(
        custom_module=custom_module,
        detections=detections,
        task_text=task_text,
        image_path=image_path,
    )

    selected_object = model2_result.get("selected_object") or {}
    should_draw = isinstance(selected_object, dict) and isinstance(selected_object.get("bbox"), list)
    if should_draw:
        _draw_selected_box(image_path, selected_object, output_image)
        final_output_image = output_image
    else:
        final_output_image = None

    output_payload = {
        "input_image_path": image_path,
        "image_path": image_path,
        "task": task_text,
        "detections": detections,
        "model2": model2_result,
        "reasoning_trace": model2_result.get("reasoning_trace"),
        "final_output_image": final_output_image,
    }

    out_json = Path(output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    return output_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Tiny YOLOv2 + custom model2 and save final image with selected bbox")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--task", type=str, required=True, help="Task text")
    parser.add_argument("--custom-model-dir", type=str, required=True, help="Folder containing task_aware_selector.py")
    parser.add_argument("--yolo-model", type=str, default="tiny_yolo_v2.pt", help="Tiny YOLOv2 model weights path")
    parser.add_argument("--detections-json", type=str, default="results/detections.json", help="Intermediate detections JSON output")
    parser.add_argument("--output-json", type=str, default="results/final_model2_output.json", help="Final combined output JSON")
    parser.add_argument("--output-image", type=str, default="results/final_model2_output.jpg", help="Final annotated output image")

    args = parser.parse_args()

    payload = main(
        image_path=args.image,
        task_text=args.task,
        custom_model_dir=args.custom_model_dir,
        yolo_model=args.yolo_model,
        detections_json=args.detections_json,
        output_json=args.output_json,
        output_image=args.output_image,
    )
    print(json.dumps(payload["model2"], indent=2))
    print(f"Saved image: {payload['final_output_image']}")
