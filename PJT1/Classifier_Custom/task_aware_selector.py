"""
task_aware_selector.py
======================
Task-Aware Object Selection Model
----------------------------------
Uses CLIP embeddings + hierarchical softmax to select the most
appropriate detected object for a user-defined task.

Pipeline:
    YOLO detections (JSON)
        └─► filter invalid
        └─► CLIP-encode object labels
    User task text
        └─► hierarchical softmax → classify task category
        └─► CLIP-encode task text
    Cosine similarity (task ↔ objects)  ×  YOLO confidence
        └─► rank objects
        └─► tie-break with raw CLIP similarity
        └─► return best object
"""

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

from semantic_parser import parse_task_to_object

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Try to import open_clip (preferred); fall back to the original openai/CLIP.
# ---------------------------------------------------------------------------
try:
    import open_clip  # pip install open-clip-torch
    _BACKEND = "open_clip"
except ImportError:
    try:
        import clip  # pip install git+https://github.com/openai/CLIP.git
        _BACKEND = "clip"
    except ImportError:
        _BACKEND = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASKS = [
    "step on something",
    "sit comfortably",
    "place flowers",
    "get potatoes out of fire",
    "water plant",
    "get lemon out of tea",
    "dig hole",
    "open bottle of beer",
    "open parcel",
    "serve wine",
    "pour sugar",
    "smear butter",
    "extinguish fire",
    "pound carpet",
]
NUM_TASKS = len(TASKS)

# Minimum valid confidence & bounding-box size thresholds
MIN_CONFIDENCE = 0.0
MIN_BBOX_AREA  = 1        # pixels²
MIN_COORD      = 0        # coordinates must be ≥ 0

# Tie-break threshold: scores within this margin are considered a tie
TIE_THRESHOLD = 0.02

# CLIP embedding dimension (ViT-B/32 → 512)
CLIP_DIM = 512


# ===========================================================================
# 1. Hierarchical Softmax Classifier
# ===========================================================================

class HierarchicalTaskClassifier(nn.Module):
    """
    Two-level hierarchical softmax classifier over 14 task categories.

    Architecture
    ────────────
    Level 1: binary router splits tasks into two broad groups
             (physical-manipulation vs. liquid/food-service)
    Level 2: per-group softmax over tasks within that group

    This mimics hierarchical softmax used in NLP to handle
    large output spaces efficiently.

    Groups
    ──────
    Group A (physical / tool tasks):
        step on something, sit comfortably, place flowers,
        get potatoes out of fire, water plant, dig hole,
        open parcel, extinguish fire, pound carpet          → indices 0,1,2,3,4,6,8,12,13

    Group B (liquid / food-service tasks):
        get lemon out of tea, open bottle of beer,
        serve wine, pour sugar, smear butter                → indices 5,7,9,10,11
    """

    # Task index → group assignment (0 = Group A, 1 = Group B)
    TASK_GROUPS = [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0]
    GROUP_A_IDX = [i for i, g in enumerate(TASK_GROUPS) if g == 0]  # 9 tasks
    GROUP_B_IDX = [i for i, g in enumerate(TASK_GROUPS) if g == 1]  # 5 tasks

    def __init__(self, input_dim: int = CLIP_DIM):
        super().__init__()

        # --- Level 1: binary router (Group A vs Group B) ---
        self.router = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2),           # 2 groups
        )

        # --- Level 2a: softmax over Group A tasks ---
        self.head_a = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, len(self.GROUP_A_IDX)),
        )

        # --- Level 2b: softmax over Group B tasks ---
        self.head_b = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, len(self.GROUP_B_IDX)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, CLIP_DIM) float tensor

        Returns
        -------
        task_probs : (batch, NUM_TASKS) – joint probability over all 14 tasks
        """
        # Level 1 – P(group | x)
        group_logits = self.router(x)                       # (B, 2)
        group_probs  = F.softmax(group_logits, dim=-1)      # (B, 2)
        p_a = group_probs[:, 0:1]                           # (B, 1)
        p_b = group_probs[:, 1:2]                           # (B, 1)

        # Level 2a – P(task | group A) × P(group A)
        logits_a  = self.head_a(x)                          # (B, |A|)
        probs_a   = F.softmax(logits_a, dim=-1) * p_a       # (B, |A|)

        # Level 2b – P(task | group B) × P(group B)
        logits_b  = self.head_b(x)                          # (B, |B|)
        probs_b   = F.softmax(logits_b, dim=-1) * p_b       # (B, |B|)

        # Scatter back to full 14-dim task probability vector
        batch = x.size(0)
        task_probs = torch.zeros(batch, NUM_TASKS, device=x.device)
        task_probs[:, self.GROUP_A_IDX] = probs_a
        task_probs[:, self.GROUP_B_IDX] = probs_b

        return task_probs   # (B, 14)  – sums to 1


class ClipTaskClassifier:
    """
    CLIP-based task text classifier over the 14 predefined task categories.

    It embeds all predefined tasks once and maps user task text to the
    nearest category via cosine similarity.
    """

    def __init__(self, clip_model, tokenizer, device: str):
        self.clip_model = clip_model
        self.tokenizer = tokenizer
        self.device = device
        self.task_embeddings = encode_object_labels(
            TASKS,
            clip_model=self.clip_model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

    def classify(self, task_text: str) -> tuple[str, int, float, torch.Tensor]:
        user_task_embedding = encode_task_text(
            task_text,
            clip_model=self.clip_model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

        similarities = (user_task_embedding @ self.task_embeddings.T).squeeze(0)
        best_index = int(similarities.argmax().item())
        best_task = TASKS[best_index]
        best_similarity = float(similarities[best_index].item())

        mapped_task_embedding = self.task_embeddings[best_index:best_index + 1]
        return best_task, best_index, best_similarity, mapped_task_embedding


# ===========================================================================
# 2. CLIP Loader
# ===========================================================================

def _load_clip(device: str = "cpu"):
    """
    Load the CLIP model using whichever backend is available.

    Returns
    -------
    model, preprocess, tokenizer, device
    """
    if _BACKEND is None:
        raise ImportError(
            "No CLIP library found. Install one of:\n"
            "  pip install open-clip-torch\n"
            "  pip install git+https://github.com/openai/CLIP.git"
        )

    # Clean up invalid system CA certificate bundle env vars if they point to non-existent paths
    for env_var in ["REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE"]:
        val = os.environ.get(env_var)
        if val and not os.path.exists(val):
            del os.environ[env_var]

    if _BACKEND == "open_clip":
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            tokenizer = open_clip.get_tokenizer("ViT-B-32")
            model = model.to(device).eval()
            return model, preprocess, tokenizer, device
        except Exception as exc:
            # Fallback to local cache loading if network fetch fails
            os.environ["HF_HUB_OFFLINE"] = "1"
            try:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained="openai"
                )
                tokenizer = open_clip.get_tokenizer("ViT-B-32")
                model = model.to(device).eval()
                return model, preprocess, tokenizer, device
            except Exception:
                raise RuntimeError(
                    "Offline CLIP load failed. Cache ViT-B-32/openai weights locally "
                    "before running offline."
                ) from exc

    else:  # original openai/CLIP
        clip_cache = Path.home() / ".cache" / "clip" / "ViT-B-32.pt"
        if not clip_cache.exists():
            raise FileNotFoundError(
                f"Offline CLIP load failed. Missing local weights: {clip_cache}"
            )

        model, preprocess = clip.load(str(clip_cache), device=device)
        tokenizer = clip.tokenize
        model.eval()
        return model, preprocess, tokenizer, device


# ===========================================================================
# 3. Module Functions
# ===========================================================================

def load_detections(json_path: str) -> list[dict[str, Any]]:
    """
    Step 1 – Load YOLO detections from a JSON file.

    Parameters
    ----------
    json_path : path to detections.json

    Returns
    -------
    List of raw detection dicts (unfiltered).
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Detections file not found: {json_path}")

    with open(path, "r") as fh:
        data = json.load(fh)

    raw = data.get("detections", [])
    print(f"[load_detections] Loaded {len(raw)} raw detections from '{json_path}'")
    return raw


def load_detection_payload(
    json_path: str,
) -> tuple[list[dict[str, Any]], str | None, str | None, str | None]:
    """
    Load detection entries + optional metadata from JSON.

    Supported keys in the JSON root:
    - image_path
    - input_image_path
    - input_image
    - image
    - task
    - final_output_image

    Returns
    -------
    detections, image_path, task_text, final_output_image
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Detections file not found: {json_path}")

    with open(path, "r") as fh:
        data = json.load(fh)

    detections = data.get("detections", [])
    task_text = data.get("task")
    final_output_image = data.get("final_output_image")

    image_path = (
        data.get("image_path")
        or data.get("input_image_path")
        or data.get("input_image")
        or data.get("image")
    )

    if image_path:
        image_path = str(image_path)
        image_path_obj = Path(image_path)
        if not image_path_obj.is_absolute():
            image_path_obj = (path.parent / image_path_obj).resolve()
        image_path = str(image_path_obj)

    if final_output_image:
        final_output_image = str(final_output_image)
        output_path_obj = Path(final_output_image)
        if not output_path_obj.is_absolute():
            output_path_obj = (path.parent / output_path_obj).resolve()
        final_output_image = str(output_path_obj)

    print(
        f"[load_detection_payload] Loaded {len(detections)} detections "
        f"(image_path={'set' if image_path else 'missing'}, "
        f"task={'set' if task_text else 'missing'}, "
        f"final_output_image={'set' if final_output_image else 'missing'})"
    )
    return detections, image_path, task_text, final_output_image


def _filter_detections(detections: list[dict]) -> list[dict]:
    """
    Step 2 – Remove invalid or out-of-bounds detections.

    A detection is invalid if:
    - confidence ≤ MIN_CONFIDENCE
    - bounding-box area < MIN_BBOX_AREA
    - x or y coordinate < MIN_COORD (negative → off-frame)
    - class_name is empty or missing
    """
    valid = []
    for det in detections:
        name  = det.get("class_name", "").strip()
        conf  = det.get("confidence", 0.0)

        area = 0.0
        in_bounds = True
        bbox = det.get("bbox")

        if isinstance(bbox, list) and len(bbox) == 4:
            try:
                x1, y1, x2_or_w, y2_or_h = [float(v) for v in bbox]
                if x1 < MIN_COORD or y1 < MIN_COORD or x2_or_w < MIN_COORD or y2_or_h < MIN_COORD:
                    in_bounds = False
                if x2_or_w > x1 and y2_or_h > y1:
                    area = (x2_or_w - x1) * (y2_or_h - y1)
                else:
                    area = x2_or_w * y2_or_h
            except (TypeError, ValueError):
                in_bounds = False
        else:
            x, y = float(det.get("x", 0)), float(det.get("y", 0))
            w, h = float(det.get("width", 0)), float(det.get("height", 0))
            area = w * h
            if x < MIN_COORD or y < MIN_COORD:
                in_bounds = False

        if not name:
            print(f"  [filter] Skipping detection with empty class_name")
            continue
        if conf <= MIN_CONFIDENCE:
            print(f"  [filter] Skipping '{name}' - confidence {conf:.3f} <= {MIN_CONFIDENCE}")
            continue
        if area < MIN_BBOX_AREA:
            print(f"  [filter] Skipping '{name}' – bbox area {area} < {MIN_BBOX_AREA}")
            continue
        if not in_bounds:
            print(f"  [filter] Skipping '{name}' – out-of-bounds bbox/coords")
            continue

        valid.append(det)

    print(f"[filter_detections] {len(valid)}/{len(detections)} detections passed filtering")
    return valid


def _extract_xyxy(det: dict[str, Any]) -> tuple[int, int, int, int]:
    """
    Convert detection box formats to integer (x1, y1, x2, y2).
    Supports both:
    - bbox: [x1, y1, x2, y2] or [x, y, w, h]
    - x, y, width, height
    """
    bbox = det.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        x1, y1, a, b = [float(v) for v in bbox]
        if a > x1 and b > y1:
            x2, y2 = a, b
        else:
            x2, y2 = x1 + a, y1 + b
    else:
        x1 = float(det.get("x", 0.0))
        y1 = float(det.get("y", 0.0))
        w = float(det.get("width", 0.0))
        h = float(det.get("height", 0.0))
        x2, y2 = x1 + w, y1 + h

    x1_i = int(round(max(0.0, x1)))
    y1_i = int(round(max(0.0, y1)))
    x2_i = int(round(max(0.0, x2)))
    y2_i = int(round(max(0.0, y2)))
    return x1_i, y1_i, x2_i, y2_i


def render_selection_image(
    detections: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    result: dict[str, Any],
    output_path: str = "selection_output.png",
    image_path: str | None = None,
    image_array: Any | None = None,
) -> str:
    """
    Draw detection boxes and labels, highlighting the selected best object.

    If image_path is provided and exists, boxes are drawn on that image.
    If image_array is provided, it is used as the drawing canvas.
    Otherwise, raises an error (no silent black fallback).
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise ImportError("Pillow is required for visualization. Install with: pip install pillow") from exc

    if image_array is not None:
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError("NumPy is required when passing image arrays.") from exc

        arr = np.asarray(image_array)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            image = Image.fromarray(arr, mode="L").convert("RGB")
        else:
            image = Image.fromarray(arr[..., :3]).convert("RGB")
    elif image_path and Path(image_path).exists():
        image = Image.open(image_path).convert("RGB")
    else:
        raise FileNotFoundError(
            "No valid input image found for overlay. "
            "Pass image_path in main(...), or include 'image_path'/'input_image_path' in detections JSON, "
            "or use run_with_yolo_output(...) so orig_img is available."
        )

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    best_name = result.get("best_object", "")
    best_score = float(result.get("score", -1e9))

    best_det_idx = None
    for idx, det in enumerate(detections):
        if det.get("class_name") == best_name:
            det_score = None
            for r in ranked:
                if r.get("class_name") == det.get("class_name") and abs(float(r.get("confidence", 0.0)) - float(det.get("confidence", 0.0))) < 1e-6:
                    det_score = float(r.get("score", -1e9))
                    break
            if det_score is not None and abs(det_score - best_score) < 1e-4:
                best_det_idx = idx
                break

    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = _extract_xyxy(det)
        is_best = idx == best_det_idx
        color = (50, 205, 50) if is_best else (255, 99, 71)
        label = f"{det.get('class_name', 'obj')} ({float(det.get('confidence', 0.0)):.2f})"

        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3 if is_best else 2)

        tx = x1
        ty = max(0, y1 - 20)
        draw.rectangle([(tx, ty), (tx + max(120, len(label) * 8), ty + 18)], fill=color)
        draw.text((tx + 3, ty + 2), label, fill=(0, 0, 0), font=font)

    task_line = f"Task: {result.get('task', '')}"
    best_line = f"Best: {best_name} | score={best_score:.4f}"
    draw.rectangle([(8, 8), (min(image.size[0] - 8, 620), 50)], fill=(240, 240, 240))
    draw.text((14, 12), task_line, fill=(10, 10, 10), font=font)
    draw.text((14, 30), best_line, fill=(10, 10, 10), font=font)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    print(f"[render_selection_image] Saved visualization to '{out}'")
    return str(out)


def detections_from_yolo_output(yolo_output: Any) -> tuple[list[dict[str, Any]], str | None, Any | None]:
    """
    Convert direct YOLO model output into the selector detection format.

    Supported inputs:
    - ultralytics Results (single) or list[Results]
    - dict/list already in the selector detection format

    Returns
    -------
    detections, image_path, image_array
    """
    if isinstance(yolo_output, dict):
        detections = yolo_output.get("detections", [])
        image_path = yolo_output.get("image_path")
        return detections, image_path, None

    if isinstance(yolo_output, list):
        if not yolo_output:
            return [], None, None
        first = yolo_output[0]
        if isinstance(first, dict):
            return yolo_output, None, None
        result_obj = first
    else:
        result_obj = yolo_output

    if result_obj is None:
        return [], None, None

    boxes = getattr(result_obj, "boxes", None)
    names = getattr(result_obj, "names", None)
    image_path = getattr(result_obj, "path", None)
    image_array = getattr(result_obj, "orig_img", None)

    detections: list[dict[str, Any]] = []
    if boxes is None:
        return detections, image_path, image_array

    xyxy = getattr(boxes, "xyxy", None)
    conf = getattr(boxes, "conf", None)
    cls = getattr(boxes, "cls", None)

    if xyxy is None or conf is None or cls is None:
        return detections, image_path, image_array

    xyxy_list = xyxy.tolist() if hasattr(xyxy, "tolist") else list(xyxy)
    conf_list = conf.tolist() if hasattr(conf, "tolist") else list(conf)
    cls_list = cls.tolist() if hasattr(cls, "tolist") else list(cls)

    for b, c, k in zip(xyxy_list, conf_list, cls_list):
        class_idx = int(k)
        if isinstance(names, dict):
            class_name = str(names.get(class_idx, class_idx))
        elif isinstance(names, list) and 0 <= class_idx < len(names):
            class_name = str(names[class_idx])
        else:
            class_name = str(class_idx)

        detections.append({
            "class_name": class_name,
            "confidence": float(c),
            "bbox": [float(v) for v in b],
        })

    return detections, image_path, image_array


def _run_selector_pipeline(
    detections: list[dict[str, Any]],
    task_text: str,
    image_path: str | None,
    image_array: Any | None,
    output_image_path: str,
    device: str | None,
) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[pipeline] Using device: {device}")

    print("\n[pipeline] Loading CLIP model …")
    clip_model, _, tokenizer, device = _load_clip(device)

    print("\n[pipeline] Filtering detections …")
    valid_detections = _filter_detections(detections)
    if not valid_detections:
        raise RuntimeError("No valid detections after filtering.")

    labels = [d["class_name"] for d in valid_detections]
    confidences = [d["confidence"] for d in valid_detections]

    # Semantic Task Parsing Layer:
    # Convert complex language intent into explicit object targets before
    # similarity scoring, so Model-2 can reason about tasks like
    # "object used to serve wine" instead of relying on raw keyword overlap.
    parsed = parse_task_to_object(task_text)
    target_object = parsed["target_object"]
    keywords = parsed["keywords"]
    semantic_task_text = " ".join(keywords) if keywords else task_text

    print(
        f"\n[pipeline] Semantic parse -> target='{target_object}', "
        f"keywords={keywords}"
    )

    print(f"\n[pipeline] Classifying task text: '{semantic_task_text}' …")
    task_classifier = ClipTaskClassifier(clip_model, tokenizer, device)
    classified_task, task_index, task_class_similarity, task_emb = task_classifier.classify(semantic_task_text)
    print(
        f"[pipeline] Task mapped to category #{task_index + 1}: '{classified_task}' "
        f"(sim={task_class_similarity:.4f})"
    )

    print(f"\n[pipeline] Encoding {len(labels)} object labels …")
    obj_embs = encode_object_labels(labels, clip_model, tokenizer, device)

    print("\n[pipeline] Computing weighted final scores …")
    scores, cos_sim = compute_similarity(task_emb, obj_embs, confidences)

    print("\n[pipeline] Ranking objects …")
    ranked = rank_objects(
        valid_detections,
        scores,
        cos_sim,
        target_object=target_object,
        keywords=keywords,
    )

    result = select_best_object(
        ranked,
        task_text,
        task_emb=task_emb,
        obj_embs=obj_embs,
        classifier=task_classifier,
        device=device,
        classified_task=classified_task,
        task_class_similarity=task_class_similarity,
    )
    result["target_object"] = target_object
    result["semantic_keywords"] = keywords

    visualization_path = render_selection_image(
        detections=valid_detections,
        ranked=ranked,
        result=result,
        output_path=output_image_path,
        image_path=image_path,
        image_array=image_array,
    )
    result["visualization_path"] = visualization_path

    reasoning_trace = {
        "task": task_text,
        "stages": [
            {
                "stage": "semantic_parse",
                "output": {
                    "task_text": task_text,
                    "target_object": target_object,
                    "keywords": keywords,
                    "semantic_task_text": semantic_task_text,
                },
            },
            {
                "stage": "filter_detections",
                "output": {
                    "input_count": len(detections),
                    "valid_count": len(valid_detections),
                    "detections": valid_detections,
                },
            },
            {
                "stage": "task_classification",
                "output": {
                    "semantic_task_text": semantic_task_text,
                    "classified_task": classified_task,
                    "task_index": int(task_index),
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
                    "candidates": [
                        {
                            "candidate_index": int(item.get("candidate_index", -1)),
                            "class_name": item.get("class_name"),
                            "confidence": float(item.get("confidence", 0.0)),
                            "score": float(item.get("score", 0.0)),
                            "cos_sim": float(item.get("cos_sim", 0.0)),
                            "priority_score": float(item.get("priority_score", 0.0)),
                            "target_match": bool(item.get("target_match", False)),
                            "keyword_match": bool(item.get("keyword_match", False)),
                        }
                        for item in ranked
                    ],
                },
            },
            {
                "stage": "ranking",
                "output": {
                    "ranked_candidates": ranked,
                },
            },
            {
                "stage": "selection",
                "output": {
                    "best_object": result.get("best_object"),
                    "selected_index": result.get("selected_index"),
                    "score": result.get("score"),
                    "cos_sim": result.get("cos_sim"),
                    "confidence": result.get("confidence"),
                    "selection_reason": result.get("selection_reason"),
                    "runner_up": result.get("runner_up"),
                },
            },
            {
                "stage": "visualization",
                "output": {
                    "visualization_path": visualization_path,
                },
            },
        ],
    }
    result["reasoning_trace"] = reasoning_trace

    print("\n[pipeline] Reasoning trace")
    for stage in reasoning_trace["stages"]:
        print(f"\n[pipeline] {stage['stage']}")
        print(json.dumps(stage["output"], indent=2))
    return result


def run_with_yolo_output(
    yolo_output: Any,
    task_text: str,
    output_image_path: str = "selection_output.png",
    image_path: str | None = None,
    device: str | None = None,
) -> dict:
    """
    Direct entrypoint for YOLO model output (no JSON intermediate required).

    Example:
        from ultralytics import YOLO
        import task_aware_selector as tas

        model = YOLO("yolov8n.pt")
        yolo_results = model("input.jpg")
        result = tas.run_with_yolo_output(yolo_results, "sit comfortably")
    """
    detections, yolo_image_path, image_array = detections_from_yolo_output(yolo_output)
    final_image_path = image_path if image_path is not None else yolo_image_path
    return _run_selector_pipeline(
        detections=detections,
        task_text=task_text,
        image_path=final_image_path,
        image_array=image_array,
        output_image_path=output_image_path,
        device=device,
    )


def encode_task_text(
    task_text: str,
    clip_model,
    tokenizer,
    device: str,
) -> torch.Tensor:
    """
    Step 3 – Embed the user task string with CLIP's text encoder.

    Returns
    -------
    task_emb : (1, CLIP_DIM) normalised float tensor
    """
    with torch.no_grad():
        if _BACKEND == "open_clip":
            tokens = tokenizer([task_text]).to(device)
            emb    = clip_model.encode_text(tokens)
        else:
            tokens = tokenizer([task_text]).to(device)
            emb    = clip_model.encode_text(tokens)

        emb = F.normalize(emb.float(), dim=-1)   # L2-normalise → unit sphere

    print(f"[encode_task_text] Encoded task: '{task_text}'  shape={tuple(emb.shape)}")
    return emb   # (1, D)


def encode_object_labels(
    labels: list[str],
    clip_model,
    tokenizer,
    device: str,
) -> torch.Tensor:
    """
    Step 4 – Embed all detected object class names with CLIP.

    Returns
    -------
    obj_embs : (N, CLIP_DIM) normalised float tensor
    """
    with torch.no_grad():
        if _BACKEND == "open_clip":
            tokens = tokenizer(labels).to(device)
            embs   = clip_model.encode_text(tokens)
        else:
            tokens = tokenizer(labels).to(device)
            embs   = clip_model.encode_text(tokens)

        embs = F.normalize(embs.float(), dim=-1)

    print(f"[encode_object_labels] Encoded {len(labels)} labels  shape={tuple(embs.shape)}")
    return embs   # (N, D)


def compute_similarity(
    task_emb:  torch.Tensor,   # (1, D)
    obj_embs:  torch.Tensor,   # (N, D)
    confidences: list[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Step 5 – Compute weighted cosine similarity scores.

    final_score_i = (clip_similarity_i * 0.7) + (yolo_confidence_i * 0.3)

    Both tensors are already L2-normalised, so cosine similarity
    reduces to a simple dot product.

    Returns
    -------
    scores : (N,) float tensor of weighted final scores
    """
    # Cosine similarity: (1, D) × (D, N) → (1, N) → (N,)
    cos_sim = (task_emb @ obj_embs.T).squeeze(0)           # (N,)

    conf_tensor = torch.tensor(confidences, dtype=torch.float32,
                               device=task_emb.device)     # (N,)

    scores = (cos_sim * 0.7) + (conf_tensor * 0.3)         # element-wise
    return scores, cos_sim   # also return raw cos_sim for tie-breaking


def rank_objects(
    detections: list[dict],
    scores:     torch.Tensor,   # (N,) weighted scores
    cos_sim:    torch.Tensor,   # (N,) raw cosine similarities (tie-break)
    target_object: str | None = None,
    keywords: list[str] | None = None,
) -> list[dict]:
    """
    Step 6 – Build a hierarchical ranked list of detections.

    Primary   sort key : weighted score (desc)
    Secondary sort key : raw cosine similarity (desc) – resolves ties
    Tertiary  sort key : YOLO confidence (desc)

    Returns
    -------
    ranked : list of dicts enriched with 'score', 'cos_sim' fields,
             ordered best → worst.
    """
    scores_np  = scores.cpu().tolist()
    cos_sim_np = cos_sim.cpu().tolist()

    def _norm(s: str) -> str:
        return " ".join(str(s).lower().split())

    target_norm = _norm(target_object) if target_object else ""
    keyword_norm = [_norm(k) for k in (keywords or []) if str(k).strip()]

    enriched = []
    for i, det in enumerate(detections):
        class_norm = _norm(det.get("class_name", ""))
        is_target_match = bool(target_norm) and (
            class_norm == target_norm
            or target_norm in class_norm
            or class_norm in target_norm
        )
        is_keyword_match = any(
            kw == class_norm or kw in class_norm or class_norm in kw
            for kw in keyword_norm
        )

        # Prioritize explicit semantic target match before CLIP fallback score.
        priority_score = float(scores_np[i])
        if is_target_match:
            priority_score += 1.50
        elif is_keyword_match:
            priority_score += 0.25

        enriched.append({
            **det,
            "candidate_index": i,
            "score":   scores_np[i],
            "cos_sim": cos_sim_np[i],
            "priority_score": priority_score,
            "target_match": bool(is_target_match),
            "keyword_match": bool(is_keyword_match),
        })

    # Sort: semantic target match first, then keyword match, then CLIP/confidence.
    ranked = sorted(
        enriched,
        key=lambda d: (
            d["target_match"],
            d["keyword_match"],
            d["priority_score"],
            d["score"],
            d["cos_sim"],
            d["confidence"],
        ),
        reverse=True,
    )
    return ranked


def select_best_object(
    ranked:       list[dict],
    task_text:    str,
    task_emb:     torch.Tensor | None = None,
    obj_embs:     torch.Tensor | None = None,
    classifier:   HierarchicalTaskClassifier | ClipTaskClassifier | None = None,
    device:       str | None = None,
    classified_task: str | None = None,
    task_class_similarity: float | None = None,
) -> dict:
    """
    Step 7 – Select and return the best candidate object.

    Returns the top-ranked object and metadata.

    Returns
    -------
    result dict with keys: task, classified_task, best_object, confidence
    """
    if not ranked:
        raise ValueError("No valid detections to select from.")

    classified = classified_task if classified_task is not None else task_text
    class_conf = task_class_similarity if task_class_similarity is not None else 1.0

    best = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    tie_break_applied = False

    # --- Tie-break check ---
    if runner_up is not None:
        second = runner_up
        score_gap = abs(best["score"] - second["score"])
        if score_gap < TIE_THRESHOLD:
            # Resolve by raw CLIP cosine similarity only
            if second["cos_sim"] > best["cos_sim"]:
                print(f"  [tie-break] '{second['class_name']}' beats "
                      f"'{best['class_name']}' on raw cos_sim "
                      f"({second['cos_sim']:.4f} > {best['cos_sim']:.4f})")
                best = second
                tie_break_applied = True

    output_conf = round(float(best["confidence"]), 4)

    result = {
        "task":            task_text,
        "classified_task": classified,
        "best_object":     best["class_name"],
        "selected_index":   int(best.get("candidate_index", -1)),
        "score":           round(best["score"],   4),
        "cos_sim":         round(best["cos_sim"], 4),
        "task_class_similarity": round(float(class_conf), 4),
        "confidence":      output_conf,
        "selection_reason": "tie_break" if tie_break_applied else "top_score",
    }
    if runner_up is not None:
        result["runner_up"] = {
            "class_name": runner_up.get("class_name"),
            "selected_index": int(runner_up.get("candidate_index", -1)),
            "score": round(float(runner_up.get("score", 0.0)), 4),
            "cos_sim": round(float(runner_up.get("cos_sim", 0.0)), 4),
            "confidence": round(float(runner_up.get("confidence", 0.0)), 4),
        }
    return result


# ===========================================================================
# 4. Main Orchestrator
# ===========================================================================

def main(
    detections_path: str = "detections.json",
    task_text:       str | None = None,
    image_path:      str | None = None,
    output_image_path: str | None = None,
    device:          str | None = None,
) -> dict:
    """
    End-to-end pipeline:

        load → filter → encode → similarity → rank → classify → select

    Parameters
    ----------
    detections_path : path to YOLO detections JSON
    task_text       : user-defined task string
    device          : 'cuda' | 'cpu' | None (auto-detect)

    Returns
    -------
    result dict  e.g.
        {
          "task":            "sit comfortably",
          "classified_task": "sit comfortably",
          "best_object":     "chair",
          "score":           0.7744,
          "cos_sim":         0.8800,
          "confidence":      0.88
        }
    """
    # ── Step 2: Load detections + optional metadata from file ───────────────
    print("\n[main] Loading detections …")
    raw_detections, detected_image_path, payload_task_text, payload_output_path = load_detection_payload(detections_path)

    final_task_text = task_text if task_text is not None else payload_task_text
    if final_task_text is None:
        final_task_text = "sit comfortably"

    final_image_path = image_path if image_path is not None else detected_image_path
    final_output_path = output_image_path if output_image_path is not None else payload_output_path
    if final_output_path is None:
        final_output_path = "selection_output.png"

    if final_image_path is None:
        raise ValueError(
            "Input image path is required for overlay output. "
            "Provide image_path argument or include 'image_path'/'input_image_path' in detections JSON."
        )
    print(f"[main] Using input image for overlay: '{final_image_path}'")

    result = _run_selector_pipeline(
        detections=raw_detections,
        task_text=final_task_text,
        image_path=final_image_path,
        image_array=None,
        output_image_path=final_output_path,
        device=device,
    )

    print("\n" + "="*55)
    print("  RESULT")
    print("="*55)
    print(json.dumps(result, indent=2))
    print("="*55)
    return result


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Task-Aware Object Selector (Model 2)")
    parser.add_argument("task_pos", nargs="?", default=None, help="Task text (positional)")
    parser.add_argument("image_pos", nargs="?", default=None, help="Input image path (positional)")
    parser.add_argument("--detections-json", "-d", type=str, default="YOLO_custom/results/detections.json",
                        help="Path to YOLO detections JSON file")
    parser.add_argument("--task", "-t", type=str, default=None, help="Task text")
    parser.add_argument("--image-path", "-i", type=str, default=None, help="Input image path")
    parser.add_argument("--output-image-path", "-o", type=str, default=None, help="Output annotated image path")
    args = parser.parse_args()

    # Resolve positional vs named arguments cleanly
    task_text = args.task or args.task_pos or "sit comfortably"
    image_path = args.image_path or args.image_pos

    det_path = args.detections_json
    if not Path(det_path).exists() and Path("detections.json").exists():
        det_path = "detections.json"

    main(
        detections_path=det_path,
        task_text=task_text,
        image_path=image_path,
        output_image_path=args.output_image_path,
    )
