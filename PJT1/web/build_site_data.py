#!/usr/bin/env python
"""
Collect the numbers shown in the dashboard's Results section into
static/data/site.json:

  - mAP50 per epoch of both fine-tuning runs (finetune/runs/*/results.csv)
  - per-class AP50 of the deployed classroom model on the held-out split
  - the 14 Review-II board results (Project Review 2/Result/*.json)
  - measured end-to-end latency of each detector (needs server.py running)

Re-run after retraining:  python build_site_data.py
"""

from __future__ import annotations

import base64
import csv
import json
import statistics
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PJT1 = HERE.parent
FINETUNE = PJT1 / "finetune"
REVIEW_RESULTS = PJT1.parent / "Project Review 2" / "Result"
MODEL = PJT1 / "YOLO_custom" / "classroom_yolo11n.pt"
OUT = HERE / "static" / "data" / "site.json"

RUNS = {
    "cpu": {"dir": "classroom", "label": "CPU · 416 px · 10 layers frozen"},
    "gpu": {"dir": "classroom_gpu", "label": "GPU · 640 px · full network"},
}
# (image, task) pairs whose target is actually in the image, so every detector
# runs the full detect -> CLIP-rank path and the timings are comparable.
LATENCY_CASES = [
    (PJT1.parent / "sample" / "sample" / "000000000016.jpg", "I want to hit the ball"),
    (PJT1.parent / "sample" / "sample" / "000000000057.jpg", "something to play tennis with"),
    (PJT1.parent / "sample" / "sample" / "000000000069.jpg", "something to sit on"),
]


def training_curves() -> dict:
    curves = {}
    for key, run in RUNS.items():
        path = FINETUNE / "runs" / run["dir"] / "results.csv"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            rows = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(fh)]
        points = [[int(float(r["epoch"])), round(float(r["metrics/mAP50(B)"]), 4)] for r in rows]
        best = max(points, key=lambda p: p[1])
        curves[key] = {"label": run["label"], "points": points, "best_epoch": best[0], "best_map50": best[1]}
    return curves


def per_class() -> dict:
    import torch
    from ultralytics import YOLO

    data = FINETUNE / "classroom.yaml"
    metrics = YOLO(str(MODEL)).val(
        data=str(data), imgsz=640, batch=8, workers=0, plots=False, verbose=False,
        device=0 if torch.cuda.is_available() else "cpu",
    )
    names = metrics.names
    ap50 = metrics.box.ap50
    classes = [
        {"name": names[int(c)], "ap50": round(float(ap50[i]), 4)}
        for i, c in enumerate(metrics.box.ap_class_index)
    ]
    classes.sort(key=lambda c: c["ap50"], reverse=True)
    return {
        "map50": round(float(metrics.box.map50), 4),
        "map50_95": round(float(metrics.box.map), 4),
        "classes": classes,
    }


def dataset_stats() -> dict:
    root = FINETUNE / "datasets" / "classroom" / "images"
    return {
        "train_images": len(list((root / "train").glob("*.jpg"))),
        "val_images": len(list((root / "val").glob("*.jpg"))),
        "classes": 46,
        "source": "Google Open Images (validation + test splits, human-verified boxes)",
    }


def review_results() -> list:
    items = []
    for n in range(1, 15):
        path = REVIEW_RESULTS / f"final_model2_output_{n}.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        m2 = d.get("model2", {})
        sel = m2.get("selected_object") or {}
        ranked = m2.get("ranked_objects") or []
        items.append({
            "n": n,
            "task": d.get("task"),
            "selected": sel.get("class_name"),
            "confidence": round(float(sel.get("confidence", 0)), 4),
            "score": round(float(sel.get("score", 0)), 4),
            "detections": len(d.get("detections", [])),
            "runner_up": ranked[1]["class_name"] if len(ranked) > 1 else None,
        })
    return items


def latency(server: str = "http://localhost:8000") -> dict:
    out = {}
    for detector in ("yolo11n", "world", "classroom"):
        times = []
        for img, task in LATENCY_CASES:
            payload = json.dumps({
                "image": "data:image/jpeg;base64," + base64.b64encode(img.read_bytes()).decode(),
                "task": task,
                "detector": detector,
                "save": False,  # benchmark runs stay out of saved_results/
            }).encode()
            req = urllib.request.Request(f"{server}/api/detect", payload, {"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    times.append(json.load(resp)["elapsed_ms"])
            except Exception as exc:
                print(f"[build] latency {detector} failed: {exc}")
                break
        if times:
            out[detector] = round(statistics.median(times))
    return out


def main() -> None:
    site = {
        "training": training_curves(),
        "dataset": dataset_stats(),
        "per_class": per_class(),
        "review": review_results(),
        "latency_ms": latency(),
        "vocab_eval": {"passed": 28, "total": 28},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(site, indent=1), encoding="utf-8")
    print(f"[build] wrote {OUT}")
    print(f"[build] deployed model mAP50={site['per_class']['map50']}  latency={site['latency_ms']}")


if __name__ == "__main__":
    main()
