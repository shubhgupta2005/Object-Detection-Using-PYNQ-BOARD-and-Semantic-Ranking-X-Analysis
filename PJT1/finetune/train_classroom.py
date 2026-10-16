#!/usr/bin/env python
"""
Fine-tune YOLO11n (COCO-pretrained) on the classroom / room dataset.

    python prepare_openimages.py      # build the dataset first
    python train_classroom.py         # CPU by default

The best checkpoint is copied to YOLO_custom/classroom_yolo11n.pt, which the
pipeline loads with --detector classroom. Being a fixed-class YOLO, it is also
the model to export (ONNX / INT8) for the PYNQ-Z2 detector.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
BASE_WEIGHTS = HERE.parent / "YOLO_custom" / "yolo11n.pt"
OUTPUT_WEIGHTS = HERE.parent / "YOLO_custom" / "classroom_yolo11n.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO11n on classroom objects")
    parser.add_argument("--data", default=str(HERE / "classroom.yaml"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=416, help="416 keeps CPU training tractable")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=12, help="Early-stop after this many epochs without improvement")
    parser.add_argument("--freeze", type=int, default=10, help="Freeze the first N layers (0 = train the whole network)")
    parser.add_argument("--lr0", type=float, default=0.005)
    parser.add_argument("--name", default="classroom", help="Run name under finetune/runs/")
    args = parser.parse_args()

    model = YOLO(str(BASE_WEIGHTS))
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=str(HERE / "runs"),
        name=args.name,
        exist_ok=True,
        pretrained=True,
        # Small dataset: keep COCO features, train the detection head harder.
        freeze=args.freeze or None,
        lr0=args.lr0,
        cos_lr=True,
        close_mosaic=5,
        plots=True,
        verbose=True,
    )

    best = HERE / "runs" / args.name / "weights" / "best.pt"
    new = YOLO(str(best)).val(data=args.data, imgsz=args.imgsz, device=args.device, plots=False)
    print(f"[train] {args.name}: val mAP50 = {new.box.map50:.3f}   mAP50-95 = {new.box.map:.3f}")

    # Only replace the deployed model when the new run is actually better.
    if OUTPUT_WEIGHTS.exists():
        old = YOLO(str(OUTPUT_WEIGHTS)).val(data=args.data, imgsz=args.imgsz, device=args.device, plots=False)
        print(f"[train] current deployed model: val mAP50 = {old.box.map50:.3f}")
        if new.box.map50 <= old.box.map50:
            print(f"[train] kept existing {OUTPUT_WEIGHTS.name} (new run is not better)")
            return
    shutil.copy2(best, OUTPUT_WEIGHTS)
    print(f"[train] best weights -> {OUTPUT_WEIGHTS}")


if __name__ == "__main__":
    main()
