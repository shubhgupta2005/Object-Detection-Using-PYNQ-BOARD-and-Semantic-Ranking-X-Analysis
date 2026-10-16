#!/usr/bin/env python
"""
Build a YOLO-format classroom / room object dataset from Google Open Images.

Uses the human-verified bounding boxes of the Open Images V5+ validation and
test splits (the 2 GB train annotation file is not needed), picks up to
--per-class images for every target class, downloads the images from the
public S3 bucket (resumable, retried) and writes:

    finetune/datasets/classroom/
        images/{train,val}/*.jpg
        labels/{train,val}/*.txt
    finetune/classroom.yaml

Run prepare first, then train_classroom.py.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import random
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

HERE = Path(__file__).resolve().parent
META_DIR = HERE / "openimages" / "meta"
DATASET_DIR = HERE / "datasets" / "classroom"
YAML_PATH = HERE / "classroom.yaml"
SPLITS = ("validation", "test")
IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"
MAX_SIDE = 640

# Our class name -> Open Images display names merged into it.
# Eraser, pencil sharpener, stapler, scissors and ring binder were dropped: the
# validation + test splits hold only 0-7 boxed images of each.
CLASS_MAP: Dict[str, List[str]] = {
    "whiteboard": ["Whiteboard"],
    "desk": ["Desk"],
    "table": ["Table", "Kitchen & dining room table"],
    "chair": ["Chair"],
    "stool": ["Stool"],
    "bench": ["Bench"],
    "couch": ["Couch", "Sofa bed", "Studio couch"],
    "book": ["Book"],
    "pen": ["Pen"],
    "pencil case": ["Pencil case"],
    "ruler": ["Ruler"],
    "calculator": ["Calculator"],
    "envelope": ["Envelope"],
    "laptop": ["Laptop"],
    "monitor": ["Computer monitor"],
    "keyboard": ["Computer keyboard"],
    "mouse": ["Computer mouse"],
    "printer": ["Printer"],
    "tablet": ["Tablet computer"],
    "tv": ["Television"],
    "remote control": ["Remote control"],
    "cell phone": ["Mobile phone"],
    "headphones": ["Headphones"],
    "microphone": ["Microphone"],
    "backpack": ["Backpack"],
    "handbag": ["Handbag"],
    "bottle": ["Bottle"],
    "mug": ["Mug", "Coffee cup"],
    "glasses": ["Glasses"],
    "clock": ["Clock", "Wall clock"],
    "watch": ["Watch"],
    "lamp": ["Lamp"],
    "ceiling fan": ["Ceiling fan"],
    "light switch": ["Light switch"],
    "door": ["Door"],
    "window": ["Window"],
    "curtain": ["Curtain"],
    "bookshelf": ["Bookcase", "Shelf"],
    "cupboard": ["Cupboard", "Filing cabinet"],
    "dustbin": ["Waste container"],
    "box": ["Box"],
    "poster": ["Poster"],
    "picture frame": ["Picture frame"],
    "towel": ["Towel"],
    "helmet": ["Helmet"],
    "umbrella": ["Umbrella"],
}
CLASS_NAMES = list(CLASS_MAP)

Box = Tuple[int, float, float, float, float]  # class index, xc, yc, w, h (normalised)


def _label_ids() -> Dict[str, int]:
    """Open Images label id (/m/...) -> our class index."""
    by_name = {}
    with (META_DIR / "class-descriptions-boxable.csv").open(encoding="utf-8") as fh:
        for label_id, display in csv.reader(fh):
            by_name[display] = label_id
    mapping = {}
    for idx, (ours, oi_names) in enumerate(CLASS_MAP.items()):
        for oi_name in oi_names:
            if oi_name not in by_name:
                raise KeyError(f"Open Images has no boxable class '{oi_name}' (for '{ours}')")
            mapping[by_name[oi_name]] = idx
    return mapping


def _load_boxes(label_ids: Dict[str, int]) -> Dict[Tuple[str, str], List[Box]]:
    """(split, image_id) -> YOLO boxes of our classes."""
    boxes: Dict[Tuple[str, str], List[Box]] = defaultdict(list)
    for split in SPLITS:
        path = META_DIR / f"{split}-annotations-bbox.csv"
        with path.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                cls = label_ids.get(row["LabelName"])
                if cls is None:
                    continue
                x1, x2 = float(row["XMin"]), float(row["XMax"])
                y1, y2 = float(row["YMin"]), float(row["YMax"])
                boxes[(split, row["ImageID"])].append(
                    (cls, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
                )
        print(f"[prepare] parsed {path.name}")
    return boxes


def _select(boxes: Dict[Tuple[str, str], List[Box]], per_class: int, seed: int) -> List[Tuple[str, str]]:
    """Pick images so every class gets up to per_class images; rarest classes first."""
    by_class: Dict[int, List[Tuple[str, str]]] = defaultdict(list)
    for key, bxs in boxes.items():
        for cls in {b[0] for b in bxs}:
            by_class[cls].append(key)

    rng = random.Random(seed)
    chosen: set = set()
    coverage: Dict[int, int] = defaultdict(int)
    for cls in sorted(range(len(CLASS_NAMES)), key=lambda c: len(by_class[c])):
        candidates = by_class[cls][:]
        rng.shuffle(candidates)
        for key in candidates:
            if coverage[cls] >= per_class:
                break
            if key in chosen:
                continue
            chosen.add(key)
            for c in {b[0] for b in boxes[key]}:
                coverage[c] += 1

    print("[prepare] images per class (available -> selected):")
    for idx, name in enumerate(CLASS_NAMES):
        print(f"    {name:18s} {len(by_class[idx]):6d} -> {coverage[idx]:4d}")
    return sorted(chosen)


def _is_val(image_id: str, val_frac: float) -> bool:
    return int(hashlib.md5(image_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < val_frac


def _download(split: str, image_id: str, dest: Path, retries: int = 6) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    url = IMAGE_URL.format(split=split, image_id=image_id)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = resp.read()
            img = Image.open(io.BytesIO(data)).convert("RGB")
            img.thumbnail((MAX_SIDE, MAX_SIDE))
            tmp = dest.with_suffix(".part")
            img.save(tmp, "JPEG", quality=90)
            tmp.replace(dest)
            return True
        except Exception as exc:  # network drops are expected; back off and retry
            if attempt == retries - 1:
                print(f"[prepare] FAILED {image_id}: {exc}")
            time.sleep(1.5 * (attempt + 1))
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the classroom object dataset from Open Images")
    parser.add_argument("--per-class", type=int, default=40, help="Target images per class")
    parser.add_argument("--val-frac", type=float, default=0.15, help="Fraction of images used for validation")
    parser.add_argument("--workers", type=int, default=16, help="Parallel downloads")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    label_ids = _label_ids()
    boxes = _load_boxes(label_ids)
    selected = _select(boxes, args.per_class, args.seed)
    print(f"[prepare] downloading {len(selected)} images ...")

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (DATASET_DIR / sub).mkdir(parents=True, exist_ok=True)

    jobs = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for split, image_id in selected:
            part = "val" if _is_val(image_id, args.val_frac) else "train"
            dest = DATASET_DIR / "images" / part / f"{image_id}.jpg"
            jobs[pool.submit(_download, split, image_id, dest)] = (split, image_id, part)

        done = ok = 0
        for fut in as_completed(jobs):
            split, image_id, part = jobs[fut]
            done += 1
            if fut.result():
                ok += 1
                lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in boxes[(split, image_id)]]
                (DATASET_DIR / "labels" / part / f"{image_id}.txt").write_text("\n".join(lines) + "\n")
            if done % 100 == 0 or done == len(jobs):
                print(f"[prepare] {done}/{len(jobs)} processed, {ok} ok")

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    YAML_PATH.write_text(
        f"# Classroom / room objects from Open Images (built by prepare_openimages.py)\n"
        f"path: {DATASET_DIR.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n",
        encoding="utf-8",
    )
    n_train = len(list((DATASET_DIR / "images" / "train").glob("*.jpg")))
    n_val = len(list((DATASET_DIR / "images" / "val").glob("*.jpg")))
    print(f"[prepare] done: {n_train} train / {n_val} val images, {len(CLASS_NAMES)} classes -> {YAML_PATH}")


if __name__ == "__main__":
    main()
