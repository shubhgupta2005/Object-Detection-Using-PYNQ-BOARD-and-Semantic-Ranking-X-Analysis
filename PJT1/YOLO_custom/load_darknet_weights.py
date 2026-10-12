#!/usr/bin/env python
"""
Darknet Tiny YOLOv2 Pretrained Weights Loader & Converter.

Downloads official Darknet pre-trained Tiny YOLOv2 weights (PASCAL VOC 20 classes / COCO)
and converts binary Darknet parameters into PyTorch state_dict for high-accuracy object detection.
"""

import os
import sys
import urllib.request
import ssl
import numpy as np
import torch
from pathlib import Path

# PASCAL VOC 20 Class Names
VOC_CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Map VOC class names to COCO / standard task ranker class names
VOC_TO_COCO_MAP = {
    "aeroplane": "airplane",
    "bicycle": "bicycle",
    "bird": "bird",
    "boat": "boat",
    "bottle": "bottle",
    "bus": "bus",
    "car": "car",
    "cat": "cat",
    "chair": "chair",
    "cow": "cow",
    "diningtable": "dining table",
    "dog": "dog",
    "horse": "horse",
    "motorbike": "motorcycle",
    "person": "person",
    "pottedplant": "potted plant",
    "sheep": "sheep",
    "sofa": "couch",
    "train": "train",
    "tvmonitor": "tv",
}

VOC_ANCHORS = [
    (1.08, 1.19),
    (3.42, 4.41),
    (6.63, 11.38),
    (9.42, 5.11),
    (16.62, 10.52),
]


def download_darknet_weights(target_path: Path) -> Path:
    """Download official Darknet tiny-yolo-voc.weights with SSL fallback."""
    if target_path.exists() and target_path.stat().st_size > 10_000_000:
        print(f"Local Darknet weights already exist: {target_path}")
        return target_path

    urls = [
        "http://pjreddie.com/media/files/yolov2-tiny-voc.weights",
        "http://pjreddie.com/media/files/yolov2-tiny.weights",
        "https://github.com/alexeyab/darknet/releases/download/darknet_yolo_v2_pre/yolov2-tiny.weights",
    ]

    # Fix invalid CA bundle environment variables
    for env_var in ["REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE"]:
        val = os.environ.get(env_var)
        if val and not os.path.exists(val):
            del os.environ[env_var]

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    for url in urls:
        try:
            print(f"Trying to download pre-trained Darknet Tiny YOLOv2 weights from {url}...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ctx, timeout=30) as response, open(target_path, "wb") as out_file:
                out_file.write(response.read())
            if target_path.exists() and target_path.stat().st_size > 10_000_000:
                print(f"✓ Successfully downloaded Darknet weights ({target_path.stat().st_size} bytes)")
                return target_path
        except Exception as e:
            print(f"Failed from {url}: {e}")

    raise RuntimeError("Failed to download Tiny YOLOv2 weights from all candidate URLs.")


def load_darknet_into_pytorch(weights_file: Path, py_model: torch.nn.Module):
    """
    Parse Darknet binary format and load weights into PyTorch TinyYOLOv2 model.
    """
    with open(weights_file, "rb") as fp:
        # Header: major, minor, revision, images_seen
        header = np.fromfile(fp, dtype=np.int32, count=4)
        weights = np.fromfile(fp, dtype=np.float32)

    ptr = 0
    layers = [
        py_model.layer1,
        py_model.layer2,
        py_model.layer3,
        py_model.layer4,
        py_model.layer5,
        py_model.layer6,
        py_model.layer7,
        py_model.layer8,
    ]

    for layer in layers:
        conv = layer[0]
        bn = layer[1]

        num_b = bn.bias.numel()

        # Read BN biases
        bn.bias.data.copy_(torch.from_numpy(weights[ptr : ptr + num_b]))
        ptr += num_b

        # Read BN weights (scales)
        bn.weight.data.copy_(torch.from_numpy(weights[ptr : ptr + num_b]))
        ptr += num_b

        # Read BN running mean
        bn.running_mean.data.copy_(torch.from_numpy(weights[ptr : ptr + num_b]))
        ptr += num_b

        # Read BN running variance
        bn.running_var.data.copy_(torch.from_numpy(weights[ptr : ptr + num_b]))
        ptr += num_b

        # Read Conv weights
        num_w = conv.weight.numel()
        conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv.weight)
        conv.weight.data.copy_(conv_w)
        ptr += num_w

    # Final conv layer (no BN)
    final_conv = py_model.conv_out
    num_b = final_conv.bias.numel()
    final_conv.bias.data.copy_(torch.from_numpy(weights[ptr : ptr + num_b]))
    ptr += num_b

    num_w = final_conv.weight.numel()
    conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(final_conv.weight)
    final_conv.weight.data.copy_(conv_w)
    ptr += num_w

    print("  [OK] Pretrained Darknet Tiny YOLOv2 weights converted & loaded into PyTorch model successfully!")


if __name__ == "__main__":
    from pipeline import TinyYOLOv2

    here = Path(__file__).resolve().parent
    weights_path = here / "yolov2-tiny-voc.weights"
    if not weights_path.exists() or weights_path.stat().st_size < 10_000_000:
        download_darknet_weights(weights_path)

    # 20 VOC classes = 5 * (5 + 20) = 125 output channels
    py_model = TinyYOLOv2(num_classes=20, num_anchors=5)
    load_darknet_into_pytorch(weights_path, py_model)

    out_pt = here / "tiny_yolo_v2.pt"
    torch.save(py_model.state_dict(), str(out_pt))
    print(f"  [OK] Saved pre-trained Tiny YOLOv2 PyTorch checkpoint to: {out_pt}")
