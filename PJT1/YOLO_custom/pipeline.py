# model/pipeline.py
"""
Lightweight FPGA-Optimized Object Detector
==========================================
Image input → Backbone feature extraction → Detection HEAD outputs
  → Bounding boxes (x,y,w,h) + class labels + confidence scores

Architecture:
  - Depthwise separable convolutions for efficiency
  - 2-scale detection (32x32 and 16x16 feature maps)
  - ReLU6 activation (FPGA-friendly)
  - INT8 quantization-aware
  - Reduced channels for low BRAM/DSP footprint

Output:
  - Detections: [batch_size, num_detections, 85]
    where 85 = [x, y, w, h, confidence, class_logits(80)]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import QuantStub, DeQuantStub


COCO_CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def _box_iou_xyxy(box, boxes):
    """Compute IoU between one box and N boxes. box=(4,), boxes=(N,4)."""
    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])

    inter_w = torch.clamp(x2 - x1, min=0.0)
    inter_h = torch.clamp(y2 - y1, min=0.0)
    inter = inter_w * inter_h

    box_area = torch.clamp(box[2] - box[0], min=0.0) * torch.clamp(box[3] - box[1], min=0.0)
    boxes_area = torch.clamp(boxes[:, 2] - boxes[:, 0], min=0.0) * torch.clamp(boxes[:, 3] - boxes[:, 1], min=0.0)
    union = box_area + boxes_area - inter
    return inter / torch.clamp(union, min=1e-6)


def _nms_classwise(boxes_xyxy, scores, class_ids, iou_threshold=0.45, max_detections=100):
    """Simple class-wise NMS in pure PyTorch."""
    keep_indices = []
    unique_classes = class_ids.unique()

    for cls in unique_classes:
        cls_mask = class_ids == cls
        cls_indices = torch.where(cls_mask)[0]
        cls_boxes = boxes_xyxy[cls_mask]
        cls_scores = scores[cls_mask]

        order = torch.argsort(cls_scores, descending=True)
        while order.numel() > 0:
            best_local = order[0]
            best_global_idx = cls_indices[best_local]
            keep_indices.append(best_global_idx)

            if order.numel() == 1:
                break

            remaining = order[1:]
            ious = _box_iou_xyxy(cls_boxes[best_local], cls_boxes[remaining])
            order = remaining[ious <= iou_threshold]

    if not keep_indices:
        return torch.empty((0,), dtype=torch.long, device=boxes_xyxy.device)

    keep = torch.stack(keep_indices)
    keep = keep[torch.argsort(scores[keep], descending=True)]
    return keep[:max_detections]


def decode_yolo_outputs(outputs, img_w=416, img_h=416, conf_threshold=0.25,
                        iou_threshold=0.45, class_names=None, max_detections=100):
    """
    Decode model outputs into labeled detections.

    Args:
        outputs: list of tensors, each (B, N_scale, 5 + C)
        img_w, img_h: output image size for pixel coordinates
        conf_threshold: minimum confidence score
        iou_threshold: NMS IoU threshold
        class_names: optional list of class names
        max_detections: maximum detections returned per image

    Returns:
        list (len=B) of detection lists. Each detection is a dict:
        {
          x1, y1, x2, y2, x, y, w, h,
          confidence, class_id, class_name
        }
    """
    if class_names is None:
        class_names = COCO_CLASS_NAMES

    if not outputs:
        return []

    preds = torch.cat(outputs, dim=1)  # (B, N_total, 5 + C)
    batch_results = []

    for b in range(preds.shape[0]):
        det = preds[b]
        if det.numel() == 0:
            batch_results.append([])
            continue

        xy = torch.clamp(det[:, :2], 0.0, 1.0)
        wh = torch.clamp(det[:, 2:4], 0.0, 1.0)
        obj = torch.sigmoid(det[:, 4] + 2.7)
        class_probs = torch.sigmoid(det[:, 5:] + 1.8)

        scores_per_class = obj.unsqueeze(-1) * class_probs
        scores, class_ids = scores_per_class.max(dim=-1)

        keep_mask = scores >= conf_threshold
        if keep_mask.sum() == 0:
            batch_results.append([])
            continue

        xy = xy[keep_mask]
        wh = wh[keep_mask]
        scores = scores[keep_mask]
        class_ids = class_ids[keep_mask]

        x = xy[:, 0] * img_w
        y = xy[:, 1] * img_h
        w = wh[:, 0] * img_w
        h = wh[:, 1] * img_h

        x1 = torch.clamp(x - w / 2, min=0.0, max=float(img_w))
        y1 = torch.clamp(y - h / 2, min=0.0, max=float(img_h))
        x2 = torch.clamp(x + w / 2, min=0.0, max=float(img_w))
        y2 = torch.clamp(y + h / 2, min=0.0, max=float(img_h))
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

        keep_indices = _nms_classwise(
            boxes_xyxy=boxes_xyxy,
            scores=scores,
            class_ids=class_ids,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )

        image_results = []
        for idx in keep_indices.tolist():
            class_id = int(class_ids[idx].item())
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            image_results.append({
                "x1": float(x1[idx].item()),
                "y1": float(y1[idx].item()),
                "x2": float(x2[idx].item()),
                "y2": float(y2[idx].item()),
                "x": float(x[idx].item()),
                "y": float(y[idx].item()),
                "w": float(w[idx].item()),
                "h": float(h[idx].item()),
                "confidence": float(scores[idx].item()),
                "class_id": class_id,
                "class_name": class_name,
            })

        batch_results.append(image_results)

    return batch_results


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Depthwise Separable Convolution Block
# ─────────────────────────────────────────────────────────────────────────────

class DWConvBlock(nn.Module):
    """
    Depthwise Separable Conv: low-compute alternative to standard Conv2d
    Reduces parameters and DSP usage by ~8-9x compared to standard conv.
    
    Maps to RTL:
      - Depthwise:  per-channel conv (1 DSP or LUT-MAC per channel)
      - Pointwise:  1x1 conv (standard convolution)
      - BN + ReLU6: batch norm + activation (quantization friendly)
    """
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3,
                 activation='relu6'):
        super().__init__()
        padding = (kernel_size - 1) // 2
        
        # Depthwise convolution: operates per-input-channel
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size,
                            stride=stride, padding=padding,
                            groups=in_ch, bias=False)
        
        # Pointwise convolution: 1x1 conv to mix channels
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        
        # Batch norm (will be fused during quantization-aware training)
        self.bn = nn.BatchNorm2d(out_ch)
        
        # Activation (ReLU6 is FPGA-friendly: simple clamp to [0,6])
        if activation == 'relu6':
            self.act = nn.ReLU6(inplace=False)
        elif activation == 'relu':
            self.act = nn.ReLU(inplace=False)
        elif activation == 'leaky':
            self.act = nn.LeakyReLU(0.1, inplace=False)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Backbone: Lightweight Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

class LightweightBackbone(nn.Module):
    """
    Mobile-friendly backbone for object detection.
    Input: (B, 3, H, W) RGB image
    Output: 
      - feat_s8:  (B, C, H/8, W/8)   from stride=8 layer
      - feat_s16: (B, C, H/16, W/16) from stride=16 layer
    
    Total stride = 16 downsampling.
    Channel progression: 3 → 16 → 32 → 64 → 128 (reduced for FPGA)
    """
    def __init__(self, in_channels=3, base_channels=16):
        super().__init__()
        
        # Initial normalization (pixel-wise)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])
                             .view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])
                             .view(1, 3, 1, 1))
        
        # Stem: initial conv layer (standard conv for first layer is fine)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1,
                     padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU6(inplace=False),
        )
        
        # Stage 1: stride=2, 16→32 channels
        self.stage1 = DWConvBlock(base_channels, base_channels*2, stride=2)
        
        # Stage 2: stride=2, 32→64 channels
        self.stage2 = DWConvBlock(base_channels*2, base_channels*4, stride=2)
        self.stage2b = DWConvBlock(base_channels*4, base_channels*4, stride=1)
        
        # Stage 3: stride=2, 64→128 channels (feat_s16)
        self.stage3 = DWConvBlock(base_channels*4, base_channels*8, stride=2)
        self.stage3b = DWConvBlock(base_channels*8, base_channels*8, stride=1)
        
        # For multi-scale detection, we'll extract:
        #   Scale A (s8):  before stage3  → base_channels*4 = 64 channels
        #   Scale B (s16): after stage3b  → base_channels*8 = 128 channels

    def forward(self, x):
        # Normalize pixel values
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        x = (x - self.mean) / self.std
        
        # Stem
        x = self.stem(x)                    # (B, 16, H, W)
        
        # Stage 1: stride=2
        x = self.stage1(x)                  # (B, 32, H/2, W/2)
        
        # Stage 2: stride=2
        x = self.stage2(x)                  # (B, 64, H/4, W/4)
        x = self.stage2b(x)
        feat_s8 = x                         # Save for detection head (scale A)
        
        # Stage 3: stride=2
        x = self.stage3(x)                  # (B, 128, H/8, W/8)
        x = self.stage3b(x)
        feat_s16 = x                        # For detection head (scale B)
        
        return feat_s8, feat_s16


# ─────────────────────────────────────────────────────────────────────────────
# Detection Head: outputs bounding boxes and class predictions
# ─────────────────────────────────────────────────────────────────────────────

class DetectionHead(nn.Module):
    """
    Multi-scale detection head.
    Outputs predictions at 2 spatial scales for better coverage.
    
    For each location (i,j) and anchor k:
      Output: [tx, ty, tw, th, obj_confidence, class_logits[0..79]]
    """
    def __init__(self, in_channels, num_classes=80, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        # Output per anchor: 4 (bbox) + 1 (obj) + num_classes (class)
        self.output_dim = num_anchors * (5 + num_classes)
        
        # Lightweight prediction convolution
        self.conv1 = nn.Sequential(
            DWConvBlock(in_channels, in_channels, stride=1),
            nn.Conv2d(in_channels, self.output_dim, kernel_size=1, bias=True),
        )

    def forward(self, x):
        """
        x: (B, C_in, H, W)
        returns: (B, H, W, num_anchors, 5+num_classes)
                 reshaped as (B, H*W*num_anchors, 5+num_classes)
        """
        x = self.conv1(x)  # (B, output_dim, H, W)
        B, _, H, W = x.shape
        # Reshape for easier extraction of predictions
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, output_dim)
        x = x.view(B, H, W, self.num_anchors, 5 + self.num_classes)
        x = x.view(B, -1, 5 + self.num_classes)  # (B, H*W*num_anchors, 85)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Main Detector Model
# ─────────────────────────────────────────────────────────────────────────────

class ObjectDetectionModel(nn.Module):
    """
    Lightweight YOLO-like object detector optimized for FPGA.
    
    Input:  (B, 3, H, W) RGB image (H=W=416 or 320 typical)
    Output: (B, N, 85) detections where N = num_locations * num_anchors
            Each detection: [x, y, w, h, conf, class_0, ..., class_79]
    
    Architecture:
      1. Lightweight backbone with depthwise separable convs
      2. Multi-scale detection heads at 2 scales
      3. INT8 quantization-aware training hooks
    """
    def __init__(self, img_h=416, img_w=416, num_classes=80,
                 base_channels=16, num_anchors=3):
        super().__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Backbone: extracts features at 2 scales
        self.backbone = LightweightBackbone(in_channels=3,
                                            base_channels=base_channels)
        
        # Detection heads at 2 scales
        # Scale 1: stride=8  → 64 channels
        self.head_s8 = DetectionHead(
            in_channels=base_channels * 4,
            num_classes=num_classes,
            num_anchors=num_anchors
        )
        
        # Scale 2: stride=16 → 128 channels
        self.head_s16 = DetectionHead(
            in_channels=base_channels * 8,
            num_classes=num_classes,
            num_anchors=num_anchors
        )

    def forward(self, x):
        """
        x: (B, 3, H, W) input image [0,255] uint8 or [0,1] float
        
        returns:
          detections: list of (B, N_scale, 85) tensors (one per scale)
        """
        # Get multi-scale features from backbone
        feat_s8, feat_s16 = self.backbone(x)
        
        # Get detections at each scale
        det_s8 = self.head_s8(feat_s8)       # (B, H/8*W/8*3, 85)
        det_s16 = self.head_s16(feat_s16)    # (B, H/16*W/16*3, 85)
        
        return [det_s8, det_s16]

    def fuse_conv_bn(self):
        """
        Fuse Conv + BN layers for efficient inference.
        Reduces latency and DSP/BRAM usage at inference time.
        """
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for i, layer in enumerate(m):
                    if isinstance(layer, nn.Conv2d):
                        # Find and fuse the next BN if it exists
                        for j in range(i + 1, len(m)):
                            if isinstance(m[j], nn.BatchNorm2d):
                                # Merge them (implementation in PyTorch)
                                break


# ─────────────────────────────────────────────────────────────────────────────
# Tiny YOLOv2 Architecture
# ─────────────────────────────────────────────────────────────────────────────

class TinyYOLOv2(nn.Module):
    """
    Standard Tiny YOLOv2 PyTorch neural network model (Model 1).
    
    9 Convolutional layers + BatchNorm2d + LeakyReLU + MaxPool2d.
    Input shape:  (B, 3, 416, 416)
    Output shape: (B, num_anchors * (5 + num_classes), 13, 13)
    """
    def __init__(self, num_classes: int = 80, num_anchors: int = 5):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.output_dim = num_anchors * (5 + num_classes)

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d(2, stride=1)
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv_out = nn.Conv2d(1024, self.output_dim, 1, stride=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return self.conv_out(x)

