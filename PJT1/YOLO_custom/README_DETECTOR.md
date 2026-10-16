# Lightweight FPGA-Optimized Object Detector

## Overview

The model has been **completely redesigned** from a task-aware object selector to a **full object detection model** that:

✅ **Takes an image as input**  
✅ **Detects all objects in the image**  
✅ **Outputs bounding boxes with class labels and confidence scores**

---

## Full Project Run Commands (Windows PowerShell)

Use these commands from workspace root (`DVCon26`) to run the entire project end-to-end.

### 1. Create and activate virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2. Install dependencies (Model-1 + Model-2)
```powershell
python -m pip install -r Classifier_Custom/requirements.txt
```

### 3. Run Model-1 detector unit test
```powershell
python YOLO_custom/test_detector.py
```

### 4. Run Model-2 selector unit tests (includes semantic parser tests)
```powershell
python -m unittest -v Classifier_Custom/test_task_aware_selector.py
```

### 5. Run YOLO11n-only inference on one image
```powershell
python YOLO_custom/yolo_inference.py "C:/path/to/input.jpg" --output "YOLO_custom/results/detections.json"
```

### 6. Run full pipeline (YOLO11n + Semantic Task Parsing + Model-2 ranking)
```powershell
python YOLO_custom/run_model2_with_image_output.py --image "C:/path/to/input.jpg" --task "object used to serve wine" --custom-model-dir "Classifier_Custom" --detections-json "YOLO_custom/results/detections.json" --output-json "YOLO_custom/results/final_model2_output.json" --output-image "YOLO_custom/results/final_model2_output.jpg"
```

### 6A. Run full pipeline via orchestrator (single recommended entrypoint)
```powershell
python YOLO_custom/orchestrator.py --image "C:/path/to/input.jpg" --text-input "object used to serve wine"
```

You can also use `--task` as an alias for `--text-input`.

With evaluation metrics (accuracy + confusion matrix):
```powershell
python YOLO_custom/orchestrator.py --image "C:/path/to/input.jpg" --text-input "object used to serve wine" --ground-truth "wine glass"
```

### 6B. Real-Time Laptop Camera Mode (Interactive UI)
To open the live laptop camera with the on-screen task text box and submit button:
```powershell
python YOLO_custom/orchestrator.py
```
Or explicitly:
```powershell
python YOLO_custom/orchestrator.py --camera
```
- **Live Stream**: Displays camera feed in a window.
- **On-Screen Text Box**: Type any task prompt (e.g. `sit comfortably`, `serve wine`, `open bottle`).
- **[SUBMIT] Button / Enter Key**: Captures the current camera frame, runs YOLO11n + Model 2 Task-Aware Selector, and marks the selected object with bounding box directly on screen.
- **[RESET] Button / 'R' Key**: Clears the selection box and resumes live preview.
- **'ESC' or 'Q' Key**: Exits the camera window.

### 7. Optional: run selector directly from detections JSON
This expects `detections.json` to exist in `Classifier_Custom`, or pass values programmatically via `main(...)`.
```powershell
cd Classifier_Custom
python task_aware_selector.py "something to sit comfortably" "C:/path/to/input.jpg"
cd ..
```

### 8. Optional training/export flow for FPGA detector artifacts
```powershell
cd YOLO_custom
python train.py
python pipeline_sim.py
python export.py
cd ..
```

### Expected output files
- `YOLO_custom/results/detections.json`
- `YOLO_custom/results/final_model2_output.json`
- `YOLO_custom/results/final_model2_output.jpg`
- `YOLO_custom/results/detector.onnx` (after export)

---

## What Changed

### Old Architecture (Task-Aware Selector)
- Input: Pre-detected objects + task type
- Job: Select which object is best for the given task
- Output: Single selected object index

### New Architecture (Object Detector - YOLO-like)
- Input: Raw image (416×416 RGB)
- Job: Detect all objects and classify them
- Output: Bounding boxes [x, y, w, h] + class predictions + confidence scores

---

## Model Architecture

### Backbone: Lightweight Feature Extractor
```
Input (3, 416, 416)
  ↓
Stem Conv (3 → 16 channels)
  ↓
Stage 1: DWConv stride=2 (16 → 32 ch, spatial: 416→208)
  ↓
Stage 2: DWConv stride=2 (32 → 64 ch, spatial: 208→104)
  ↓ [Extract features at stride=8 for Scale A]
  ↓
Stage 3: DWConv stride=2 (64 → 128 ch, spatial: 104→52)
  ↓ [Features at stride=16 for Scale B]
```

### Detection Heads: Multi-Scale Predictions
**Scale A (stride=8, 52×52 feature map):**
- 52×52×3 = 8,112 anchor locations
- Output per anchor: [x, y, w, h, confidence, class_logits_0..79] = 85 values

**Scale B (stride=16, 26×26 feature map):**
- 26×26×3 = 2,028 anchor locations
- Same 85 values per anchor

---

## Key Features

### 1. **FPGA-Friendly**
- ✅ Depthwise separable convolutions (8-9× fewer DSPs than standard conv)
- ✅ ReLU6 activation (quantization-friendly)
- ✅ Minimal BRAM usage
- ✅ INT8 quantization support

### 2. **Multi-Scale Detection**
- 2 detection scales (stride 8× and 16×)
- Captures objects of different sizes
- Covers small and large objects

### 3. **Quantization-Aware Training**
- FP32 warmup phase: 10 epochs
- INT8 QAT fine-tuning: 40 epochs
- ~1-2% accuracy loss with INT8 quantization

### 4. **Lightweight**
- Only ~1-2M parameters (vs. 12M+ for YOLOv5n)
- ~4MB model size in FP32, ~1MB in INT8
- Suitable for resource-constrained FPGA

---

## How to Use

### 1. **Test the Model**
```powershell
python test_detector.py
```
This will:
- Create the model
- Run a forward pass
- Check output shapes and ranges
- Verify QAT preparation works

### 2. **Train the Model**
```powershell
python train.py
```
This will:
- Create synthetic training data
- Train for 50 epochs (10 FP32 + 40 QAT)
- Save to `results/detector_model.pt`
- Generate training history

To train using the first 2000 COCO train2017 images:
```powershell
python train.py --images-dir "<path-to-coco>/train2017" --annotations-file "<path-to-coco>/annotations/instances_train2017.json" --subset-size 2000 --subset-output "results/instances_train2017_train2000.json"
```
This creates a subset annotation file and trains on those first 2000 images.

### 3. **Run Functional Simulation**
```powershell
python pipeline_sim.py
```
This will:
- Load the trained model
- Run inference on test images
- Generate detection statistics
- Create golden reference vectors for RTL verification
- Save to `results/golden_vectors.json`

### 4. **Export for FPGA**
```powershell
python export.py
```
This will:
- Export to ONNX format (`detector.onnx`)
- Extract INT8 weights
- Generate C headers for embedded systems
- Create metadata files for hardware integration

---

## Output Format

### Detection Output
Each scale outputs a tensor of shape `(batch_size, num_locations, 85)` where:

```
[0:2]   = [center_x, center_y]        (normalized 0-1)
[2:4]   = [width, height]             (normalized 0-1)
[4]     = objectness_confidence        (0-1)
[5:85]  = class_logits                (80 classes)
```

---

## Usage Example

```python
import torch
from pipeline import ObjectDetectionModel

# Create model
model = ObjectDetectionModel(img_h=416, img_w=416, num_classes=80)
model.load_state_dict(torch.load('results/detector_model.pt'))
model.eval()

# Load and prepare image
image = torch.randn(1, 3, 416, 416)  # or load real image

# Run inference
with torch.no_grad():
    detections = model(image)  # List of 2 scales

# Process detections
det_s8, det_s16 = detections
print(f"Scale 1 (8×): {det_s8.shape}")   # (1, 8112, 85)
print(f"Scale 2 (16×): {det_s16.shape}") # (1, 2028, 85)

# Extract bounding boxes for scale 1
bboxes = det_s8[..., :4]      # (1, 8112, 4)
confs = det_s8[..., 4]        # (1, 8112)
classes = det_s8[..., 5:]     # (1, 8112, 80)
```

### Two-Model Task-Aware Inference (YOLO → Selector)

If you have a second model that must choose the most appropriate object for a task,
use the bridge in `example_detector_usage.py`:

```python
import importlib.util
from pathlib import Path
import torch

from example_detector_usage import (
  ObjectDetector,
  TaskAwareObjectIdentifier,
  CUSTOM_MODEL_FILE,
  CUSTOM_MODEL_CLASS,
  CUSTOM_WEIGHTS_PATH,
)

detector = ObjectDetector(model_path='results/detector_model.pt')

custom_model_file = Path(CUSTOM_MODEL_FILE)
spec = importlib.util.spec_from_file_location('custom_second_model_module', str(custom_model_file))
custom_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(custom_module)

CustomModel = getattr(custom_module, CUSTOM_MODEL_CLASS)
second_model = CustomModel()

weights = torch.load(str(CUSTOM_WEIGHTS_PATH), map_location='cpu')
if isinstance(weights, dict) and 'state_dict' in weights:
  weights = weights['state_dict']

second_model.load_state_dict(weights, strict=False)
second_model.eval()

pipeline = TaskAwareObjectIdentifier(detector, second_model)
result = pipeline.identify(image_or_path=image_np_or_path, task='grasping')

print(result['selected_object'])
```

The second model receives:
- `input_image`: original YOLO input image tensor `(1, 3, 416, 416)`
- `detections`: decoded YOLO candidate objects (dict list)
- `class_ids`, `confidences`, `boxes_xyxy`: tensorized candidate metadata

---

## Model Configuration

All parameters are in the scripts:
- `img_h=416, img_w=416` - Input image size
- `num_classes=80` - COCO dataset classes
- `base_channels=16` - Lightweight backbone
- `num_anchors=3` - Anchors per location

Can be easily modified for different configs:
```python
# Smaller/faster:
model = ObjectDetectionModel(img_h=320, img_w=320, base_channels=8)

# Larger/more accurate:
model = ObjectDetectionModel(img_h=640, img_w=640, base_channels=32)
```

---

## Hardware Integration

Once trained and exported:

1. **ONNX Model** → Feed to Vivado HLS/Quartus for synthesis
2. **C Headers** → Embed INT8 weights in FPGA firmware
3. **Golden Vectors** → Use for RTL co-simulation with cocotb
4. **Metadata** → Guides hardware team on quantization scales

---

## Training Progress

FP32 Phase (10 epochs):
- Learns basic feature extraction
- No quantization awareness

QAT Phase (40 epochs):
- Inserts fake quantization operators
- Learns to minimize quantization error
- Final model is INT8 compatible

---

## Performance Expectations

With the lightweight architecture:
- **Latency**: ~50-100ms per frame on FPGA (100MHz, pipelined)
- **Throughput**: ~10-20 frames/sec
- **Model size**: ~1MB (INT8)
- **BRAM usage**: ~200-300KB
- **DSP usage**: ~100-150 DSPs (Genesys-2 has 900)

---

## Next Steps

1. ✅ Train the model locally
2. ✅ Export to ONNX
3. ✅ Generate RTL cocotb testbench using golden vectors
4. ✅ Synthesize on Genesys-2
5. ✅ Run on actual hardware

---

## Files

| File | Purpose |
|------|---------|
| `pipeline.py` | Model architecture (ObjectDetectionModel) |
| `train.py` | Training script with QAT support |
| `export.py` | Export to ONNX and INT8 weights |
| `pipeline_sim.py` | Functional simulation & golden vector generation |
| `test_detector.py` | Quick unit tests |

---

## References

- **YOLOv5/v8**: Lightweight detection backbone design
- **Depthwise Separable Convolution**: Resource-efficient feature extraction
- **INT8 Quantization**: PyTorch QAT framework
- **FPGA Acceleration**: Hardware-software co-design principles

