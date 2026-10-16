#!/usr/bin/env python
"""
Practical Example: Using the Object Detector
Demonstrates how to:
1. Load a trained model
2. Prepare input images
3. Run inference
4. Visualize detections
"""

import torch
import numpy as np
import json
import os
import argparse
import importlib.util
from pathlib import Path
from pipeline import ObjectDetectionModel, decode_yolo_outputs, COCO_CLASS_NAMES
import yolo_inference

PROJECT_ROOT = Path(__file__).resolve().parent

CUSTOM_MODEL_FILE = PROJECT_ROOT / 'CUSTOM_MODEL_FILE'
CUSTOM_MODEL_CLASS = 'CUSTOM_MODEL_CLASS'
CUSTOM_WEIGHTS_PATH = PROJECT_ROOT / 'CUSTOM_WEIGHTS_PATH'

class ObjectDetector:
    """Wrapper for easy inference"""
    
    def __init__(self, model_path='results/detector_model.pt', 
                 device='cpu'):
        model_path = Path(model_path)
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
        self.device = device
        self.model = ObjectDetectionModel(
            img_h=416, img_w=416,
            num_classes=80,
            base_channels=16,
            num_anchors=3
        )
        
        try:
            state_dict = torch.load(str(model_path), map_location=device)
            self.model.load_state_dict(state_dict, strict=False)
            print(f"✓ Loaded model from {model_path}")
        except FileNotFoundError:
            print(f"⚠  Model not found at {model_path} - using random weights")
        
        self.model = self.model.to(device)
        self.model.eval()
        
        self.class_names = COCO_CLASS_NAMES
    
    def prepare_image(self, image_or_path):
        """
        Prepare image for inference
        Args:
            image_or_path: numpy array (H,W,3) [0,255] or path to image
        Returns:
            tensor (1, 3, 416, 416) ready for inference
        """
        if isinstance(image_or_path, str):
            # Load from file (requires PIL)
            try:
                from PIL import Image
                img = Image.open(image_or_path).convert('RGB')
                img = np.array(img, dtype=np.uint8)
            except ImportError:
                print("PIL required for image loading. Falling back to random image.")
                img = np.random.randint(0, 256, (416, 416, 3), dtype=np.uint8)
        else:
            img = image_or_path
        
        # Resize to 416x416
        scale_h = 416 / img.shape[0]
        scale_w = 416 / img.shape[1]
        scale = min(scale_h, scale_w)
        
        new_h = int(img.shape[0] * scale)
        new_w = int(img.shape[1] * scale)
        
        # Simple nearest-neighbor resize
        img_resized = np.zeros((416, 416, 3), dtype=np.uint8)
        img_resized[:new_h, :new_w] = np.array([
            [img[int(i/scale), int(j/scale)] 
             for j in range(new_w)] 
            for i in range(new_h)
        ])
        
        # Convert to tensor
        tensor = torch.from_numpy(img_resized).permute(2, 0, 1)  # (3, H, W)
        tensor = tensor.unsqueeze(0).float() / 255.0  # (1, 3, H, W), normalize
        return tensor.to(self.device)
    
    def detect(self, image_or_path, conf_threshold=0.75):
        """
        Run detection on image
        Args:
            image_or_path: image tensor or path
            conf_threshold: minimum confidence score
        Returns:
            detections: list of dicts with bbox, class, confidence
        """
        img_tensor = self.prepare_image(image_or_path)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)

        decoded = decode_yolo_outputs(
            outputs=outputs,
            img_w=416,
            img_h=416,
            conf_threshold=conf_threshold,
            iou_threshold=0.45,
            class_names=self.class_names,
            max_detections=100,
        )
        return decoded[0] if decoded else []
    
    def postprocess_detections(self, detections, nms_threshold=0.45):
        """
        Apply Non-Maximum Suppression to remove duplicate detections
        Args:
            detections: list of detection dicts
            nms_threshold: IoU threshold for NMS
        Returns:
            filtered detections
        """
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda d: d['confidence'], 
                          reverse=True)
        
        keep = []
        for det in detections:
            # Check if this detection overlaps too much with kept ones
            should_keep = True
            for kept in keep:
                # Do not suppress across different classes
                if det.get('class_id') != kept.get('class_id'):
                    continue

                # Simple IoU calculation
                x1_min = det['x1']
                y1_min = det['y1']
                x1_max = det['x2']
                y1_max = det['y2']

                x2_min = kept['x1']
                y2_min = kept['y1']
                x2_max = kept['x2']
                y2_max = kept['y2']
                
                # Intersection
                xi_min = max(x1_min, x2_min)
                yi_min = max(y1_min, y2_min)
                xi_max = min(x1_max, x2_max)
                yi_max = min(y1_max, y2_max)
                
                if xi_max > xi_min and yi_max > yi_min:
                    intersection = (xi_max - xi_min) * (yi_max - yi_min)
                    area1 = max(0.0, x1_max - x1_min) * max(0.0, y1_max - y1_min)
                    area2 = max(0.0, x2_max - x2_min) * max(0.0, y2_max - y2_min)
                    union = area1 + area2 - intersection
                    iou = intersection / union if union > 0 else 0
                    
                    if iou > nms_threshold:
                        should_keep = False
                        break
            
            if should_keep:
                keep.append(det)
        
        return keep

    def detect_and_save_json(self, image_or_path, output_json_path,
                             conf_threshold=0.75, apply_postprocess=False,
                             nms_threshold=0.45):
        """
        Run detection and save labeled objects to a JSON file.
        Returns the saved payload dictionary.
        """
        detections = self.detect(image_or_path, conf_threshold=conf_threshold)
        if apply_postprocess:
            detections = self.postprocess_detections(
                detections,
                nms_threshold=nms_threshold,
            )

        label_counts = {}
        for det in detections:
            name = det.get('class_name', 'unknown')
            label_counts[name] = label_counts.get(name, 0) + 1

        payload = {
            'num_detections': len(detections),
            'num_unique_labels': len(label_counts),
            'detected_labels': sorted(label_counts.keys()),
            'label_counts': label_counts,
            'detections': detections,
        }

        output_dir = os.path.dirname(output_json_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)

        print(f"  ✓ Saved labeled detections → {output_json_path}")
        return payload


class TaskAwareObjectIdentifier:
    """
    Two-model inference bridge:
      1) YOLO detector finds object candidates
      2) Second model receives original image + candidates + task
         and returns the most task-appropriate object index
    """

    def __init__(self, detector, second_model,
                 conf_threshold=0.75, apply_postprocess=True,
                 nms_threshold=0.45):
        self.detector = detector
        self.second_model = second_model
        self.conf_threshold = conf_threshold
        self.apply_postprocess = apply_postprocess
        self.nms_threshold = nms_threshold

    def _run_second_model(self, image_tensor, detections, task):
        class_ids = torch.tensor(
            [d.get('class_id', -1) for d in detections],
            dtype=torch.long,
            device=image_tensor.device,
        )
        confidences = torch.tensor(
            [d.get('confidence', 0.0) for d in detections],
            dtype=torch.float32,
            device=image_tensor.device,
        )
        boxes_xyxy = torch.tensor(
            [[d.get('x1', 0.0), d.get('y1', 0.0), d.get('x2', 0.0), d.get('y2', 0.0)] for d in detections],
            dtype=torch.float32,
            device=image_tensor.device,
        )

        result = self.second_model.forward(
            input_image=image_tensor,
            task=task,
            detections=detections,
            class_ids=class_ids,
            confidences=confidences,
            boxes_xyxy=boxes_xyxy,
        )

        if isinstance(result, dict):
            return int(result.get('selected_index', -1)), result
        if isinstance(result, torch.Tensor):
            if result.numel() == 1:
                return int(result.item()), {'selected_index': int(result.item())}
            return int(torch.argmax(result).item()), {
                'selected_index': int(torch.argmax(result).item()),
                'scores': result.detach().cpu().tolist(),
            }
        if isinstance(result, (int, np.integer)):
            return int(result), {'selected_index': int(result)}

        raise ValueError(
            'Second model must return int, tensor, or dict with selected_index.'
        )

    def identify(self, image_or_path, task):
        image_tensor = self.detector.prepare_image(image_or_path)

        with torch.no_grad():
            outputs = self.detector.model(image_tensor)

        decoded = decode_yolo_outputs(
            outputs=outputs,
            img_w=416,
            img_h=416,
            conf_threshold=self.conf_threshold,
            iou_threshold=0.45,
            class_names=self.detector.class_names,
            max_detections=100,
        )
        detections = decoded[0] if decoded else []

        if self.apply_postprocess:
            detections = self.detector.postprocess_detections(
                detections,
                nms_threshold=self.nms_threshold,
            )

        if not detections:
            return {
                'task': task,
                'selected_object': None,
                'selected_index': -1,
                'num_candidates': 0,
                'candidates': [],
            }

        selected_index, raw_second_model = self._run_second_model(
            image_tensor=image_tensor,
            detections=detections,
            task=task,
        )

        if selected_index < 0 or selected_index >= len(detections):
            selected_index = int(np.argmax([d['confidence'] for d in detections]))

        return {
            'task': task,
            'selected_index': selected_index,
            'selected_object': detections[selected_index],
            'num_candidates': len(detections),
            'candidates': detections,
            'second_model_raw': raw_second_model,
        }


def example_1_synthetic_image():
    """Example 1: Test on synthetic random image"""
    print("\n" + "="*60)
    print("  EXAMPLE 1: Synthetic Image")
    print("="*60)
    
    detector = ObjectDetector()
    
    # Create a random test image
    test_image = np.random.randint(0, 256, (416, 416, 3), dtype=np.uint8)
    
    print("\nRunning detection...")
    detections = detector.detect(test_image, conf_threshold=0.75)
    detections = detector.postprocess_detections(detections)
    
    print(f"\nFound {len(detections)} detections:")
    for i, det in enumerate(detections[:5]):  # Show first 5
        print(f"  {i+1}. {det['class_name']:15} conf={det['confidence']:.2f} "
              f"bbox=({det['x']:.0f},{det['y']:.0f},{det['w']:.0f},{det['h']:.0f})")


def example_2_batch_inference():
    """Example 2: Batched inference on multiple images"""
    print("\n" + "="*60)
    print("  EXAMPLE 2: Batch Inference")
    print("="*60)
    
    detector = ObjectDetector()
    
    print("\nProcessing batch of 4 images...")
    batch_images = torch.randint(0, 256, (4, 3, 416, 416), dtype=torch.uint8)
    
    with torch.no_grad():
        outputs = detector.model(batch_images.float() / 255.0)
    
    for batch_idx, (det_s8, det_s16) in enumerate([(outputs[0], outputs[1])]):
        total_detections = 0
        for scale_det in [det_s8, det_s16]:
            high_conf = (scale_det[batch_idx, :, 4] > 0.5).sum().item()
            total_detections += high_conf
        
        print(f"  Image {batch_idx}: {total_detections} high-confidence detections")


def example_3_quantization():
    """Example 3: Show quantization effect"""
    print("\n" + "="*60)
    print("  EXAMPLE 3: INT8 Quantization")
    print("="*60)
    
    print("\nPreparing model for INT8 quantization...")
    try:
        from torch.ao.quantization import prepare_qat, get_default_qat_qconfig
        
        detector = ObjectDetector()
        
        # Prepare for QAT
        detector.model.train()
        detector.model.qconfig = get_default_qat_qconfig('fbgemm')
        detector.model = prepare_qat(detector.model)
        
        print("  ✓ QAT hooks installed")
        
        # Run sample inference with quantization active
        detector.model.eval()
        test_image = torch.randint(0, 256, (1, 3, 416, 416), 
                                  dtype=torch.uint8).float() / 255.0
        
        with torch.no_grad():
            outputs = detector.model(test_image)
        
        print("  ✓ Forward pass with quantization works")
        print(f"    Output 1 shape: {outputs[0].shape}")
        print(f"    Output 2 shape: {outputs[1].shape}")
        
    except ImportError:
        print("  PyTorch quantization module not available")


def example_4_live_stats():
    """Example 4: Collect statistics across multiple inferences"""
    print("\n" + "="*60)
    print("  EXAMPLE 4: Inference Statistics")
    print("="*60)
    
    detector = ObjectDetector()
    
    print("\nCollecting stats over 10 random images...")
    import time
    
    times = []
    all_detections = []
    
    for i in range(10):
        test_image = np.random.randint(0, 256, (416, 416, 3), 
                                      dtype=np.uint8)
        
        t0 = time.time()
        dets = detector.detect(test_image, conf_threshold=0.75)
        dt = time.time() - t0
        times.append(dt)
        
        all_detections.extend(dets)
    
    avg_time = np.mean(times) * 1000  # Convert to ms
    fps = 1.0 / np.mean(times)
    
    print(f"\n  Results over 10 images:")
    print(f"    Average latency: {avg_time:.1f} ms/image")
    print(f"    Throughput: {fps:.1f} FPS")
    print(f"    Total detections: {len(all_detections)}")
    print(f"    Avg detections per image: {len(all_detections)/10:.1f}")


def example_5_export_json():
    """Example 5: Save labeled detections to JSON file"""
    print("\n" + "="*60)
    print("  EXAMPLE 5: Save Detections to JSON")
    print("="*60)

    detector = ObjectDetector()
    test_image = np.random.randint(0, 256, (416, 416, 3), dtype=np.uint8)

    detector.detect_and_save_json(
        image_or_path=test_image,
        output_json_path=str(PROJECT_ROOT / 'results' / 'detections.json'),
        conf_threshold=0.75,
        apply_postprocess=True,
        nms_threshold=0.45,
    )


def example_6_task_aware_selection():
    """Example 6: YOLO + second model (task-aware object selection)"""
    print("\n" + "="*60)
    print("  EXAMPLE 6: Task-Aware Final Object Selection")
    print("="*60)

    detector = ObjectDetector()

    custom_model_file = Path(CUSTOM_MODEL_FILE)
    if not custom_model_file.is_absolute():
        custom_model_file = PROJECT_ROOT / custom_model_file

    if custom_model_file.exists():
        spec = importlib.util.spec_from_file_location(
            'custom_second_model_module',
            str(custom_model_file),
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f'Unable to load custom model module from: {custom_model_file}')

        custom_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_module)

        custom_model_cls = getattr(custom_module, CUSTOM_MODEL_CLASS)
        second_model = custom_model_cls()

        custom_weights_path = Path(CUSTOM_WEIGHTS_PATH)
        if not custom_weights_path.is_absolute():
            custom_weights_path = PROJECT_ROOT / custom_weights_path

        if custom_weights_path.exists():
            state = torch.load(str(custom_weights_path), map_location=detector.device)
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            second_model.load_state_dict(state, strict=False)
        second_model = second_model.to(detector.device)
        second_model.eval()
    else:
        from model2_task_ranker import TaskAwareRankerModel
        second_model = TaskAwareRankerModel().to(detector.device)
        second_model.eval()

    pipeline = TaskAwareObjectIdentifier(
        detector=detector,
        second_model=second_model,
        conf_threshold=0.75,
        apply_postprocess=True,
        nms_threshold=0.45,
    )

    test_image = np.random.randint(0, 256, (416, 416, 3), dtype=np.uint8)
    task_name = 'grasping'

    result = pipeline.identify(test_image, task=task_name)
    selected = result['selected_object']

    print(f"\nTask: {task_name}")
    print(f"Candidates from YOLO: {result['num_candidates']}")
    if selected is None:
        print("Selected object: None (no detections)")
        return

    print(
        f"Selected object: {selected['class_name']} "
        f"(conf={selected['confidence']:.2f}, idx={result['selected_index']})"
    )


def example_7_yolo11n_json_export(image_path, model_name='yolo11n.pt', output_json='results/detections.json'):
    """Example 7: Run YOLO11n single-image inference and export detections JSON."""
    print("\n" + "="*60)
    print("  EXAMPLE 7: YOLO11n JSON Export")
    print("="*60)

    image_file = Path(image_path)
    if not image_file.is_absolute():
        image_file = PROJECT_ROOT / image_file

    output_file = Path(output_json)
    if not output_file.is_absolute():
        output_file = PROJECT_ROOT / output_file

    # Call the dedicated inference module to keep output schema stable
    # for downstream task-aware ranking models.
    payload = yolo_inference.main(
        image_path=str(image_file),
        output_path=str(output_file),
    )

    print(f"  ✓ Saved detections JSON to: {output_file}")
    print(f"  ✓ Detections count: {len(payload.get('detections', []))}")


def main(ultra_image=None, ultra_model='yolo11n.pt', ultra_output='results/detections.json'):
    print("\n" + "#"*60)
    print("  OBJECT DETECTOR - PRACTICAL EXAMPLES")
    print("#"*60)
    
    try:
        example_1_synthetic_image()
        example_2_batch_inference()
        example_3_quantization()
        example_4_live_stats()
        example_5_export_json()
        example_6_task_aware_selection()
        if ultra_image:
            example_7_yolo11n_json_export(
                image_path=ultra_image,
                model_name=ultra_model,
                output_json=ultra_output,
            )
        
        print("\n" + "#"*60)
        print("  ✓ All examples completed successfully!")
        print("#"*60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run detector examples and optional YOLO11n JSON export.')
    parser.add_argument('--ultra-image', type=str, default=None, help='Optional image path for yolo_inference.py bridge execution')
    parser.add_argument('--ultra-model', type=str, default='yolo11n.pt', help='YOLO11n model name/path')
    parser.add_argument('--ultra-output', type=str, default='results/detections.json', help='Output JSON path for YOLO11n detections')
    args = parser.parse_args()

    main(
        ultra_image=args.ultra_image,
        ultra_model=args.ultra_model,
        ultra_output=args.ultra_output,
    )
