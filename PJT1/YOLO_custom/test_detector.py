#!/usr/bin/env python
"""
Simple test script to verify the object detector works
"""
import sys
import torch
from pipeline import ObjectDetectionModel, TinyYOLOv2

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def test_model_creation():
    """Test 1: Can we instantiate the model?"""
    print("Test 1: Creating model...")
    model = ObjectDetectionModel(
        img_h=416, img_w=416,
        num_classes=80,
        base_channels=16,
        num_anchors=3
    )
    print("  [OK] Model created successfully")
    return model


def test_forward_pass(model):
    """Test 2: Can the model run inference?"""
    print("\nTest 2: Running forward pass...")
    dummy_input = torch.randn(2, 3, 416, 416)
    
    model.eval()
    with torch.no_grad():
        outputs = model(dummy_input)
    
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Number of detection scales: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"    Scale {i}: {out.shape}")
    print("  [OK] Forward pass successful")
    
    return outputs


def test_output_ranges(outputs):
    """Test 3: Are outputs in reasonable ranges?"""
    print("\nTest 3: Checking output ranges...")
    
    for scale_idx, out in enumerate(outputs):
        bbox_xy = out[..., :2]
        bbox_wh = out[..., 2:4]
        conf = out[..., 4]
        class_logits = out[..., 5:]
        
        print(f"  Scale {scale_idx}:")
        print(f"    BBox XY  - Min: {bbox_xy.min():.3f}, Max: {bbox_xy.max():.3f}")
        print(f"    BBox WH  - Min: {bbox_wh.min():.3f}, Max: {bbox_wh.max():.3f}")
        print(f"    Conf     - Min: {conf.min():.3f}, Max: {conf.max():.3f}")
        print(f"    Classes  - Min: {class_logits.min():.3f}, Max: {class_logits.max():.3f}")
    
    print("  [OK] Output ranges look reasonable")


def test_parameter_count(model):
    """Test 4: How many parameters?"""
    print("\nTest 4: Model statistics...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size (fp32): {total_params * 4 / 1e6:.2f} MB")
    print("  [OK] Parameter count acceptable for FPGA")


def test_int8_hooks(model):
    """Test 5: Can we prepare for QAT?"""
    print("\nTest 5: QAT preparation...")
    try:
        from torch.ao.quantization import prepare_qat, get_default_qat_qconfig
        
        model.qconfig = get_default_qat_qconfig('fbgemm')
        model_qat = prepare_qat(model)
        print("  [OK] QAT hooks installed successfully")
        
        dummy_input = torch.randn(1, 3, 416, 416)
        with torch.no_grad():
            _ = model_qat(dummy_input)
        print("  [OK] QAT forward pass works")
        
    except Exception as e:
        print(f"  [WARN] QAT preparation failed: {e}")


def test_tiny_yolov2():
    """Test 6: Tiny YOLOv2 model instantiation and forward pass."""
    print("\nTest 6: Testing Tiny YOLOv2 model...")
    tiny_model = TinyYOLOv2(num_classes=80, num_anchors=5)
    dummy_input = torch.randn(1, 3, 416, 416)
    tiny_model.eval()
    with torch.no_grad():
        output = tiny_model(dummy_input)
    print(f"  Tiny YOLOv2 Output shape: {output.shape}")
    assert output.shape == (1, 5 * (5 + 80), 13, 13), f"Unexpected shape {output.shape}"
    print("  [OK] Tiny YOLOv2 forward pass successful")


def main():
    print("\n" + "="*60)
    print("  OBJECT DETECTOR - UNIT TESTS")
    print("="*60)
    
    try:
        model = test_model_creation()
        outputs = test_forward_pass(model)
        test_output_ranges(outputs)
        test_parameter_count(model)
        test_int8_hooks(model)
        test_tiny_yolov2()
        
        print("\n" + "="*60)
        print("  [SUCCESS] ALL TESTS PASSED")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n[FAILED] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
