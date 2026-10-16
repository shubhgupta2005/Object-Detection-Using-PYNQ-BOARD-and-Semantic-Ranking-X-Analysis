#!/usr/bin/env python
"""
Full pipeline orchestrator with Real-Time Laptop Camera UI.

Features:
  1) Live Laptop Camera Mode (Default):
     - Displays live webcam stream.
     - On-screen interactive text box for task prompt input.
     - On-screen [SUBMIT] button (or press Enter).
     - Captures frame, runs Model-1 (YOLO11n) + Model-2 (Task-Aware Selector).
     - Marks the final selected object with bounding box directly on real-time camera view.
  2) CLI File Mode:
     - Run with `--image <path> --task <task>` for single-image batch processing.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from run_model2_with_image_output import main as run_yolo_and_model2

# Live tracking of the selected object: frames are downscaled to this width for
# the tracker, and the box is dropped after this many consecutive failed updates.
TRACK_WIDTH = 640
TRACK_MAX_MISSES = 15


def _normalize_cli_input(value: str) -> str:
    """Trim spaces and remove one layer of wrapping quotes."""
    cleaned = (value or "").strip()
    if len(cleaned) >= 2 and (
        (cleaned[0] == '"' and cleaned[-1] == '"')
        or (cleaned[0] == "'" and cleaned[-1] == "'")
    ):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _normalize_label(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


def _compute_accuracy(y_true: list[str], y_pred: list[str]) -> float:
    if not y_true:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if _normalize_label(t) == _normalize_label(p))
    return correct / len(y_true)


def _compute_confusion_matrix(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    labels = sorted({_normalize_label(v) for v in y_true + y_pred})
    index = {label: i for i, label in enumerate(labels)}

    matrix = [[0 for _ in labels] for _ in labels]
    for t, p in zip(y_true, y_pred):
        ti = index[_normalize_label(t)]
        pi = index[_normalize_label(p)]
        matrix[ti][pi] += 1

    return {
        "labels": labels,
        "matrix": matrix,
        "rows": "actual",
        "cols": "predicted",
    }


def run_full_pipeline(
    image_path: str,
    text_input: str,
    ground_truth: str | None = None,
    custom_model_dir: str = "Classifier_Custom",
    detections_json: str = "YOLO_custom/results/detections.json",
    output_json: str = "YOLO_custom/results/final_model2_output.json",
    output_image: str = "YOLO_custom/results/final_model2_output.jpg",
    detector: str = "yolo11n",
) -> dict:
    """
    Run end-to-end pipeline with a single call.
    """
    image_file = Path(image_path)
    if not image_file.exists():
        raise FileNotFoundError(f"Input image not found: {image_file}")

    repo_root = Path(__file__).resolve().parent.parent

    def _resolve_repo_path(path_str: str) -> str:
        p = Path(path_str)
        if p.is_absolute():
            return str(p)
        return str((repo_root / p).resolve())

    model_dir = Path(custom_model_dir)
    if not model_dir.exists():
        model_dir = (repo_root / custom_model_dir).resolve()

    detections_json = _resolve_repo_path(detections_json)
    output_json = _resolve_repo_path(output_json)
    output_image = _resolve_repo_path(output_image)

    payload = run_yolo_and_model2(
        image_path=str(image_file),
        task_text=text_input,
        custom_model_dir=str(model_dir),
        detections_json=detections_json,
        output_json=output_json,
        output_image=output_image,
        detector=detector,
    )

    if ground_truth:
        selected = str(payload.get("model2", {}).get("selected_object", {}).get("class_name", ""))
        y_true = [ground_truth]
        y_pred = [selected]
        payload["evaluation"] = {
            "ground_truth": ground_truth,
            "prediction": selected,
            "accuracy": _compute_accuracy(y_true, y_pred),
            "confusion_matrix": _compute_confusion_matrix(y_true, y_pred),
        }

        output_file = Path(output_json)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# Real-Time Laptop Camera Orchestrator UI
# ─────────────────────────────────────────────────────────────────────────────

class CameraOrchestratorUI:
    """
    Interactive Real-Time OpenCV GUI with live laptop camera, on-screen text input box,
    clickable Submit button, and visual marking of the final task-aware selected object.
    """

    def __init__(
        self,
        camera_index: int = 0,
        custom_model_dir: str = "Classifier_Custom",
        detections_json: str = "YOLO_custom/results/detections.json",
        output_json: str = "YOLO_custom/results/final_model2_output.json",
        output_image: str = "YOLO_custom/results/final_model2_output.jpg",
        detector: str = "yolo11n",
    ):
        self.camera_index = camera_index
        self.detector = detector
        self.custom_model_dir = custom_model_dir
        self.detections_json = detections_json
        self.output_json = output_json
        self.output_image = output_image

        self.window_name = "Assistive Robotics - Camera Orchestrator (Tiny YOLOv2 + Model 2)"
        self.typed_text = ""
        self.active_input = True
        self.current_selection: Optional[Dict[str, Any]] = None
        self.last_task = ""
        self.status_message = "Type a task prompt below & click SUBMIT or press ENTER"
        self.is_processing = False
        self.should_submit = False
        self.should_reset = False

        # Background inference + live tracking state
        self._result_lock = threading.Lock()
        self._pending_result: Optional[Tuple[np.ndarray, str, Optional[Dict[str, Any]], Optional[Exception]]] = None
        self._tracker = None
        self._track_scale = 1.0
        self._track_misses = 0
        self._track_busy = threading.Event()
        self._track_result: Optional[Tuple[Any, bool, Any]] = None

        # Button coordinates (updated dynamically based on frame width/height)
        self.input_box_rect = (0, 0, 0, 0)
        self.submit_btn_rect = (0, 0, 0, 0)
        self.reset_btn_rect = (0, 0, 0, 0)

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check text box click
            bx1, by1, bx2, by2 = self.input_box_rect
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                self.active_input = True
                return

            # Check Submit button click
            sx1, sy1, sx2, sy2 = self.submit_btn_rect
            if sx1 <= x <= sx2 and sy1 <= y <= sy2:
                self.should_submit = True
                return

            # Check Reset button click
            rx1, ry1, rx2, ry2 = self.reset_btn_rect
            if rx1 <= x <= rx2 and ry1 <= y <= ry2:
                self.should_reset = True
                return

            # Clicking elsewhere deselects focus
            self.active_input = False

    def _run_pipeline_job(self, image_path: str, captured_frame: np.ndarray, task_prompt: str) -> None:
        """Background worker: run Model 1 + Model 2 and hand the result to the UI loop."""
        try:
            payload = run_full_pipeline(
                image_path=image_path,
                text_input=task_prompt,
                custom_model_dir=self.custom_model_dir,
                detections_json=self.detections_json,
                output_json=self.output_json,
                output_image=self.output_image,
                detector=self.detector,
            )
            selected = payload.get("model2", {}).get("selected_object", {})
            with self._result_lock:
                self._pending_result = (captured_frame, task_prompt, selected, None)
        except Exception as exc:
            with self._result_lock:
                self._pending_result = (captured_frame, task_prompt, None, exc)

    def _consume_pending_result(self) -> None:
        with self._result_lock:
            pending, self._pending_result = self._pending_result, None
        if pending is None:
            return

        captured_frame, task_prompt, selected, error = pending
        self.is_processing = False

        if error is not None:
            self.status_message = f"Error during processing: {error}"
            print(f"[ERROR] Pipeline error: {error}")
            return

        if not (selected and selected.get("class_name") and isinstance(selected.get("bbox"), list)):
            self.current_selection = None
            self.status_message = "No matching candidate found for this task."
            print("[INFO] No matching candidate found for this task.")
            return

        self.current_selection = dict(selected)
        self.last_task = task_prompt
        cname = selected.get("class_name")
        score = selected.get("score", 0.0)
        self.status_message = f"Found: {cname.upper()} (score: {score*100:.1f}%) - tracking"
        print(f"[SUCCESS] Selected object: {cname} (confidence: {selected.get('confidence')})")
        self._start_tracking(captured_frame, selected["bbox"])

    def _start_tracking(self, frame: np.ndarray, bbox: List[float]) -> None:
        """Initialise a tracker on the frame the detection was made on."""
        h, w = frame.shape[:2]
        self._track_scale = TRACK_WIDTH / w if w > TRACK_WIDTH else 1.0
        x1, y1, x2, y2 = [float(v) for v in bbox]
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(w - 1.0, x2), min(h - 1.0, y2)
        s = self._track_scale
        roi = (int(x1 * s), int(y1 * s), max(1, int((x2 - x1) * s)), max(1, int((y2 - y1) * s)))

        self._tracker = cv2.TrackerMIL_create()
        self._tracker.init(self._scaled(frame), roi)
        self._track_misses = 0

    def _stop_tracking(self) -> None:
        self._tracker = None
        self._track_misses = 0
        with self._result_lock:
            self._track_result = None

    def _scaled(self, frame: np.ndarray) -> np.ndarray:
        if self._track_scale == 1.0:
            return frame
        return cv2.resize(frame, None, fx=self._track_scale, fy=self._track_scale, interpolation=cv2.INTER_AREA)

    def _track_step(self, tracker: Any, frame: np.ndarray) -> None:
        """Tracker worker: MIL costs ~70 ms/update, so keep it off the render loop."""
        try:
            ok, roi = tracker.update(self._scaled(frame))
        except cv2.error:
            ok, roi = False, None
        with self._result_lock:
            self._track_result = (tracker, ok, roi)
        self._track_busy.clear()

    def _update_tracking(self, frame: np.ndarray) -> None:
        """Move the selected box along with the object in the live feed."""
        if self._tracker is None or self.current_selection is None:
            return

        with self._result_lock:
            result, self._track_result = self._track_result, None
        if not self._track_busy.is_set():
            self._track_busy.set()
            threading.Thread(target=self._track_step, args=(self._tracker, frame.copy()), daemon=True).start()

        # Ignore results from a tracker that has since been replaced or stopped.
        if result is None or result[0] is not self._tracker:
            return
        _, ok, roi = result
        if ok:
            self._track_misses = 0
            x, y, bw, bh = roi
            s = self._track_scale
            self.current_selection["bbox"] = [x / s, y / s, (x + bw) / s, (y + bh) / s]
            return

        self._track_misses += 1
        if self._track_misses >= TRACK_MAX_MISSES:
            name = str(self.current_selection.get("class_name", "object")).upper()
            self._stop_tracking()
            self.current_selection = None
            self.status_message = f"Lost track of {name} - press SUBMIT to detect again"

    def _draw_corners(self, img, pt1, pt2, color, length=15, thickness=3):
        """Draw tech-style robotic corner accents on bounding boxes."""
        x1, y1 = pt1
        x2, y2 = pt2
        # Top-left
        cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
        cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
        # Top-right
        cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
        # Bottom-left
        cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
        cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
        # Bottom-right
        cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
        cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)

    def run(self) -> None:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            # Try camera index 1 if 0 is unavailable
            cap = cv2.VideoCapture(1)
            if not cap.isOpened():
                print(f"[ERROR] Could not open camera {self.camera_index} or 1. Please check webcam permissions.")
                return

        # Attempt to set standard resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        print("\n" + "=" * 70)
        print("  LIVE REAL-TIME CAMERA ORCHESTRATOR LAUNCHED")
        print("  - Type task prompt in on-screen box (e.g. 'sit comfortably', 'serve wine')")
        print("  - Click [SUBMIT] or press ENTER to capture & classify")
        print("  - Click [RESET] or press 'R' to clear selection")
        print("  - Press 'ESC' or 'Q' to quit")
        print("=" * 70 + "\n")

        repo_root = Path(__file__).resolve().parent.parent
        temp_capture_path = repo_root / "YOLO_custom" / "results" / "camera_capture.jpg"
        temp_capture_path.parent.mkdir(parents=True, exist_ok=True)

        cursor_blink = True
        last_blink_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] Failed to read frame from camera.")
                time.sleep(0.05)
                continue

            h, w, _ = frame.shape
            self._consume_pending_result()
            self._update_tracking(frame)
            display = frame.copy()

            # Blink cursor every 0.5s
            if time.time() - last_blink_time > 0.5:
                cursor_blink = not cursor_blink
                last_blink_time = time.time()

            # ─────────────────────────────────────────────────────────────
            # 1. Update UI Rectangles
            # ─────────────────────────────────────────────────────────────
            panel_height = 80
            panel_y1 = h - panel_height
            self.input_box_rect = (20, h - 68, w - 240, h - 14)
            self.submit_btn_rect = (w - 225, h - 68, w - 125, h - 14)
            self.reset_btn_rect = (w - 110, h - 68, w - 15, h - 14)

            # ─────────────────────────────────────────────────────────────
            # 2. Draw Active Selection (Marked Bounding Box)
            # ─────────────────────────────────────────────────────────────
            if self.current_selection and isinstance(self.current_selection.get("bbox"), list):
                bbox = self.current_selection["bbox"]
                if len(bbox) == 4:
                    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)

                    class_name = str(self.current_selection.get("class_name", "object")).upper()
                    conf = float(self.current_selection.get("confidence", 0.0))
                    score = float(self.current_selection.get("score", 0.0))

                    # Neon Green box & corners
                    neon_green = (0, 255, 60)
                    cv2.rectangle(display, (x1, y1), (x2, y2), neon_green, 2)
                    self._draw_corners(display, (x1, y1), (x2, y2), (0, 255, 255), length=20, thickness=3)

                    # Label badge
                    label_text = f" {class_name} | Conf: {conf*100:.0f}% | Match: {score*100:.0f}% "
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.55
                    (tw, th_text), _ = cv2.getTextSize(label_text, font, font_scale, 2)

                    badge_y1 = max(35, y1 - 25)
                    badge_y2 = badge_y1 + th_text + 10
                    badge_x2 = min(w - 5, x1 + tw + 10)

                    cv2.rectangle(display, (x1, badge_y1), (badge_x2, badge_y2), neon_green, -1)
                    cv2.putText(display, label_text, (x1 + 4, badge_y2 - 6), font, font_scale, (0, 0, 0), 2)

                    # Top-left card showing active selection details
                    card_w, card_h = 360, 65
                    overlay_card = display.copy()
                    cv2.rectangle(overlay_card, (15, 55), (15 + card_w, 55 + card_h), (20, 20, 25), -1)
                    cv2.addWeighted(overlay_card, 0.80, display, 0.20, 0, display)
                    cv2.rectangle(display, (15, 55), (15 + card_w, 55 + card_h), (0, 255, 60), 1)

                    cv2.putText(display, f"SELECTED: {class_name}", (25, 78), font, 0.60, (0, 255, 60), 2)
                    cv2.putText(display, f"Task: {self.last_task}", (25, 102), font, 0.50, (230, 230, 230), 1)

            # ─────────────────────────────────────────────────────────────
            # 3. Draw Header Bar
            # ─────────────────────────────────────────────────────────────
            header_h = 45
            header_overlay = display.copy()
            cv2.rectangle(header_overlay, (0, 0), (w, header_h), (15, 15, 20), -1)
            cv2.addWeighted(header_overlay, 0.85, display, 0.15, 0, display)
            cv2.line(display, (0, header_h), (w, header_h), (50, 50, 60), 1)

            cv2.putText(
                display,
                "Assistive Robotics | Task-Aware Vision System (Tiny YOLOv2 + Model 2)",
                (20, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 240, 255),
                2,
            )

            # Status message in header right
            (sw, _), _ = cv2.getTextSize(self.status_message, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.putText(
                display,
                self.status_message,
                (max(w - sw - 20, 480), 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 220, 180),
                1,
            )

            # ─────────────────────────────────────────────────────────────
            # 4. Draw Bottom Control Bar
            # ─────────────────────────────────────────────────────────────
            footer_overlay = display.copy()
            cv2.rectangle(footer_overlay, (0, panel_y1), (w, h), (18, 18, 22), -1)
            cv2.addWeighted(footer_overlay, 0.88, display, 0.12, 0, display)
            cv2.line(display, (0, panel_y1), (w, panel_y1), (60, 60, 75), 1)

            # Draw Text Box
            bx1, by1, bx2, by2 = self.input_box_rect
            border_color = (0, 220, 255) if self.active_input else (110, 110, 125)
            cv2.rectangle(display, (bx1, by1), (bx2, by2), (32, 32, 38), -1)
            cv2.rectangle(display, (bx1, by1), (bx2, by2), border_color, 2 if self.active_input else 1)

            if self.typed_text:
                text_show = self.typed_text + ("|" if (self.active_input and cursor_blink) else "")
                cv2.putText(
                    display,
                    text_show,
                    (bx1 + 12, by1 + 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )
            else:
                placeholder = "Type task prompt here (e.g. sit comfortably, serve wine)..."
                if self.active_input and cursor_blink:
                    placeholder = "|"
                cv2.putText(
                    display,
                    placeholder,
                    (bx1 + 12, by1 + 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (140, 140, 155),
                    1,
                )

            # Draw Submit Button
            sx1, sy1, sx2, sy2 = self.submit_btn_rect
            cv2.rectangle(display, (sx1, sy1), (sx2, sy2), (0, 175, 75), -1)
            cv2.rectangle(display, (sx1, sy1), (sx2, sy2), (0, 230, 100), 2)
            cv2.putText(
                display,
                "SUBMIT",
                (sx1 + 14, sy1 + 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
            )

            # Draw Reset Button
            rx1, ry1, rx2, ry2 = self.reset_btn_rect
            cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (45, 45, 170), -1)
            cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (80, 80, 220), 1)
            cv2.putText(
                display,
                "RESET",
                (rx1 + 16, ry1 + 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (240, 240, 240),
                2,
            )

            # Show window
            cv2.imshow(self.window_name, display)

            # ─────────────────────────────────────────────────────────────
            # 5. Handle Keyboard & Mouse Actions
            # ─────────────────────────────────────────────────────────────
            key = cv2.waitKey(20) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                print("\n[INFO] Exiting Camera Orchestrator.")
                break

            # Reset button or 'R' / 'C' key
            if self.should_reset or key in (ord("r"), ord("R"), ord("c"), ord("C")):
                self._stop_tracking()
                self.current_selection = None
                self.status_message = "Selection cleared. Live feed active."
                self.should_reset = False
                continue

            # Submit triggered by Enter key or button click
            if key in (10, 13) or self.should_submit:
                self.should_submit = False
                task_prompt = self.typed_text.strip()
                if not task_prompt:
                    self.status_message = "Please enter a task text first!"
                    continue

                if self.is_processing:
                    self.status_message = "Still processing previous request..."
                    continue

                # Save captured frame and run the models in a background thread
                # so the live feed keeps running while inference happens.
                cv2.imwrite(str(temp_capture_path), frame)
                print(f"\n[CAMERA] Captured frame saved to: {temp_capture_path}")
                print(f"[CAMERA] Running pipeline for task: '{task_prompt}'...")

                self._stop_tracking()
                self.current_selection = None
                self.is_processing = True
                self.status_message = f"Processing '{task_prompt}'..."
                threading.Thread(
                    target=self._run_pipeline_job,
                    args=(str(temp_capture_path), frame.copy(), task_prompt),
                    daemon=True,
                ).start()
                continue

            # Backspace key
            if key in (8, 127):
                self.typed_text = self.typed_text[:-1]
                continue

            # Character input
            if self.active_input and (32 <= key <= 126):
                # Limit length to fit in text box
                if len(self.typed_text) < 70:
                    self.typed_text += chr(key)

        cap.release()
        cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run full pipeline: YOLO11n detection + semantic task-aware object selection"
    )
    parser.add_argument("--image", type=str, default=None, help="Input image path (if omitted, launches camera UI)")
    parser.add_argument(
        "--text-input",
        type=str,
        default=None,
        help="Text-based task input (example: 'object used to serve wine')",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Alias for --text-input",
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        help="Explicitly launch real-time laptop camera mode with on-screen text input and submit button",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Webcam device index (default: 0)",
    )
    parser.add_argument(
        "--detector",
        choices=["yolo11n", "world", "classroom"],
        default="yolo11n",
        help="Model 1 backend: fixed COCO YOLO11n or open-vocabulary YOLO-World (detects whatever you type)",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=None,
        help="Optional expected class label for accuracy/confusion-matrix evaluation",
    )
    parser.add_argument(
        "--custom-model-dir",
        type=str,
        default="Classifier_Custom",
        help="Directory containing task_aware_selector.py",
    )
    parser.add_argument(
        "--detections-json",
        type=str,
        default="YOLO_custom/results/detections.json",
        help="Intermediate detections JSON output",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="YOLO_custom/results/final_model2_output.json",
        help="Final combined output JSON",
    )
    parser.add_argument(
        "--output-image",
        type=str,
        default="YOLO_custom/results/final_model2_output.jpg",
        help="Final annotated output image",
    )
    parser.add_argument(
        "--putty",
        action="store_true",
        help="Launch interactive PuTTY terminal bridge (synced with model inference and live output in PuTTY)",
    )
    parser.add_argument(
        "--putty-port",
        type=int,
        default=5000,
        help="Port for PuTTY terminal bridge (default: 5000)",
    )
    parser.add_argument(
        "--no-auto-putty",
        action="store_true",
        help="Do not auto-launch putty.exe window (wait for manual PuTTY connection)",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()

    # If --putty mode is requested, launch the PuTTY interactive bridge
    if args.putty:
        from putty_bridge import PuttyOrchestratorServer
        server = PuttyOrchestratorServer(
            host="0.0.0.0",
            port=args.putty_port,
            camera_index=args.camera_index,
            auto_launch_putty=not args.no_auto_putty,
            custom_model_dir=args.custom_model_dir,
        )
        server.start()
    # Determine whether to run Camera Mode or Single-Image File Mode:
    # If no image path is passed OR --camera is set, open the interactive Real-Time Laptop Camera UI!
    elif args.image is None or args.camera:
        app = CameraOrchestratorUI(
            camera_index=args.camera_index,
            custom_model_dir=args.custom_model_dir,
            detections_json=args.detections_json,
            output_json=args.output_json,
            output_image=args.output_image,
            detector=args.detector,
        )
        app.run()
    else:
        # File-based batch/CLI mode
        final_text = args.text_input if args.text_input else args.task
        if not final_text:
            final_text = _normalize_cli_input(input("Enter text-based task input: "))

        gt_label = _normalize_cli_input(args.ground_truth) if args.ground_truth else None

        result = run_full_pipeline(
            image_path=args.image,
            text_input=final_text,
            ground_truth=gt_label,
            custom_model_dir=args.custom_model_dir,
            detections_json=args.detections_json,
            output_json=args.output_json,
            output_image=args.output_image,
            detector=args.detector,
        )

        model2 = result.get("model2", {})
        selected = model2.get("selected_object", {})
        summary = {
            "task": result.get("task"),
            "selected_object": selected.get("class_name"),
            "confidence": selected.get("confidence"),
            "score": selected.get("score"),
            "output_json": args.output_json,
            "output_image": result.get("final_output_image", args.output_image),
        }
        if result.get("evaluation"):
            summary["evaluation"] = result["evaluation"]
        print(json.dumps(summary, indent=2))
