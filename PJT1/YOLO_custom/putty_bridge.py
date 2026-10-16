"""
PuTTY Interactive Terminal Bridge for Assistive Robotics Vision System.
Allows running and controlling Model 1 (Tiny YOLOv2) + Model 2 (Task-Aware Selector)
directly from a PuTTY terminal (via Raw TCP or Telnet), and displays all results,
ranked detections, and reasoning traces live inside the PuTTY console.
"""

from __future__ import annotations

import os
import sys
import time
import socket
import select
import threading
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

import cv2

# Add parent directories to sys.path
here = Path(__file__).resolve().parent
repo_root = here.parent.parent
sys.path.insert(0, str(here))
sys.path.insert(0, str(repo_root))

from run_model2_with_image_output import main as run_yolo_and_model2

# ANSI Color codes supported by PuTTY VT100 / xterm
ESC = "\033"
RESET = f"{ESC}[0m"
BOLD = f"{ESC}[1m"
DIM = f"{ESC}[2m"

BLACK = f"{ESC}[30m"
RED = f"{ESC}[31m"
GREEN = f"{ESC}[32m"
YELLOW = f"{ESC}[33m"
BLUE = f"{ESC}[34m"
MAGENTA = f"{ESC}[35m"
CYAN = f"{ESC}[36m"
WHITE = f"{ESC}[37m"

BG_DARK = f"{ESC}[40m"
BG_BLUE = f"{ESC}[44m"


class PuttyTerminalSession:
    """Handles an interactive PuTTY terminal session over a socket connection."""

    def __init__(
        self,
        client_sock: socket.socket,
        client_addr: tuple,
        camera_index: int = 0,
        custom_model_dir: str = "Classifier_Custom",
        detections_json: str = "YOLO_custom/results/detections.json",
        output_json: str = "YOLO_custom/results/final_model2_output.json",
        output_image: str = "YOLO_custom/results/final_model2_output.jpg",
    ):
        self.sock = client_sock
        self.addr = client_addr
        self.camera_index = camera_index
        self.custom_model_dir = custom_model_dir
        self.detections_json = detections_json
        self.output_json = output_json
        self.output_image = output_image

        self.last_result: Optional[Dict[str, Any]] = None
        self.last_task: str = ""
        self.running = True

    def send(self, text: str) -> None:
        """Send formatted text to the PuTTY terminal, converting newlines to CRLF."""
        try:
            # PuTTY raw mode expects CRLF (\r\n) for newlines
            formatted = text.replace("\r\n", "\n").replace("\n", "\r\n")
            self.sock.sendall(formatted.encode("utf-8", errors="replace"))
        except (socket.error, BrokenPipeError):
            self.running = False

    def send_line(self, text: str = "") -> None:
        """Send text followed by CRLF."""
        self.send(f"{text}\n")

    def read_line(self, prompt: str = "") -> str:
        """
        Interactive line reader that supports PuTTY raw typing, echo, and backspace.
        """
        if prompt:
            self.send(prompt)

        buffer = []
        while self.running:
            try:
                data = self.sock.recv(1)
                if not data:
                    self.running = False
                    return ""

                char_code = ord(data)

                # Enter key (\r or \n)
                if char_code in (10, 13):
                    self.send("\r\n")
                    break

                # Backspace (\x08 or \x7f)
                elif char_code in (8, 127):
                    if buffer:
                        buffer.pop()
                        # Move back, blank char, move back
                        self.send("\b \b")

                # Printable ASCII characters
                elif 32 <= char_code <= 126:
                    buffer.append(chr(char_code))
                    self.send(chr(char_code))

            except (socket.error, socket.timeout):
                self.running = False
                return ""

        return "".join(buffer).strip()

    def print_banner(self) -> None:
        """Display the welcome banner in PuTTY."""
        banner = f"""
{CYAN}╔══════════════════════════════════════════════════════════════════════════════╗{RESET}
{CYAN}║{BOLD}{YELLOW}     ASSISTIVE ROBOTICS | TASK-AWARE VISION SYSTEM (Tiny YOLOv2 + Model 2)    {RESET}{CYAN}║{RESET}
{CYAN}║{WHITE}                       PuTTY Interactive Terminal Console                     {RESET}{CYAN}║{RESET}
{CYAN}╚══════════════════════════════════════════════════════════════════════════════╝{RESET}
{GREEN}[+]{RESET} Connected from: {BOLD}{self.addr[0]}:{self.addr[1]}{RESET}
{GREEN}[+]{RESET} Model Pipeline: {BOLD}Model 1 (Tiny YOLOv2) -> Model 2 (Task-Aware Selector){RESET}
{GREEN}[+]{RESET} Default Input : {BOLD}Laptop Webcam (Device {self.camera_index}){RESET}

{BOLD}COMMANDS:{RESET}
  {YELLOW}<task description>{RESET} : Type ANY task (e.g. {CYAN}something used to drink water{RESET}) to capture & run
  {YELLOW}capture <task>{RESET}     : Capture camera frame and classify for given task
  {YELLOW}image <path> <task>{RESET}: Run pipeline on a saved image file
  {YELLOW}status{RESET}             : Show model and camera status
  {YELLOW}json{RESET}               : View raw JSON output of last result
  {YELLOW}clear{RESET} / {YELLOW}cls{RESET}        : Clear PuTTY screen
  {YELLOW}help{RESET}               : Show this help menu
  {YELLOW}exit{RESET} / {YELLOW}quit{RESET}        : Disconnect from PuTTY
"""
        self.send(banner)

    def capture_frame(self, save_path: Path) -> bool:
        """Capture one high-resolution frame from the webcam."""
        self.send_line(f"{CYAN}[*] Accessing camera device {self.camera_index}...{RESET}")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            # Try camera index 1
            cap = cv2.VideoCapture(1)

        if not cap.isOpened():
            self.send_line(f"{YELLOW}[!] Camera busy or not found. Checking for existing capture...{RESET}")
            if save_path.exists():
                self.send_line(f"{GREEN}[+] Using latest available frame: {save_path.name}{RESET}")
                return True
            self.send_line(f"{RED}[ERROR] No camera frame available.{RESET}")
            return False

        # Configure resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Warmup grab
        frame = None
        for _ in range(3):
            ret, tmp = cap.read()
            if ret and tmp is not None:
                frame = tmp
            time.sleep(0.05)

        cap.release()

        if frame is None:
            if save_path.exists():
                self.send_line(f"{YELLOW}[!] Frame grab timeout. Falling back to latest saved frame.{RESET}")
                return True
            self.send_line(f"{RED}[ERROR] Failed to capture image from camera.{RESET}")
            return False

        cv2.imwrite(str(save_path), frame)
        h, w, _ = frame.shape
        self.send_line(f"{GREEN}[+] Live frame captured successfully ({w}x{h}).{RESET}")
        return True

    def run_pipeline(self, image_path: str, task_text: str) -> None:
        """Run the full detection + task-aware ranking pipeline and print results in PuTTY."""
        self.send_line()
        self.send_line(f"{CYAN}{'═' * 70}{RESET}")
        self.send_line(f"{BOLD}[*] RUNNING VISION PIPELINE{RESET}")
        self.send_line(f"    Image : {image_path}")
        self.send_line(f"    Task  : {BOLD}{YELLOW}{task_text}{RESET}")
        self.send_line(f"{CYAN}{'═' * 70}{RESET}")

        t0 = time.time()

        try:
            payload = run_yolo_and_model2(
                image_path=image_path,
                task_text=task_text,
                custom_model_dir=self.custom_model_dir,
                detections_json=self.detections_json,
                output_json=self.output_json,
                output_image=self.output_image,
            )
        except Exception as e:
            self.send_line(f"{RED}[ERROR] Pipeline execution failed: {e}{RESET}")
            return

        elapsed = time.time() - t0
        self.last_result = payload
        self.last_task = task_text

        # Format and display the results in PuTTY
        self.display_result_table(payload, elapsed)

    def display_result_table(self, payload: Dict[str, Any], elapsed: float) -> None:
        """Print a structured, highlighted result table inside PuTTY terminal."""
        model2 = payload.get("model2", {})
        task = payload.get("task", self.last_task)
        classified_task = model2.get("classified_task", "N/A")
        task_sim = float(model2.get("task_class_similarity", 0.0))

        selected = model2.get("selected_object", {}) or {}
        class_name = str(selected.get("class_name", "None")).upper()
        conf = float(selected.get("confidence", 0.0))
        score = float(selected.get("score", 0.0))
        cos_sim = float(selected.get("cos_sim", 0.0))
        crop_sim = float(selected.get("image_cos_sim", 0.0))
        bbox = selected.get("bbox", [])

        ranked_objects = model2.get("ranked_objects", [])
        detections = payload.get("detections", [])

        self.send_line()
        self.send_line(f"{GREEN}╔══════════════════════════════════════════════════════════════════════════════╗{RESET}")
        self.send_line(f"{GREEN}║{BOLD}{WHITE}                         FINAL MODEL INFERENCE RESULT                         {RESET}{GREEN}║{RESET}")
        self.send_line(f"{GREEN}╠══════════════════════════════════════════════════════════════════════════════╣{RESET}")
        self.send_line(f"{GREEN}║{RESET}  Task Prompt      : {BOLD}{YELLOW}{task:<54}{RESET} {GREEN}║{RESET}")
        self.send_line(f"{GREEN}║{RESET}  Classified Task  : {CYAN}{classified_task:<40}{RESET} (Sim: {task_sim*100:.1f}%) {GREEN}║{RESET}")
        self.send_line(f"{GREEN}║{RESET}  Inference Time   : {WHITE}{elapsed:.2f} seconds{RESET}{' ' * 42} {GREEN}║{RESET}")
        self.send_line(f"{GREEN}╠══════════════════════════════════════════════════════════════════════════════╣{RESET}")
        self.send_line(f"{GREEN}║{RESET}  {BOLD}SELECTED OBJECT  :{RESET} {BOLD}{GREEN}★ {class_name:<50}{RESET} {GREEN}║{RESET}")
        self.send_line(f"{GREEN}║{RESET}  Detector Conf    : {BOLD}{conf*100:6.2f}%{RESET}{' ' * 49} {GREEN}║{RESET}")
        self.send_line(f"{GREEN}║{RESET}  Semantic Match   : {BOLD}{cos_sim*100:6.2f}%{RESET}{' ' * 49} {GREEN}║{RESET}")
        self.send_line(f"{GREEN}║{RESET}  Crop Similarity  : {BOLD}{crop_sim*100:6.2f}%{RESET}{' ' * 49} {GREEN}║{RESET}")
        self.send_line(f"{GREEN}║{RESET}  Composite Score  : {BOLD}{YELLOW}{score:6.4f}{RESET}{' ' * 50} {GREEN}║{RESET}")

        if len(bbox) == 4:
            bbox_str = f"[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]"
            self.send_line(f"{GREEN}║{RESET}  Bounding Box     : {CYAN}{bbox_str:<54}{RESET} {GREEN}║{RESET}")

        self.send_line(f"{GREEN}╠══════════════════════════════════════════════════════════════════════════════╣{RESET}")
        self.send_line(f"{GREEN}║{RESET}  {BOLD}RANKED CANDIDATES ({len(ranked_objects)} detected){RESET}:{' ' * 43} {GREEN}║{RESET}")

        for idx, obj in enumerate(ranked_objects, 1):
            obj_name = str(obj.get("class_name", "")).upper()
            obj_score = float(obj.get("score", 0.0))
            obj_conf = float(obj.get("confidence", 0.0))
            obj_sim = float(obj.get("cos_sim", 0.0))
            marker = "★" if idx == 1 else " "
            row = f"   {marker} #{idx:<2} {obj_name:<12} | Score: {obj_score:.4f} | Conf: {obj_conf*100:5.1f}% | Match: {obj_sim*100:5.1f}%"
            self.send_line(f"{GREEN}║{RESET} {row:<68} {GREEN}║{RESET}")

        self.send_line(f"{GREEN}╠══════════════════════════════════════════════════════════════════════════════╣{RESET}")
        self.send_line(f"{GREEN}║{RESET}  {BOLD}SAVED ARTIFACTS:{RESET}{' ' * 57} {GREEN}║{RESET}")
        self.send_line(f"{GREEN}║{RESET}  JSON : {WHITE}YOLO_custom/results/final_model2_output.json{RESET}{' ' * 19} {GREEN}║{RESET}")
        self.send_line(f"{GREEN}║{RESET}  IMAGE: {WHITE}YOLO_custom/results/final_model2_output.jpg{RESET}{' ' * 20} {GREEN}║{RESET}")
        self.send_line(f"{GREEN}╚══════════════════════════════════════════════════════════════════════════════╝{RESET}")
        self.send_line()

    def handle_status(self) -> None:
        """Display system status in PuTTY."""
        repo = repo_root
        cam_file = repo / "YOLO_custom" / "results" / "camera_capture.jpg"
        out_json = repo / "YOLO_custom" / "results" / "final_model2_output.json"
        out_img = repo / "YOLO_custom" / "results" / "final_model2_output.jpg"

        self.send_line(f"\n{BOLD}{CYAN}=== SYSTEM & MODEL STATUS ==={RESET}")
        self.send_line(f"  Model 1 Backend : {GREEN}Tiny YOLOv2 (Upgraded Ultralytics Architecture){RESET}")
        self.send_line(f"  Model 2 Backend : {GREEN}Task-Aware Semantic Selector (CLIP ViT-B/32){RESET}")
        self.send_line(f"  Camera Device   : {YELLOW}Device Index {self.camera_index}{RESET}")
        self.send_line(f"  Last Capture    : {'[EXISTS]' if cam_file.exists() else '[NONE]'}")
        self.send_line(f"  Last JSON Output: {'[EXISTS]' if out_json.exists() else '[NONE]'}")
        self.send_line(f"  Last Image Out  : {'[EXISTS]' if out_img.exists() else '[NONE]'}")
        if self.last_task:
            self.send_line(f"  Last Run Task   : {BOLD}{self.last_task}{RESET}")
        self.send_line()

    def handle_json(self) -> None:
        """Print the raw JSON output in PuTTY."""
        if not self.last_result:
            json_file = repo_root / "YOLO_custom" / "results" / "final_model2_output.json"
            if json_file.exists():
                import json
                try:
                    with json_file.open("r", encoding="utf-8") as f:
                        self.last_result = json.load(f)
                except Exception:
                    pass

        if not self.last_result:
            self.send_line(f"{YELLOW}[!] No results to display yet. Run a task first.{RESET}")
            return

        import json
        formatted_json = json.dumps(self.last_result, indent=2)
        self.send_line(f"\n{CYAN}--- RAW OUTPUT JSON ---{RESET}")
        self.send_line(formatted_json)
        self.send_line(f"{CYAN}--- END JSON ---{RESET}\n")

    def run(self) -> None:
        """Main interaction loop with the PuTTY terminal client."""
        self.print_banner()

        repo = repo_root
        temp_cam = repo / "YOLO_custom" / "results" / "camera_capture.jpg"

        while self.running:
            prompt = f"{BOLD}{GREEN}PuTTY-Orchestrator{RESET}> "
            line = self.read_line(prompt)
            if not self.running:
                break

            if not line:
                continue

            cmd_lower = line.lower().strip()

            if cmd_lower in ("exit", "quit", "q"):
                self.send_line(f"\n{YELLOW}[+] Closing PuTTY session. Goodbye!{RESET}\n")
                break

            elif cmd_lower in ("clear", "cls"):
                # VT100 clear screen and move cursor to (1,1)
                self.send("\033[2J\033[H")
                self.print_banner()

            elif cmd_lower in ("help", "?"):
                self.print_banner()

            elif cmd_lower == "status":
                self.handle_status()

            elif cmd_lower == "json":
                self.handle_json()

            elif cmd_lower.startswith("image "):
                # Syntax: image <path> <task text>
                parts = line[6:].strip().split(maxsplit=1)
                if len(parts) < 2:
                    self.send_line(f"{RED}[ERROR] Usage: image <image_path> <task description>{RESET}")
                else:
                    img_path, task_desc = parts
                    self.run_pipeline(image_path=img_path, task_text=task_desc)

            elif cmd_lower.startswith("capture "):
                # Syntax: capture <task text>
                task_desc = line[8:].strip()
                if not task_desc:
                    self.send_line(f"{RED}[ERROR] Please provide a task description (e.g. capture drink water){RESET}")
                else:
                    if self.capture_frame(temp_cam):
                        self.run_pipeline(image_path=str(temp_cam), task_text=task_desc)

            else:
                # Default convenient behavior:
                # Any text typed directly is treated as the task prompt!
                task_desc = line.strip()
                self.send_line(f"{CYAN}[*] Task received: '{task_desc}'{RESET}")
                if self.capture_frame(temp_cam):
                    self.run_pipeline(image_path=str(temp_cam), task_text=task_desc)

        try:
            self.sock.close()
        except Exception:
            pass


class PuttyOrchestratorServer:
    """
    TCP Socket Server that listens for PuTTY terminal connections (Raw or Telnet).
    Can automatically launch PuTTY on Windows.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5000,
        camera_index: int = 0,
        auto_launch_putty: bool = True,
        custom_model_dir: str = "Classifier_Custom",
    ):
        self.host = host
        self.port = port
        self.camera_index = camera_index
        self.auto_launch_putty = auto_launch_putty
        self.custom_model_dir = custom_model_dir
        self.server_sock: Optional[socket.socket] = None
        self.running = False

    def launch_putty_client(self) -> None:
        """Launch the PuTTY GUI client on Windows connected via Raw TCP to localhost."""
        putty_exe = Path(r"C:\Program Files\PuTTY\putty.exe")
        if not putty_exe.exists():
            putty_exe = Path(r"C:\Program Files (x86)\PuTTY\putty.exe")

        if putty_exe.exists():
            cmd = [str(putty_exe), "-raw", "127.0.0.1", "-P", str(self.port)]
            print(f"[+] Launching PuTTY terminal: {' '.join(cmd)}")
            subprocess.Popen(cmd)
        else:
            print("[WARN] PuTTY executable not found in default path. Please open PuTTY manually:")
            print(f"       Host: 127.0.0.1  |  Port: {self.port}  |  Connection type: Raw")

    def start(self) -> None:
        """Start listening for PuTTY connections."""
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            self.server_sock.bind((self.host, self.port))
        except OSError as e:
            print(f"[ERROR] Could not bind to {self.host}:{self.port} ({e})")
            print("        Another process may already be using this port. Try another port with --port.")
            return

        self.server_sock.listen(5)
        self.running = True

        print("\n" + "=" * 70)
        print("  PUTTY ORCHESTRATOR BRIDGE SERVER ACTIVE")
        print(f"  - Listening on: {self.host}:{self.port}")
        print(f"  - Connection Type: Raw TCP or Telnet")
        print(f"  - Connect PuTTY to: 127.0.0.1 (Port {self.port})")
        print("=" * 70 + "\n")

        # Automatically launch PuTTY GUI client if requested
        if self.auto_launch_putty:
            threading.Thread(target=self.launch_putty_client, daemon=True).start()

        try:
            while self.running:
                client_sock, client_addr = self.server_sock.accept()
                print(f"[+] New PuTTY terminal connection from {client_addr[0]}:{client_addr[1]}")

                session = PuttyTerminalSession(
                    client_sock=client_sock,
                    client_addr=client_addr,
                    camera_index=self.camera_index,
                    custom_model_dir=self.custom_model_dir,
                )

                # Handle session in worker thread so multiple sessions can connect if needed
                thread = threading.Thread(target=session.run, daemon=True)
                thread.start()

        except KeyboardInterrupt:
            print("\n[+] Shutting down PuTTY Orchestrator server...")
        finally:
            self.stop()

    def stop(self) -> None:
        self.running = False
        if self.server_sock:
            try:
                self.server_sock.close()
            except Exception:
                pass


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="PuTTY Interactive Terminal Bridge for Vision System")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Listen address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Listen port (default: 5000)")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument(
        "--no-auto-putty",
        action="store_true",
        help="Do not automatically launch putty.exe GUI (wait for manual connection)",
    )
    args = parser.parse_args()

    server = PuttyOrchestratorServer(
        host=args.host,
        port=args.port,
        camera_index=args.camera_index,
        auto_launch_putty=not args.no_auto_putty,
    )
    server.start()


if __name__ == "__main__":
    main()
