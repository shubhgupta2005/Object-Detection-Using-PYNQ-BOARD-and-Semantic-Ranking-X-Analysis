#!/usr/bin/env python
"""
Local web dashboard for the Task-Aware Vision System.

Serves a single-page UI on http://localhost:8000 that captures a webcam frame
in the browser, sends it here with the user's task prompt, runs Model 1
(detector) + Model 2 (task-aware selector) and returns the ranked result.

Uses only the Python standard library on top of the existing pipeline.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
YOLO_DIR = REPO_ROOT / "YOLO_custom"
sys.path.insert(0, str(YOLO_DIR))

import run_model2_with_image_output as rm  # noqa: E402
from open_vocab import OBJECT_USES, rank_vocabulary  # noqa: E402
from orchestrator import run_full_pipeline  # noqa: E402

STATIC_DIR = HERE / "static"
CAPTURE_PATH = YOLO_DIR / "results" / "web_capture.jpg"
CLIP_WEIGHTS = YOLO_DIR / "weights" / "clip" / "ViT-B-32.pt"
CLASSROOM_WEIGHTS = YOLO_DIR / "classroom_yolo11n.pt"
# A complete ViT-B/32 checkpoint is ~338 MB; anything smaller is a partial download.
CLIP_WEIGHTS_MIN_BYTES = 330_000_000
MAX_BODY_BYTES = 15 * 1024 * 1024

# Review-II demo videos, served straight from the review folder by a fixed key.
REVIEW_DIR = REPO_ROOT.parent / "Project Review 2"
MEDIA = {
    "pynq-board.mp4": REVIEW_DIR / "PYNQ BOARD VIDEO.mp4",
    "program.mp4": REVIEW_DIR / "program video.mp4",
    "detection-run.mp4": REVIEW_DIR / "All object detection video.mp4",
    "detection-readings.mp4": REVIEW_DIR / "clear readings  All object detection video.mp4",
}
MEDIA_CHUNK = 256 * 1024

# Review-III documents. Browsers can only display PDFs, so each document is viewed from a PDF export
# placed in the same folder (newest *.pdf whose name contains `keyword`); the original is a download.
REVIEW3_DIR = REPO_ROOT.parent / "Review 3"
DOCS = {
    "report": {"title": "Final project report", "keyword": "report", "exclude": "review-iii",
               "source": REVIEW3_DIR / "Final_Report_Task_Aware_Object_Identification (1).docx"},
    "slides": {"title": "Review-III presentation", "keyword": "review", "exclude": "report",
               "source": REVIEW3_DIR / "Review-III_Task_Aware_Object_Identification.pptx"},
}
DOC_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}

STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".json": "application/json",
}

# Every dashboard result is kept on disk: saved_results/<id>/{input.jpg, annotated.jpg, result.json}.
SAVE_DIR = REPO_ROOT / "saved_results"
SAVED_FILES = ("input.jpg", "annotated.jpg")
HISTORY_LIMIT = 50

# PYNQ-Z2 board detection. Over the network the board answers on SSH / Jupyter at its
# static IP; over USB its FTDI FT2232 USB-UART (vendor id 0403) shows up as a COM port.
BOARD_IPS = ["192.168.2.99"]
BOARD_PORTS = {22: "SSH", 9090: "Jupyter", 80: "HTTP"}
BOARD_POLL_SECONDS = 4
# When True, tests can only run while the board is connected. TEMPORARILY OFF for testing: set back to True.
REQUIRE_BOARD = False
_board_lock = threading.Lock()
_board_state: Dict[str, Any] = {"connected": False, "checked_at": None, "network": None, "usb": [],
                                "required": REQUIRE_BOARD}

# Offline speech-to-text for the mic button: Whisper runs on this computer, no internet needed.
# It decodes the browser's audio with ffmpeg, so both must be present.
WHISPER_MODEL = "base.en"
VOICE_PATH = YOLO_DIR / "results" / "web_voice.webm"
_whisper = None
_whisper_lock = threading.Lock()

# The models are loaded once and are not thread-safe; run one request at a time.
_pipeline_lock = threading.Lock()
_ready = threading.Event()


def _world_available() -> bool:
    return CLIP_WEIGHTS.exists() and CLIP_WEIGHTS.stat().st_size >= CLIP_WEIGHTS_MIN_BYTES


def _detectors() -> Dict[str, bool]:
    return {"yolo11n": True, "world": _world_available(), "classroom": CLASSROOM_WEIGHTS.exists()}


def _voice_available() -> bool:
    return importlib.util.find_spec("whisper") is not None and shutil.which("ffmpeg") is not None


def _load_whisper():
    global _whisper
    if _whisper is None:
        import whisper
        _whisper = whisper.load_model(WHISPER_MODEL)
    return _whisper


def _transcribe(audio: bytes) -> str:
    with _whisper_lock:
        model = _load_whisper()
        VOICE_PATH.parent.mkdir(parents=True, exist_ok=True)
        VOICE_PATH.write_bytes(audio)
        return model.transcribe(str(VOICE_PATH), fp16=False, language="en")["text"].strip()


def _warm_up() -> None:
    """Load every model once at startup so the first click in the UI is fast."""
    from PIL import Image

    warm_path = YOLO_DIR / "results" / "web_warmup.jpg"
    warm_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (640, 480), (90, 90, 90)).save(warm_path)
    detectors = [d for d, ok in _detectors().items() if ok]
    started = time.perf_counter()
    try:
        for detector in detectors:
            with _pipeline_lock:
                run_full_pipeline(
                    image_path=str(warm_path),
                    text_input="something to drink from",
                    detections_json="YOLO_custom/results/web_detections.json",
                    output_json="YOLO_custom/results/web_model2_output.json",
                    output_image="YOLO_custom/results/web_model2_output.jpg",
                    detector=detector,
                )
        print(f"[web] Models warmed up ({', '.join(detectors)}) in {time.perf_counter() - started:.1f}s")
    except Exception as exc:
        print(f"[web] Warm-up failed (requests will load models lazily): {exc!r}", file=sys.stderr)
    finally:
        _ready.set()
    if _voice_available():
        try:
            with _whisper_lock:
                _load_whisper()
            print(f"[web] Voice input ready (Whisper {WHISPER_MODEL}, offline)")
        except Exception as exc:
            print(f"[web] Whisper failed to load (mic falls back to the browser): {exc!r}", file=sys.stderr)


def _probe_board_network() -> Dict[str, Any] | None:
    for host in BOARD_IPS:
        open_ports = []
        for port, name in BOARD_PORTS.items():
            try:
                with socket.create_connection((host, port), timeout=0.6):
                    open_ports.append({"port": port, "service": name})
            except OSError:
                continue
        if open_ports:
            return {"host": host, "ports": open_ports}
    return None


def _probe_board_usb() -> list:
    """COM ports of FTDI USB-UARTs (the PYNQ-Z2's micro-USB port). Windows only."""
    if sys.platform != "win32":
        return []
    query = (
        "Get-CimInstance Win32_PnPEntity -Filter \"DeviceID LIKE '%VID_0403%'\" "
        "| Where-Object { $_.Name -match '\\(COM\\d+\\)' } | ForEach-Object { $_.Name }"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", query],
            capture_output=True, text=True, timeout=8,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    ports = []
    for name in filter(None, (line.strip() for line in out.splitlines())):
        match = re.search(r"\((COM\d+)\)", name)
        ports.append({"port": match.group(1) if match else "?", "name": name})
    return ports


def _check_board() -> Dict[str, Any]:
    network, usb = _probe_board_network(), _probe_board_usb()
    state = {
        "connected": bool(network or usb),
        "network": network,
        "usb": usb,
        "checked_at": time.strftime("%H:%M:%S"),
    }
    with _board_lock:
        _board_state.update(state)
    return state


def _board_watcher() -> None:
    while True:
        try:
            _check_board()
        except Exception as exc:  # never let the watcher thread die
            print(f"[web] board check failed: {exc!r}", file=sys.stderr)
        time.sleep(BOARD_POLL_SECONDS)


def _doc_pdf(doc: Dict[str, Any]) -> Path | None:
    pdfs = [p for p in REVIEW3_DIR.glob("*.pdf")
            if doc["keyword"] in p.name.lower() and doc["exclude"] not in p.name.lower()]
    return max(pdfs, key=lambda p: p.stat().st_mtime, default=None)


def _docs_list() -> list:
    docs = []
    for key, doc in DOCS.items():
        pdf, source = _doc_pdf(doc), doc["source"]
        docs.append({
            "key": key,
            "title": doc["title"],
            "pdf": pdf.name if pdf else None,
            "pdf_size": pdf.stat().st_size if pdf else None,
            "source": source.name if source.is_file() else None,
            "source_size": source.stat().st_size if source.is_file() else None,
            "source_ext": source.suffix.lstrip("."),
        })
    return docs


def _save_result(result: Dict[str, Any], image_bytes: bytes) -> tuple:
    """Write one dashboard run to saved_results/<id>/ and return (id, saved_at)."""
    from PIL import Image, ImageDraw

    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    slug = re.sub(r"[^a-z0-9]+", "-", result["task"].lower()).strip("-")[:40] or "task"
    save_id = f"{stamp}_{int(time.time() * 1000) % 1000:03d}_{slug}"
    folder = SAVE_DIR / save_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "input.jpg").write_bytes(image_bytes)

    image = Image.open(folder / "input.jpg").convert("RGB")
    draw = ImageDraw.Draw(image)
    width = max(2, image.width // 320)
    selected = result.get("selected") or {}
    for det in result.get("ranked", []):
        if det.get("bbox") != selected.get("bbox"):
            draw.rectangle(det["bbox"], outline=(139, 152, 171), width=width)
    if selected.get("bbox"):
        draw.rectangle(selected["bbox"], outline=(163, 230, 53), width=width * 2)
        label = f"{selected['class_name']}  {selected.get('score', 0) * 100:.0f}%"
        x1, y1 = selected["bbox"][0], max(0, selected["bbox"][1] - 22)
        draw.rectangle([x1, y1, x1 + 9 * len(label) + 10, y1 + 20], fill=(163, 230, 53))
        draw.text((x1 + 5, y1 + 4), label, fill=(11, 18, 0))
    image.save(folder / "annotated.jpg", quality=90)

    saved_at = time.strftime("%Y-%m-%d %H:%M:%S")
    record = {**result, "id": save_id, "saved_at": saved_at}
    (folder / "result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return save_id, saved_at


def _load_history(limit: int = HISTORY_LIMIT) -> list:
    """Saved runs, newest first."""
    if not SAVE_DIR.exists():
        return []
    records = []
    for folder in sorted((p for p in SAVE_DIR.iterdir() if p.is_dir()), reverse=True)[:limit]:
        try:
            records.append(json.loads((folder / "result.json").read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue  # half-written or hand-edited folder: skip it
    return records


def _vocabulary_scores(task: str, top: int = 8) -> list:
    """Top task-to-object affordance matches, shown in the dashboard's terminal."""
    clip_model, _, tokenizer, device = rm._get_clip(rm._load_custom_selector_module("Classifier_Custom"))
    ranked = rank_vocabulary(task, clip_model, tokenizer, device)
    return [{"name": name, "use": OBJECT_USES[name], "sim": round(sim, 4)} for name, sim in ranked[:top]]


def _image_size(path: Path) -> list:
    from PIL import Image

    with Image.open(path) as img:
        return list(img.size)


def _run_detection(body: Dict[str, Any]) -> Dict[str, Any]:
    task = str(body.get("task", "")).strip()
    if not task:
        raise ValueError("Task prompt is empty.")

    detector = body.get("detector", "yolo11n")
    if detector not in ("yolo11n", "world", "classroom"):
        raise ValueError(f"Unknown detector: {detector}")
    if not _detectors()[detector]:
        if detector == "classroom":
            raise ValueError("Classroom model not trained yet: run finetune/train_classroom.py.")
        raise ValueError("YOLO-World needs the CLIP weights in YOLO_custom/weights/clip/ViT-B-32.pt.")

    image_data = str(body.get("image", ""))
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]
    image_bytes = base64.b64decode(image_data, validate=True)
    if not image_bytes:
        raise ValueError("No image received.")

    with _pipeline_lock:
        CAPTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CAPTURE_PATH.write_bytes(image_bytes)

        started = time.perf_counter()
        payload = run_full_pipeline(
            image_path=str(CAPTURE_PATH),
            text_input=task,
            detections_json="YOLO_custom/results/web_detections.json",
            output_json="YOLO_custom/results/web_model2_output.json",
            output_image="YOLO_custom/results/web_model2_output.jpg",
            detector=detector,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000

        vocab_scores = _vocabulary_scores(task)
        image_size = _image_size(CAPTURE_PATH)

    model2 = payload.get("model2", {})
    result = {
        "vocab_scores": vocab_scores,
        "image_size": image_size,
        "task": task,
        "detector": detector,
        "vocabulary": payload.get("vocabulary"),
        "classified_task": model2.get("classified_task"),
        "selected": model2.get("selected_object") or None,
        "ranked": model2.get("ranked_objects", []),
        "detections": payload.get("detections", []),
        "status": model2.get("status", "ok"),
        "elapsed_ms": round(elapsed_ms),
    }
    if body.get("save") is False:  # e.g. build_site_data.py latency benchmarks
        return result
    try:
        result["id"], result["saved_at"] = _save_result(result, image_bytes)
    except Exception as exc:  # saving is a convenience; never fail the request over it
        print(f"[web] could not save result: {exc!r}", file=sys.stderr)
    return result


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "TaskAwareVision/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(f"[web] {self.address_string()} {fmt % args}\n")

    def _send_json(self, status: HTTPStatus, data: Dict[str, Any]) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_media(self, path: Path, content_type: str = "video/mp4", disposition: str | None = None) -> None:
        """Stream a file (MP4 by default) with HTTP Range support so the browser can seek."""
        size = path.stat().st_size
        start, end = 0, size - 1
        range_header = self.headers.get("Range", "")
        if range_header.startswith("bytes="):
            first, _, last = range_header[6:].split(",")[0].partition("-")
            try:
                if first:
                    start = int(first)
                    end = min(int(last), size - 1) if last else size - 1
                else:  # suffix range: last N bytes
                    start = max(0, size - int(last))
            except ValueError:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            if start > end or start >= size:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(HTTPStatus.OK)

        self.send_header("Content-Type", content_type)
        if disposition:
            self.send_header("Content-Disposition", f'{disposition}; filename="{path.name}"')
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()

        remaining = end - start + 1
        try:
            with path.open("rb") as fh:
                fh.seek(start)
                while remaining > 0:
                    chunk = fh.read(min(MEDIA_CHUNK, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            pass  # the browser cancels requests whenever the user seeks

    def do_GET(self) -> None:
        route = self.path.split("?", 1)[0]
        if route == "/api/board":
            with _board_lock:
                state = dict(_board_state)
            self._send_json(HTTPStatus.OK, state)
            return
        if route == "/api/board/check":
            self._send_json(HTTPStatus.OK, _check_board())
            return
        if route == "/api/history":
            self._send_json(HTTPStatus.OK, {"items": _load_history()})
            return
        if route.startswith("/saved/"):
            # /saved/<id>/input.jpg or /saved/<id>/annotated.jpg
            parts = route[len("/saved/"):].split("/")
            if (len(parts) != 2 or parts[1] not in SAVED_FILES
                    or not re.fullmatch(r"[\w.-]+", parts[0]) or parts[0] in (".", "..")):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            target = SAVE_DIR / parts[0] / parts[1]
            if not target.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            body = target.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=86400")
            self.end_headers()
            self.wfile.write(body)
            return
        if route.startswith("/review/"):
            # Review-II board outputs: /review/<1-14>.jpg -> final_model2_output_<n>.jpg
            stem = route[len("/review/"):].removesuffix(".jpg")
            target = REVIEW_DIR / "Result" / f"final_model2_output_{stem}.jpg"
            if not stem.isdigit() or not target.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            body = target.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=3600")
            self.end_headers()
            self.wfile.write(body)
            return
        if route == "/api/docs":
            self._send_json(HTTPStatus.OK, {"docs": _docs_list()})
            return
        if route.startswith("/docs/"):
            # /docs/<key>.pdf shows the PDF in the page; /docs/<key>/download sends the original file.
            name = route[len("/docs/"):]
            key, download = (name[:-len("/download")], True) if name.endswith("/download") else (name.removesuffix(".pdf"), False)
            doc = DOCS.get(key)
            path = (doc["source"] if download else _doc_pdf(doc)) if doc else None
            if path is None or not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._send_media(path, DOC_TYPES.get(path.suffix.lower(), "application/octet-stream"),
                             "attachment" if download else "inline")
            return
        if route.startswith("/media/"):
            media = MEDIA.get(route[len("/media/"):])
            if media is None or not media.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._send_media(media)
            return
        if route == "/api/status":
            self._send_json(HTTPStatus.OK, {"ready": _ready.is_set(), "detectors": _detectors(),
                                             "voice": _voice_available()})
            return

        rel = "index.html" if route in ("/", "") else route.lstrip("/")
        target = (STATIC_DIR / rel).resolve()
        if STATIC_DIR not in target.parents or not target.is_file() or target.suffix not in STATIC_TYPES:
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        body = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", STATIC_TYPES[target.suffix])
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path == "/api/transcribe":
            self._handle_transcribe()
            return
        if self.path != "/api/detect":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        # Runs are only allowed while the PYNQ-Z2 is attached; re-probe in case it was just plugged in.
        with _board_lock:
            connected = _board_state["connected"]
        if REQUIRE_BOARD and not connected and not _check_board()["connected"]:
            self._send_json(HTTPStatus.SERVICE_UNAVAILABLE,
                            {"error": "PYNQ-Z2 board not connected. Connect the board to run the test."})
            return

        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY_BYTES:
            self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "Request body missing or too large."})
            return

        try:
            body = json.loads(self.rfile.read(length))
            result = _run_detection(body)
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except Exception as exc:  # surface pipeline failures to the UI
            print(f"[web] pipeline error: {exc!r}", file=sys.stderr)
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"Pipeline error: {exc}"})
            return

        self._send_json(HTTPStatus.OK, result)


    def _handle_transcribe(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY_BYTES:
            self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "Recording missing or too large."})
            return
        if not _voice_available():
            self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "Offline voice needs openai-whisper and ffmpeg."})
            return
        try:
            text = _transcribe(self.rfile.read(length))
        except Exception as exc:
            print(f"[web] transcription error: {exc!r}", file=sys.stderr)
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"Could not transcribe: {exc}"})
            return
        self._send_json(HTTPStatus.OK, {"text": text})


def main() -> None:
    parser = argparse.ArgumentParser(description="Task-Aware Vision System web dashboard")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically")
    parser.add_argument("--board-ip", action="append", default=[],
                        help="PYNQ-Z2 IP address to watch (repeatable; default 192.168.2.99)")
    args = parser.parse_args()
    if args.board_ip:
        BOARD_IPS[:] = args.board_ip
    threading.Thread(target=_board_watcher, daemon=True).start()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), DashboardHandler)
    url = f"http://localhost:{args.port}"
    print(f"[web] Dashboard running at {url}  (Ctrl+C to stop)")
    threading.Thread(target=_warm_up, daemon=True).start()
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] Stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
