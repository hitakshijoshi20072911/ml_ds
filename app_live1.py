import argparse
import os
import threading
import time
from pathlib import Path

import cv2
from inference_sdk import InferenceConfiguration, InferenceHTTPClient

MODEL_ID = "road-damage-detection-v2-xg6by/3"
API_URL = "https://serverless.roboflow.com"
DEFAULT_CONFIDENCE = 0.30
DEFAULT_IOU = 0.45


def get_client(confidence: float, iou: float) -> InferenceHTTPClient:
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY is not set. Set it before running the application."
        )

    config = InferenceConfiguration(
        api_key_transport="header",
        confidence_threshold=confidence,
        iou_threshold=iou,
        disable_active_learning=True,
    )
    return InferenceHTTPClient(api_url=API_URL, api_key=api_key).configure(config)


def draw_predictions(frame, result):
    output = frame.copy()
    predictions = result.get("predictions", []) if result else []

    for pred in predictions:
        x, y = float(pred["x"]), float(pred["y"])
        w, h = float(pred["width"]), float(pred["height"])
        confidence = float(pred.get("confidence", 0.0))
        class_name = str(pred.get("class", "damage"))

        x1 = max(0, int(x - w / 2))
        y1 = max(0, int(y - h / 2))
        x2 = min(output.shape[1] - 1, int(x + w / 2))
        y2 = min(output.shape[0] - 1, int(y + h / 2))

        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name} {confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        box_y1 = max(0, y1 - th - 8)
        cv2.rectangle(output, (x1, box_y1), (x1 + tw + 8, y1), (0, 255, 0), -1)
        cv2.putText(output, label, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

    return output


class LatestFrameDetector:
    """Runs one remote inference at a time and always discards stale frames."""

    def __init__(self, client, detect_every: int):
        self.client = client
        self.detect_every = max(1, detect_every)
        self.condition = threading.Condition()
        self.latest_frame = None
        self.latest_frame_id = 0
        self.result = {"predictions": []}
        self.result_frame_id = 0
        self.inference_fps = 0.0
        self.stop_requested = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def submit(self, frame, frame_id):
        # Copying is important because OpenCV reuses the capture buffer.
        with self.condition:
            self.latest_frame = frame.copy()
            self.latest_frame_id = frame_id
            self.condition.notify()

    def snapshot(self):
        with self.condition:
            return self.result, self.result_frame_id, self.inference_fps

    def stop(self):
        with self.condition:
            self.stop_requested = True
            self.condition.notify()
        self.thread.join(timeout=2.0)

    def _run(self):
        while True:
            with self.condition:
                while self.latest_frame is None and not self.stop_requested:
                    self.condition.wait(timeout=0.2)
                if self.stop_requested:
                    return
                frame = self.latest_frame
                frame_id = self.latest_frame_id
                self.latest_frame = None

            if frame_id % self.detect_every != 0:
                continue

            started = time.perf_counter()
            try:
                result = self.client.infer(frame, model_id=MODEL_ID)
            except Exception as exc:
                print(f"Inference warning: {exc}")
                continue

            elapsed = time.perf_counter() - started
            with self.condition:
                self.result = result or {"predictions": []}
                self.result_frame_id = frame_id
                self.inference_fps = 1.0 / elapsed if elapsed else 0.0


def parse_source(value):
    try:
        return int(value)
    except ValueError:
        return value


def main():
    parser = argparse.ArgumentParser(description="Fast Roboflow road-damage video inference")
    parser.add_argument("source", help="Video path, webcam index, or stream URL")
    parser.add_argument("--output", default=None, help="Optional annotated MP4 path")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU)
    parser.add_argument(
        "--detect-every", type=int, default=3,
        help="Run remote detection every N frames (default: 3). Lower for accuracy, higher for speed.",
    )
    parser.add_argument("--display-width", type=int, default=1280)
    parser.add_argument("--display-height", type=int, default=720)
    parser.add_argument(
        "--no-display", action="store_true",
        help="Process without opening a preview window (useful on servers).",
    )
    args = parser.parse_args()

    capture = cv2.VideoCapture(parse_source(args.source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if not source_fps or source_fps <= 1 or source_fps != source_fps:
        source_fps = 20.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    writer = None
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (width, height)
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Could not create output video: {output_path}")

    detector = LatestFrameDetector(get_client(args.conf, args.iou), args.detect_every)
    detector.start()
    print(f"Running {MODEL_ID} with asynchronous inference")
    print(f"Source={args.source} | source FPS={source_fps:.2f} | detect every {args.detect_every} frame(s)")
    print("Playback stays responsive; stale frames are dropped when the API is slower than the video.")
    if not args.no_display:
        print("Press Q or ESC to stop.")

    frame_id = 0
    last_report = time.perf_counter()
    displayed_fps = 0.0
    displayed_at = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_id += 1
            detector.submit(frame, frame_id)

            result, result_frame_id, inference_fps = detector.snapshot()
            detections = result.get("predictions", [])
            annotated = draw_predictions(frame, result)
            now = time.perf_counter()
            interval = now - displayed_at
            if interval > 0:
                displayed_fps = 0.9 * displayed_fps + 0.1 / interval
            displayed_at = now

            cv2.putText(
                annotated,
                f"Frame: {frame_id} | Detections: {len(detections)} | "
                f"Display FPS: {displayed_fps:.1f} | Inference FPS: {inference_fps:.1f}",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA,
            )
            cv2.putText(
                annotated, f"Detection result from frame {result_frame_id}",
                (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA,
            )

            if writer is not None:
                writer.write(annotated)
            if not args.no_display:
                display = annotated
                if args.display_width > 0 and args.display_height > 0:
                    display = cv2.resize(annotated, (args.display_width, args.display_height))
                cv2.imshow("Roboflow Road Damage - Fast Live Inference", display)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

            if now - last_report > 5:
                print(f"Read {frame_id} frames | latest detections={len(detections)} | display FPS={displayed_fps:.1f}")
                last_report = now
    finally:
        detector.stop()
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    if args.output:
        print(f"Saved annotated video: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
