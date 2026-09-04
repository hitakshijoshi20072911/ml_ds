import argparse
import os
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
            "ROBOFLOW_API_KEY is not set. Set it in PowerShell with "
            '$env:ROBOFLOW_API_KEY="your_key" before running.'
        )

    config = InferenceConfiguration(
        api_key_transport="header",
        confidence_threshold=confidence,
        iou_threshold=iou,
        disable_active_learning=True,
    )

    return InferenceHTTPClient(
        api_url=API_URL,
        api_key=api_key,
    ).configure(config)


def draw_predictions(frame, result):
    output = frame.copy()
    predictions = result.get("predictions", [])

    for pred in predictions:
        x = float(pred["x"])
        y = float(pred["y"])
        w = float(pred["width"])
        h = float(pred["height"])
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
        cv2.putText(
            output,
            label,
            (x1 + 4, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return output


def main():
    parser = argparse.ArgumentParser(description="Roboflow road-damage video/live inference")
    parser.add_argument(
        "source",
        help="Video path, webcam index (0, 1, ...), or a stream URL such as RTMP/HTTP supported by OpenCV/FFmpeg",
    )
    parser.add_argument("--output", default=None, help="Optional output MP4 path")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE, help="Confidence threshold (0-1)")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU, help="NMS IoU threshold (0-1)")
    parser.add_argument("--display-width", type=int, default=1280, help="Display width; 0 keeps original")
    parser.add_argument("--display-height", type=int, default=720, help="Display height; 0 keeps original")
    args = parser.parse_args()

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(
            "ROBOFLOW_API_KEY is not set. Set it in PowerShell with "
            '$env:ROBOFLOW_API_KEY="your_key" before running.'
        )

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1 or fps != fps:
        fps = 20.0

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    writer = None
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Could not create output video: {output_path}")

    client = get_client(args.conf, args.iou)
    print(f"Running {MODEL_ID}")
    print(f"Source={args.source}")
    print(f"Confidence={args.conf:.2f}, IoU={args.iou:.2f}")
    print("Press Q or ESC to stop.")

    frame_id = 0
    last_time = time.perf_counter()
    shown_fps = 0.0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("End of stream / could not read next frame.")
                break

            frame_id += 1
            started = time.perf_counter()
            result = client.infer(frame, model_id=MODEL_ID)
            annotated = draw_predictions(frame, result)
            detections = result.get("predictions", [])

            elapsed = time.perf_counter() - started
            shown_fps = 1.0 / elapsed if elapsed > 0 else 0.0

            cv2.putText(
                annotated,
                f"Frame: {frame_id} | Detections: {len(detections)} | Inference FPS: {shown_fps:.2f}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            display = annotated
            if args.display_width > 0 and args.display_height > 0:
                display = cv2.resize(annotated, (args.display_width, args.display_height))

            cv2.imshow("Roboflow Road Damage - Live Inference", display)
            if writer is not None:
                writer.write(annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            # Prevent an accidental tight loop when a source behaves unusually.
            now = time.perf_counter()
            if now - last_time > 5:
                print(f"Processed {frame_id} frames | latest detections={len(detections)}")
                last_time = now
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    if args.output:
        print(f"Saved annotated video: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
