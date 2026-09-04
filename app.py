import argparse
import json
import os
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


def draw_predictions(image, result):
    output = image.copy()
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

    cv2.putText(
        output,
        f"Detections: {len(predictions)}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def main():
    parser = argparse.ArgumentParser(description="Roboflow road-damage image inference")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--output", default="outputs/image_result.jpg", help="Annotated output image path")
    parser.add_argument("--json", default=None, help="Optional path to save raw Roboflow JSON")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE, help="Confidence threshold (0-1)")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU, help="NMS IoU threshold (0-1)")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")

    client = get_client(args.conf, args.iou)
    print(f"Running {MODEL_ID} on: {image_path}")
    print(f"Confidence={args.conf:.2f}, IoU={args.iou:.2f}")

    result = client.infer(image, model_id=MODEL_ID)
    predictions = result.get("predictions", [])
    print(f"Detections: {len(predictions)}")

    for i, pred in enumerate(predictions, 1):
        print(
            f"  {i}. {pred.get('class', 'damage')} "
            f"confidence={float(pred.get('confidence', 0.0)):.3f} "
            f"box=({pred.get('x')}, {pred.get('y')}, {pred.get('width')}, {pred.get('height')})"
        )

    output = draw_predictions(image, result)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), output):
        raise IOError(f"Failed to write output image: {output_path}")
    print(f"Saved annotated image: {output_path.resolve()}")

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved raw JSON: {json_path.resolve()}")


if __name__ == "__main__":
    main()
