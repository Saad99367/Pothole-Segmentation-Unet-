"""RoadGuard AI: webcam/video pothole segmentation.

Examples:
    python webcam.py --source 0
    python webcam.py --source videos/road.mp4
    python webcam.py --source videos/road.mp4 --save outputs/road_result.mp4
    python webcam.py --source 0 --width 1280 --height 720
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_model(weights: Path, device: torch.device):
    if not weights.is_file():
        raise FileNotFoundError(f"Model not found: {weights}")

    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError("Expected a state_dict or checkpoint dictionary, not a serialized full model")
    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}

    # Detect the ResNet backbone from the saved parameter names.
    if "encoder.layer1.0.conv3.weight" in state_dict:
        encoder = "resnet50"
        # Distinguish larger bottleneck architectures by block counts.
        if "encoder.layer3.22.conv1.weight" in state_dict:
            encoder = "resnet101"
        if "encoder.layer3.35.conv1.weight" in state_dict:
            encoder = "resnet152"
    else:
        encoder = "resnet34" if "encoder.layer1.2.conv1.weight" in state_dict else "resnet18"

    print(f"Detected encoder: {encoder}")
    model = smp.Unet(
        encoder_name=encoder, encoder_weights=None,
        in_channels=3, classes=1, activation=None,
    )
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def predict(model, frame, device, size, threshold):
    height, width = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    tensor = torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    probabilities = cv2.resize(probabilities, (width, height), interpolation=cv2.INTER_LINEAR)
    return probabilities >= threshold


def main():
    parser = argparse.ArgumentParser(description="RoadGuard AI real-time inference")
    parser.add_argument("--source", default="0", help="0 for webcam, or a video file path")
    parser.add_argument("--weights", default=str(Path(__file__).resolve().parent / "models" / "best.pth"), help="Model checkpoint")
    parser.add_argument("--size", type=int, default=256, help="Model input size")
    parser.add_argument("--width", type=int, default=960, help="Processed/display frame width")
    parser.add_argument("--height", type=int, default=540, help="Processed/display frame height")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save", default=None, help="Optional output video path")
    args = parser.parse_args()
    if args.size <= 0 or args.width <= 0 or args.height <= 0 or not 0 <= args.threshold <= 1:
        parser.error("--size, --width and --height must be positive; --threshold must be between 0 and 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = load_model(Path(args.weights), device)
    print(f"Loaded model: {args.weights}")

    source = int(args.source) if args.source.isdecimal() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    writer = None
    fps = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # Resize before inference and display to reduce processing cost.
            frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)
            start = time.perf_counter()
            mask = predict(model, frame, device, args.size, args.threshold)
            overlay = frame.copy()
            overlay[mask] = (0, 0, 255)
            result = cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)
            elapsed = time.perf_counter() - start
            instant_fps = 1 / max(elapsed, 1e-6)
            fps = instant_fps if fps == 0 else 0.85 * fps + 0.15 * instant_fps
            cv2.putText(result, f"RoadGuard AI | FPS: {fps:.1f}", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if args.save:
                if writer is None:
                    out = Path(args.save)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    video_fps = cap.get(cv2.CAP_PROP_FPS)
                    if not np.isfinite(video_fps) or video_fps <= 0:
                        video_fps = 25.0
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                                             video_fps, (w, h))
                    if not writer.isOpened():
                        raise RuntimeError(f"Cannot save video: {out}")
                writer.write(result)

            cv2.imshow("RoadGuard AI | Press Q to quit", result)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
