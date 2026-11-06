"""RoadGuard AI — Streamlit pothole segmentation for uploaded road images.

Install: pip install streamlit torch torchvision segmentation-models-pytorch opencv-python-headless numpy pillow
Run:     streamlit run app.py

Put the trained checkpoint at models/best.pth, or set ROADGUARD_WEIGHTS.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import segmentation_models_pytorch as smp
from PIL import Image, UnidentifiedImageError

APP_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = Path(os.environ.get("ROADGUARD_WEIGHTS", str(APP_DIR / "models" / "best.pth")))
MODEL_SIZE = 256  # Must match training image size.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def infer_encoder(state_dict):
    """Identify common ResNet backbones from their saved layer shapes/counts."""
    key = "encoder.layer1.0.conv1.weight"
    if key not in state_dict:
        raise ValueError("Cannot detect ResNet encoder from checkpoint keys.")
    kernel = tuple(state_dict[key].shape[-2:])
    if kernel == (3, 3):
        # BasicBlock: ResNet18 has two blocks in layer1; ResNet34 has three.
        return "resnet34" if "encoder.layer1.2.conv1.weight" in state_dict else "resnet18"
    if kernel == (1, 1):
        # Bottleneck: ResNet50 has 6 blocks in layer3, ResNet101 has 23.
        if "encoder.layer3.22.conv1.weight" in state_dict:
            return "resnet101"
        return "resnet50"
    raise ValueError(f"Unsupported ResNet block kernel shape: {kernel}")


@st.cache_resource(show_spinner="Loading RoadGuard AI model...")
def load_model(weights_path: str):
    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a state_dict or a checkpoint containing model_state_dict.")
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, dict):
        raise ValueError("Invalid model_state_dict in checkpoint.")
    state = {k.removeprefix("module."): v for k, v in state.items()}
    encoder = infer_encoder(state)
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, device, encoder


def predict(model, device, rgb_image: np.ndarray) -> np.ndarray:
    height, width = rgb_image.shape[:2]
    resized = cv2.resize(rgb_image, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    tensor = torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
    return cv2.resize(probs, (width, height), interpolation=cv2.INTER_LINEAR)


def create_overlay(rgb_image: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    result = rgb_image.copy()
    red = np.zeros_like(result)
    red[:, :, 0] = 255  # RGB red
    blended = cv2.addWeighted(result, 1.0 - alpha, red, alpha, 0)
    result[mask] = blended[mask]
    return result


def png_bytes(rgb_image: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".png", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError("Unable to encode output image")
    return encoded.tobytes()


def main():
    st.set_page_config(page_title="RoadGuard AI", page_icon="🛣️", layout="wide")
    st.title("🛣️ RoadGuard AI")
    st.write("Upload a road image to detect and segment potholes using a trained U-Net model.")

    with st.sidebar:
        st.header("Inference Settings")
        threshold = st.slider("Mask threshold", 0.05, 0.95, 0.50, 0.05)
        alpha = st.slider("Overlay opacity", 0.10, 0.90, 0.45, 0.05)
        st.caption(f"Model checkpoint: `{WEIGHTS_PATH.name}`")
        st.caption("Input size: 256 × 256 | ImageNet normalization")

    uploaded = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded is None:
        st.info("Choose an image to start pothole segmentation.")
        return

    try:
        image = Image.open(uploaded).convert("RGB")
        rgb = np.array(image)
    except (UnidentifiedImageError, OSError) as exc:
        st.error(f"Could not open the image: {exc}")
        return

    try:
        model, device, encoder = load_model(str(WEIGHTS_PATH))
        probabilities = predict(model, device, rgb)
    except Exception as exc:
        st.error(f"Model inference failed: {exc}")
        st.info("Check that models/best.pth contains the trained U-Net checkpoint and matches the model architecture.")
        return

    mask = probabilities >= threshold
    mask_rgb = np.repeat((mask.astype(np.uint8) * 255)[:, :, None], 3, axis=2)
    overlay = create_overlay(rgb, mask, alpha)

    st.success(f"Inference complete | Encoder: {encoder} | Device: {device}")
    st.metric("Segmented image area", f"{100 * mask.mean():.2f}%")
    st.caption("This percentage measures pixels classified as potholes, not a confidence score.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(rgb, caption="Original Road Image", use_container_width=True)
    with c2:
        st.image(mask_rgb, caption="Predicted Binary Mask", use_container_width=True)
    with c3:
        st.image(overlay, caption="Pothole Segmentation Overlay", use_container_width=True)

    st.download_button(
        "Download Segmented Image",
        data=png_bytes(overlay),
        file_name="roadguard_result.png",
        mime="image/png",
        type="primary",
    )


if __name__ == "__main__":
    main()
