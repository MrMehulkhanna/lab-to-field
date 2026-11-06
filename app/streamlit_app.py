"""Lab to Field: Streamlit demo.

Run:  uv run streamlit run app/streamlit_app.py
Uses the exported ONNX model (no PyTorch needed for prediction). If the PyTorch checkpoint
is present (local runs), it also draws a Grad-CAM heatmap.
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "export"
FIGS = ROOT / "results" / "figures"
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

st.set_page_config(page_title="Lab to Field · Crop Disease Check", page_icon="🌿", layout="wide")


@st.cache_resource
def load_session(precision: str):
    return ort.InferenceSession(str(EXPORT / f"plant_{precision}.onnx"), providers=["CPUExecutionProvider"])


@st.cache_resource
def load_meta():
    return json.loads((EXPORT / "labels.json").read_text())


def preprocess(img: Image.Image, size: int = 224) -> np.ndarray:
    """Same as training-time evaluation: resize shorter side to 255, center-crop 224, normalise."""
    img = img.convert("RGB")
    w, h = img.size
    short = int(size * 1.14)  # same rounding as torchvision Resize(255) + CenterCrop(224)
    new_w, new_h = (short, int(short * h / w)) if w <= h else (int(short * w / h), short)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    left, top = int(round((new_w - size) / 2.0)), int(round((new_h - size) / 2.0))
    img = img.crop((left, top, left + size, top + size))
    x = (np.asarray(img, dtype=np.float32) / 255 - MEAN) / STD
    return x.transpose(2, 0, 1)[None].astype(np.float32), img


def gradcam(img224: Image.Image):
    """Optional: heatmap from the PyTorch checkpoint, if it exists locally."""
    ckpt = ROOT / "checkpoints" / "05_final_field.pt"
    if not ckpt.exists():
        return None
    import torch
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from lab_to_field.config import N_CLASSES
    from lab_to_field.models import build_model

    model = build_model("mobilenetv3_large_100", N_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    rgb = np.asarray(img224, dtype=np.float32) / 255
    x = torch.tensor(((rgb - MEAN) / STD).transpose(2, 0, 1)[None], dtype=torch.float32)
    with GradCAM(model=model, target_layers=[model.blocks[-1]]) as cam:
        heat = cam(input_tensor=x, targets=None)[0]  # None = explain the predicted class
    return show_cam_on_image(rgb, heat, use_rgb=True)


meta = load_meta()
classes, T, threshold = meta["classes"], meta["temperature"], meta["abstain_threshold"]

st.title("🌿 Lab to Field: Crop Disease Check")
st.caption("MobileNetV3 trained to work on real field photos, not only lab images. Runs offline on a CPU "
           "from an 8.5 MB ONNX file, and says “not sure” instead of guessing.")

tab_predict, tab_slides, tab_results, tab_about = st.tabs(
    ["Diagnose a leaf", "Presentation", "Project results", "How it works"])

with tab_predict:
    left, right = st.columns([1, 1])
    with left:
        precision = st.segmented_control("Model file", ["fp16", "int8", "fp32"], default="fp16", required=True,
                                         help="FP16 (8.5 MB): default, same accuracy as FP32. INT8 (4.7 MB): "
                                              "smallest, loses a few points of field accuracy. FP32 (17 MB): original.")
        src = st.segmented_control("Image source", ["Sample field photo", "Upload", "Camera"],
                                   default="Sample field photo", required=True)
        true_label = None
        if src == "Sample field photo":
            samples = sorted((ROOT / "app" / "samples").glob("*.jpg"))
            choice = st.selectbox("PlantDoc test photo (never used in training)", samples, format_func=lambda p: p.stem)
            file, true_label = choice, choice.stem.split(" (")[0]
        elif src == "Upload":
            file = st.file_uploader("Leaf photo", type=["jpg", "jpeg", "png", "webp"])
        else:
            file = st.camera_input("Take a photo of one leaf")
        st.info("Tips: one leaf, fill the frame, daylight, no heavy blur.")
    if file:
        img = Image.open(file)
        x, img224 = preprocess(img)
        logits = load_session(precision).run(None, {"image": x})[0][0] / T
        p = np.exp(logits - logits.max())
        p /= p.sum()
        top = np.argsort(-p)[:3]
        with left:
            st.image(img, width="stretch", caption=f"True label: {true_label}" if true_label else None)
        with right:
            conf = float(p[top[0]])
            if conf >= threshold:
                st.success(f"**{classes[top[0]]}**  ·  confidence {conf:.0%}")
            else:
                st.warning(f"**Not sure** (best guess: {classes[top[0]]}, {conf:.0%}). "
                           "Take a clearer photo of a single leaf, or ask your local KVK / agriculture officer.")
            st.write("Top 3 (calibrated probabilities)")
            for i in top:
                st.progress(float(p[i]), text=f"{classes[i]}: {p[i]:.1%}")
            heat = gradcam(img224)
            if heat is not None:
                st.image(heat, caption="Grad-CAM: red areas drove the decision", width="stretch")
            else:
                st.caption("Grad-CAM heatmaps need the PyTorch checkpoint (local version); "
                           "see the Project results tab for examples.")
            st.caption(f"Abstain threshold {threshold:.2f} and temperature {T:.2f} were set on validation data.")

with tab_slides:
    slides = sorted((ROOT / "app" / "slides").glob("slide-*.jpg"))
    notes = json.loads((ROOT / "app" / "slides" / "notes.json").read_text())
    n = st.slider("Slide", 1, len(slides), 1)
    st.image(str(slides[n - 1]), width="stretch", caption=notes[n - 1]["title"])
    with st.container(border=True):
        st.markdown("**Explanation**")
        st.write(notes[n - 1]["notes"])
    deck = ROOT / "docs" / "Lab_to_Field_presentation.pptx"
    st.download_button("Download the presentation (.pptx)", deck.read_bytes(), file_name=deck.name,
                       mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                       icon=":material/download:")

with tab_results:
    figs = [("01_lab_vs_field.png", "Lab photos vs field photos"), ("02_gap.png", "The lab-to-field gap (scratch CNN)"),
            ("03_transfer.png", "Transfer learning"), ("04_ablation.png", "Augmentation ablation"),
            ("05_reliability.png", "Calibration"), ("05_risk_coverage.png", "Answering only when sure"),
            ("06_gradcam.png", "Grad-CAM: lab-only vs final model")]
    for f, cap in figs:
        if (FIGS / f).exists():
            st.subheader(cap)
            st.image(str(FIGS / f))

with tab_about:
    st.markdown("""
**Problem.** Plant-disease models score ~99% on lab photos (PlantVillage) but ~30% on real field photos.
This app is trained and tested to close that gap.

**Pipeline.** Photo → resize and crop to 224×224 → MobileNetV3 (ONNX, FP16) → logits ÷ temperature →
softmax → answer if confidence ≥ threshold, otherwise “not sure”.

**Honest evaluation.** PlantDoc test photos were never used for training or tuning.
Not a substitute for an agronomist.
""")
