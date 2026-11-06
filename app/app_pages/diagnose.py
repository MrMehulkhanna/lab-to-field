import json
from pathlib import Path

import numpy as np
import pandas as pd
import onnxruntime as ort
import streamlit as st
from PIL import Image

from common import EXPORT, ROOT

MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


@st.cache_resource
def load_session(precision: str):
    return ort.InferenceSession(str(EXPORT / f"plant_{precision}.onnx"), providers=["CPUExecutionProvider"])


@st.cache_resource
def load_meta():
    return json.loads((EXPORT / "labels.json").read_text())


def preprocess(img: Image.Image, size: int = 224):
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


def predict(img: Image.Image, precision: str):
    """Calibrated class probabilities for one photo."""
    x, img224 = preprocess(img)
    logits = load_session(precision).run(None, {"image": x})[0][0] / T
    p = np.exp(logits - logits.max())
    return p / p.sum(), img224


def label_from_name(path: Path) -> str | None:
    """'Tomato late blight - my farm.jpg' -> 'Tomato late blight' if that is a known class."""
    head = path.stem.split(" - ")[0].split(" (")[0].strip()
    return next((c for c in classes if c.lower() == head.lower()), None)


meta = load_meta()
classes, T, threshold = meta["classes"], meta["temperature"], meta["abstain_threshold"]
TEST_DIR = ROOT / "test_images"
st.caption("MobileNetV3 trained to work on real field photos. Runs on a CPU from an 8.5 MB ONNX file and says “not sure” instead of guessing.")

left, right = st.columns([1, 1], gap="large")
with left:
    precision = st.segmented_control("Model file", ["fp16", "int8", "fp32"], default="fp16", required=True,
                                     help="FP16 (8.5 MB): default, same accuracy as FP32. INT8 (4.7 MB): "
                                          "smallest, loses a few points of field accuracy. FP32 (17 MB): original.")
    src = st.segmented_control("Image source", ["Sample field photo", "My test photos", "Upload", "Camera"],
                               default="Sample field photo", required=True)
    true_label, file, test_all = None, None, False
    if src == "Sample field photo":
        samples = sorted((ROOT / "app" / "samples").glob("*.jpg"))
        choice = st.selectbox("PlantDoc test photo (never used in training)", samples, format_func=lambda p: p.stem)
        file, true_label = choice, choice.stem.split(" (")[0]
    elif src == "My test photos":
        mine = sorted(p for p in TEST_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        if not mine:
            st.warning("No photos yet. Add .jpg or .png files to the `test_images/` folder of the project.", icon=":material/folder_open:")
        else:
            choice = st.selectbox(f"Photo from test_images/ ({len(mine)} photos)", mine, format_func=lambda p: p.name)
            file, true_label = choice, label_from_name(choice)
            test_all = st.button("Test all photos in the folder", icon=":material/playlist_play:", width="stretch")
    elif src == "Upload":
        file = st.file_uploader("Leaf photo", type=["jpg", "jpeg", "png", "webp"])
    else:
        file = st.camera_input("Take a photo of one leaf")
    st.info("Tips: one leaf, fill the frame, daylight, no heavy blur.", icon=":material/tips_and_updates:")

if test_all:
    st.subheader(f"All photos in test_images/ ({len(mine)})", anchor=False)
    rows, grid = [], st.container(horizontal=True)
    for path in mine:
        prob, _ = predict(Image.open(path), precision)
        k, conf, truth = int(prob.argmax()), float(prob.max()), label_from_name(path)
        answer = classes[k] if conf >= threshold else "Not sure"
        verdict = "" if truth is None else (" ✓" if answer == truth else (" ?" if answer == "Not sure" else " ✗"))
        rows.append({"photo": path.name, "prediction": answer, "best guess": classes[k], "confidence": conf,
                     "true label (from file name)": truth or "", "result": verdict.strip()})
        with grid.container(border=True, width=230):
            st.image(str(path), width="stretch", alt=path.name)
            st.markdown(f"**{answer}**{verdict}")
            st.caption(f"{conf:.0%} confidence · {path.name}")
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, width="stretch",
                 column_config={"confidence": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
    labelled = df[df["true label (from file name)"] != ""]
    if len(labelled):
        answered = labelled[labelled.prediction != "Not sure"]
        m = st.container(horizontal=True)
        m.metric("Labelled photos", len(labelled))
        m.metric("Answered", f"{len(answered)}", f"{len(labelled) - len(answered)} not sure", delta_color="off")
        m.metric("Correct when answered", f"{(answered.prediction == answered['true label (from file name)']).mean():.0%}" if len(answered) else "–")
elif file:
    img = Image.open(file)
    p, img224 = predict(img, precision)
    top = np.argsort(-p)[:3]
    with left:
        st.image(img, width="stretch", caption=f"True label: {true_label}" if true_label else None, alt="Leaf photo")
    with right:
        conf = float(p[top[0]])
        if conf >= threshold:
            st.success(f"**{classes[top[0]]}**  ·  confidence {conf:.0%}", icon=":material/check_circle:")
        else:
            st.warning(f"Best guess: {classes[top[0]]} ({conf:.0%}). Take a clearer photo of a single leaf, "
                       "or ask your local KVK / agriculture officer.", title="Not sure", icon=":material/help:")
        st.write("Top 3 (calibrated probabilities)")
        for i in top:
            st.progress(float(p[i]), text=f"{classes[i]}: {p[i]:.1%}")
        heat = gradcam(img224)
        if heat is not None:
            st.image(heat, caption="Grad-CAM: red areas drove the decision", width="stretch", alt="Grad-CAM heatmap")
        else:
            st.caption("Grad-CAM heatmaps need the PyTorch checkpoint (local version); see page 6 for examples.")
        st.caption(f"Abstain threshold {threshold:.2f} and temperature {T:.2f} were set on validation data.")
