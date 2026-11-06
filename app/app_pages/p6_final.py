import pandas as pd
import streamlit as st

from common import BLUE, GREEN, ORANGE, acc, bars, ece, figure, intro, remember, what_why_result

intro("Put the best choices together, check what the model looks at, and ship it small enough for a phone.",
      "Builds on Modules 3 and 5 · preview of Module 7 (edge deployment)")

what_why_result(
    "Final model: MobileNetV3 trained on lab + field photos, background replacement, AdamW, label smoothing. Explained with Grad-CAM, exported to ONNX.",
    "Farmers need an answer offline, on a cheap phone, and a model that admits when it is unsure.",
    f"**{acc('05_final_field')}%** field accuracy, calibration error **{ece('05_final_field')}**, 8.5 MB, runs on a CPU.",
)

row = st.container(horizontal=True)
row.metric("Field test accuracy", f"{acc('05_final_field')}%", "95% CI 58–70%", delta_color="off", border=True)
row.metric("Calibration error", f"{ece('05_final_field'):.2f}", "down from 0.69", delta_color="off", border=True)
row.metric("Photos answered", "71%", "says “not sure” on the rest", delta_color="off", border=True)
row.metric("Accuracy when it answers", "77.8%", border=True)

st.subheader("Grad-CAM: where does the model look?", anchor=False)
c1, c2 = st.columns([1, 2], gap="large")
with c1:
    st.markdown("**How it works**\n1. Take the score of the predicted class\n2. Back-propagate it to the last convolutional feature maps\n"
                "3. Average the gradients per channel = how important each channel is\n4. Weighted sum → ReLU → upsample = heatmap")
    st.markdown("Red = regions that drove the decision. Heat on the background means the model is using a shortcut.")
    st.info("The lab-only model gets 1 of 5 right; the final model gets 4 of 5. A collage photo gets only 11% confidence, "
            "so the app says “not sure”. One correct apple scab prediction still uses some background: the fix is partial.",
            icon=":material/visibility:")
with c2:
    figure("06_gradcam.png", "Top: field photo. Middle: lab-only model. Bottom: final model.")

st.subheader("Shipping it: smaller model files", anchor=False)
c1, c2 = st.columns(2, gap="large")
with c1:
    fmt = pd.DataFrame({"format": ["FP32", "FP16 (shipped)", "INT8"], "size (MB)": [16.9, 8.5, 4.7]})
    bars(fmt, "format", "size (MB)", colors=[BLUE], y_max=20, sort=list(fmt.format), title="Model file size (MB)", height=240)
with c2:
    acc_df = pd.DataFrame({"format": ["FP32", "FP16 (shipped)", "INT8"], "acc": [64.4, 64.4, 60.6]})
    bars(acc_df, "format", "acc", colors=[GREEN], y_max=75, sort=list(acc_df.format), title="Field accuracy (%)", height=240)
st.markdown("**ONNX** is a standard model file that runs without PyTorch (Python, browser, Android); the export matches PyTorch to 2×10⁻⁶. "
            "**FP16** halves the size for free. **INT8** is 3.6× smaller but loses 3.8 points on MobileNetV3, so it needs quantization-aware training (Module 7).")

st.subheader("What happens when you use the app", anchor=False)
st.graphviz_chart("""
digraph {
  rankdir=LR; bgcolor="transparent"; node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=11 color="#2C5F2D" fillcolor="#EEF4E6" margin="0.15,0.08"];
  edge [color="#7A8577"];
  a [label="Leaf photo\\nsample, upload, camera"]; b [label="Resize 255,\\ncrop 224, normalise"];
  c [label="MobileNetV3\\nONNX FP16, 8.5 MB"]; d [label="Scores ÷ temperature\\n→ softmax"];
  e [label="Confidence ≥ 0.54 ?" shape=diamond fillcolor="white"];
  f [label="Show disease + top 3" fillcolor="#2C5F2D" fontcolor="white"]; g [label="“Not sure”: retake photo\\nor ask an expert" fillcolor="#D9822B" fontcolor="white"];
  a -> b -> c -> d -> e; e -> f [label="yes"]; e -> g [label="no"];
}
""", width="content")

c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown("**Limitations**\n- PlantDoc is small (236 test photos) and somewhat noisy\n- No wheat, rice or cotton classes\n"
                "- The model still sometimes uses background cues")
with c2:
    st.markdown("**Next steps (Group Project 2)**\n- Punjab field photos labelled by experts\n- Distil DINOv2 into MobileNetV3\n"
                "- INT8 with quantization-aware training\n- Leaf segmentation (Module 6) and an Android app")

remember([
    f"Final: **{acc('05_final_field')}% field**, calibration error **{ece('05_final_field')}**, answers 71% of photos at 77.8%.",
    "**Grad-CAM** = class-score gradients on the last conv maps → heatmap of where the model looked.",
    "**ONNX** runs anywhere; **FP16** 8.5 MB with no loss; **INT8** loses 3.8 points.",
    "Three lessons: lab accuracy lies · remove leakage and shortcuts · a model that knows when it's unsure is more useful.",
])
