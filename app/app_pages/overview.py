import altair as alt
import pandas as pd
import streamlit as st

from common import GREEN, ORANGE, acc, ece, figure

st.markdown("#### Crop disease detection that works on real farm photos, not just lab images")
st.markdown(
    "Plant-disease models score about **99% on lab photos** (one leaf on a plain board) but collapse on **real field photos**. "
    "This project measures that gap honestly, closes part of it with the techniques from Modules 1–5, "
    "and makes the model say **“not sure”** when it doesn't know."
)

row = st.container(horizontal=True)
row.metric("Scratch CNN, lab photos", f"{acc('02_simple_cnn', 'lab_test')}%", border=True)
row.metric("Scratch CNN, field photos", f"{acc('02_simple_cnn')}%", border=True)
row.metric("Final model, field photos", f"{acc('05_final_field')}%", f"+{acc('05_final_field') - acc('02_simple_cnn'):.1f} pts", border=True)
row.metric("Calibration error", f"{ece('05_final_field'):.2f}", f"{ece('05_final_field') - ece('02_simple_cnn'):.2f}",
           delta_color="inverse", border=True)
row.metric("Model size", "8.5 MB", "runs offline", delta_color="off", border=True)

left, right = st.columns([3, 2], gap="large")
with left:
    st.subheader("The journey: field accuracy at each step", anchor=False)
    steps = pd.DataFrame({
        "step": ["1 · CNN from scratch", "2 · Transfer learning", "3 · Background augmentation", "4 · Final model"],
        "field accuracy": [acc("02_simple_cnn"), acc("03_mnv3_finetune"), acc("04_aug_bg_synthetic"), acc("05_final_field")],
        "module": ["M2–M3", "M2–M3", "M4", "M5 + field photos"],
    })
    base = alt.Chart(steps).encode(
        x=alt.X("step:N", sort=None, title=None, axis=alt.Axis(labelAngle=0, labelLimit=170)),
        y=alt.Y("field accuracy:Q", title="Field accuracy (%)", scale=alt.Scale(domain=[0, 80])))
    chart = (base.mark_line(color=GREEN, strokeWidth=3)
             + base.mark_circle(size=160, color=ORANGE)
             + base.mark_text(dy=-16, fontSize=13, fontWeight="bold").encode(text=alt.Text("field accuracy:Q", format=".1f"))
             + base.mark_text(dy=20, fontSize=11, color="#7A8577").encode(text="module:N"))
    st.altair_chart(chart.properties(height=330), width="stretch")
with right:
    st.subheader("Lab photo vs field photo", anchor=False)
    figure("01_lab_vs_field.png", "Left: lab photos (PlantVillage). Right: field photos (PlantDoc), same diseases.")

st.subheader("How the project is built: one step per course module", anchor=False)
st.graphviz_chart("""
digraph {
  rankdir=LR; bgcolor="transparent"; node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=11 color="#2C5F2D" fillcolor="#EEF4E6" margin="0.18,0.1"];
  edge [color="#7A8577"];
  a [label="Problem and data\\nM1: task, leakage, shortcut"];
  b [label="CNN from scratch\\nM2, M3: blocks, shapes, params"];
  c [label="Transfer learning\\nM2, M3: probe, fine-tune, architectures"];
  d [label="Augmentation\\nM4: ablation, background replacement"];
  e [label="Training choices\\nM5: optimizers, losses, activations, metrics"];
  f [label="Final model and app\\nGrad-CAM, ONNX, Streamlit" fillcolor="#2C5F2D" fontcolor="white"];
  a -> b -> c -> d -> e -> f;
}
""", width="stretch")

st.info("Use the **Explained step by step** menu at the top to walk through each part, with the charts that prove each claim. "
        "**Diagnose a leaf** runs the real model; **Slides** has the presentation.", icon=":material/info:")
