import pandas as pd
import streamlit as st

from common import GREEN, GREY, ORANGE, bars, figure, intro, remember, what_why_result

intro("Before training anything: what kind of problem is this, and can we trust the data?", "Course Module 1 · Understanding the vision problem")

what_why_result(
    "Defined the task, picked the datasets, and audited them for leakage and shortcuts.",
    "A model is only as trustworthy as its test. Lab accuracy can be inflated by leaks and background cues.",
    "Found and removed **37 leaked leaves** and **15 duplicate images**, and proved a **background shortcut** exists.",
)

st.subheader("Task identification: why image classification", anchor=False)
c1, c2 = st.columns([3, 2], gap="large")
with c1:
    st.table(pd.DataFrame({
        "Question": ["Input", "Output", "Need the location of the disease?", "Need exact pixels?", "Where does it run?", "What makes it hard?"],
        "Answer": ["One photo of a leaf", "One label: crop + disease (e.g. Tomato late blight)", "No, so not object detection",
                   "No, so not segmentation", "On a phone, offline: small and fast model", "Field photos look nothing like training photos"],
    }).set_index("Question"))
    st.markdown("**Domain shift:** the training data (lab) and the real-world data (field) come from different distributions. "
                "Published models fall from ~99% to ~30% because of it.")
with c2:
    figure("01_lab_vs_field.png", "Same diseases: lab photos (left), field photos (right)")

st.subheader("The datasets and how they are split", anchor=False)
st.graphviz_chart("""
digraph {
  rankdir=LR; bgcolor="transparent"; node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=11 margin="0.15,0.08"];
  pv [label="PlantVillage\\nlab photos, 28 shared classes" fillcolor="#EEF4E6" color="#2C5F2D"];
  pd [label="PlantDoc\\nreal field photos" fillcolor="#FBEBDD" color="#D9822B"];
  tr [label="Lab train\\n(split by leaf)" fillcolor="white" color="#2C5F2D"];
  va [label="Lab validation" fillcolor="white" color="#2C5F2D"];
  te [label="Lab test" fillcolor="white" color="#2C5F2D"];
  ft [label="Field train\\n(field track only)" fillcolor="white" color="#D9822B"];
  fv [label="Field validation\\nused to CHOOSE settings" fillcolor="white" color="#D9822B"];
  fx [label="Field test, 236 photos\\nFINAL EXAM: only reported" fillcolor="#D9822B" fontcolor="white" color="#D9822B"];
  edge [color="#7A8577"];
  pv -> tr; pv -> va; pv -> te; pd -> ft; pd -> fv; pd -> fx;
}
""", width="content")

st.subheader("Honest evaluation: two leaks found and fixed", anchor=False)
row = st.container(horizontal=True)
row.metric("Leaves shared by the official train and test split", "37", "removed from training", delta_color="off", border=True)
row.metric("PlantDoc train images identical to test images", "15", "removed (pHash)", delta_color="off", border=True)
row.metric("Shared classes", "28", "chance = 3.6%", delta_color="off", border=True)
c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown(
        "- **Data leakage** = the model has already seen a test example during training, so the score is inflated.\n"
        "- **Leak 1:** PlantVillage has several photos of the *same leaf*. We split by `leaf_id` (GroupShuffleSplit), so each leaf is in one split only.\n"
        "- **Leak 2:** PlantDoc was collected from the web, so some pictures were copied into both folders. "
        "**Perceptual hashing (pHash)** gives each image a fingerprint that survives resizing; identical fingerprints = same photo."
    )
with c2:
    figure("01_duplicates.png", "Train copy (top) vs test photo (bottom): pHash distance 0")

st.subheader("The shortcut: guessing the disease from the background", anchor=False)
c1, c2 = st.columns([2, 3], gap="large")
with c1:
    st.markdown(
        "We threw away the leaf and kept only the **4 corners** of each photo (mostly background), "
        "then trained a simple classifier on them.\n\n"
        "If corners alone beat chance, the background is leaking the label, and a CNN can learn this shortcut instead of the disease."
    )
with c2:
    bars(pd.DataFrame({"test set": ["Lab photos", "Field photos"], "accuracy (%)": [46.1, 5.9]}),
         "test set", "accuracy (%)", colors=[GREEN, ORANGE], y_max=55, rule=3.6, rule_label="chance 3.6%",
         title="Accuracy from background corners only", height=280)
    st.caption("Lab backgrounds predict the disease; field backgrounds do not.")

remember([
    "It's **image classification**: one photo in, one label out.",
    "**Domain shift**: lab and field photos come from different distributions.",
    "Two **leaks** removed: 37 shared leaves (split by leaf) and 15 duplicate web images (pHash).",
    "**Shortcut**: corners alone give 46% on lab photos vs 6% in the field, so lab accuracy rewards cheating.",
    "Settings are chosen on **field validation**; the **field test** is the final exam, only reported.",
])
