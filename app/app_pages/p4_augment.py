import pandas as pd
import streamlit as st

from common import GREEN, ORANGE, acc, bars, figure, intro, remember, what_why_result

intro("Change the training images so the model can't rely on things that differ in the field.", "Course Module 4 · Data augmentation")

what_why_result(
    "A controlled **ablation**: same model and recipe, only the augmentation changes, one step at a time.",
    "Module 1 showed backgrounds leak the label, and field photos differ in lighting, sharpness and clutter.",
    f"**Background replacement** was the clear winner: {acc('04_aug_none')}% → **{acc('04_aug_bg_synthetic')}%** in the field, with no field photos used.",
)

st.subheader("The recipes and what each one targets", anchor=False)
st.table(pd.DataFrame({
    "Recipe": ["none", "basic", "color", "randaug", "bg_synthetic", "bg_field"],
    "What it adds": ["resize + center crop", "random crop/zoom + flips", "+ brightness/contrast/saturation jitter, blur, rotation",
                     "+ 2 random strong operations per image", "+ leaf pasted on random colours and textures",
                     "+ leaf pasted on crops of unlabeled field photos"],
    "Field problem it targets": ["(reference)", "leaf position, scale, orientation", "sunlight, shadows, camera blur",
                                 "general robustness", "the background shortcut", "the shortcut + realistic clutter"],
}).set_index("Recipe"))
st.caption("Augmentation is applied on the fly: every epoch sees a new random version of each photo, only for training images.")

c1, c2 = st.columns([2, 3], gap="large")
with c1:
    st.subheader("How background replacement works", anchor=False)
    st.graphviz_chart("""
    digraph {
      rankdir=TB; bgcolor="transparent"; node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=11 color="#2C5F2D" fillcolor="#EEF4E6" width=2.6];
      edge [color="#7A8577"];
      a [label="1. Segmented leaf\\n(leaf on black)"];
      b [label="2. Mask = non-black pixels"];
      c [label="3. Shrink + blur the edge"];
      d [label="4. Pick a new background"];
      e [label="5. Paste leaf through mask" fillcolor="#2C5F2D" fontcolor="white"];
      a -> b -> c -> d -> e;
    }
    """)
    st.markdown("The disease stays the same but the background is new every time, so **the background can no longer predict the label**.")
with c2:
    figure("04_bg_replacement.png", "Original → segmented → mask → new background")
    figure("04_augmentations.png", "Five random versions of the same photo under each recipe")

st.subheader("The ablation result", anchor=False)
order = ["none", "basic", "color", "randaug", "bg_synthetic", "bg_field"]
key = {k: f"04_aug_{k}" for k in order} | {"basic": "03_mnv3_finetune"}
df = pd.DataFrame([{"recipe": k, "set": s, "accuracy (%)": acc(key[k], split)} for k in order
                   for s, split in [("Field validation (used to choose)", "field_val"), ("Field test (only reported)", "field_test")]])
bars(df, "recipe", "accuracy (%)", color="set", colors=[GREEN, ORANGE], y_max=45, sort=order, height=340)

c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown("- **Background replacement** wins on both validation and test: about +12–13 points over none.\n"
                "- **RandAugment doesn't help:** generic changes don't target the real gap.")
with c2:
    st.warning("Validation chose **bg_field** (by 1.2 points) while test favoured **bg_synthetic**. We followed validation anyway: "
               "choosing on the test set is cheating. With a few hundred field photos, a few points is noise.",
               title="Honest model selection", icon=":material/balance:")

remember([
    "**Augmentation** = random changes to training images so the model generalises.",
    "**Ablation** = change one thing at a time to know what caused the gain.",
    "**Background replacement** removes the shortcut: +13 points, no field data needed.",
    "Targeted augmentation beats generic augmentation (RandAugment didn't help).",
    "Pick the winner on **validation**, never on test.",
])
