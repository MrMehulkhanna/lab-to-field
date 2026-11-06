import pandas as pd
import streamlit as st

from common import GREEN, MOSS, ORANGE, acc, bars, ece, intro, remember, what_why_result

intro("Reuse networks that already learned from millions of images, and compare architectures on field accuracy.",
      "Course Modules 2 and 3 · Transfer learning, advanced architectures")

what_why_result(
    "Replaced the last layer of pretrained networks with a 28-class layer; compared linear probe vs fine-tuning across 4 architectures.",
    "A small model trained from zero can't learn general visual features from 12k lab photos. Pretrained models already know edges, textures and shapes.",
    f"Field accuracy rose from {acc('02_simple_cnn')}% to **~30%**; frozen **DINOv2 reached {acc('03_dinov2_probe')}%**; adding real field photos gave **{acc('03_mnv3_finetune_field')}%**.",
)

st.subheader("Two ways to transfer", anchor=False)
row = st.container(horizontal=True)
with row.container(border=True, height="stretch"):
    st.markdown("**:material/lock: Linear probe (feature extraction)**")
    st.markdown("- Freeze the whole backbone\n- Train only the new last layer\n- Fast; tests if pretrained features are good as they are")
with row.container(border=True, height="stretch"):
    st.markdown("**:material/lock_open: Fine-tuning**")
    st.markdown("- Train every layer with a **small learning rate** (1e-3)\n- Better fit to our data\n- Risk: it can over-fit to lab shortcuts")

st.subheader("The architectures we compared", anchor=False)
st.table(pd.DataFrame({
    "Model": ["ResNet-50 (2015)", "MobileNetV3-Large (2019)", "EfficientNet-B0 (2019)", "DINOv2 ViT-S/14 (2023)"],
    "Key idea": ["Residual connections: each block learns x + F(x), so very deep networks train",
                 "Depthwise separable convs, squeeze-and-excitation, hard-swish: built for phones",
                 "Scales depth, width and resolution together (compound scaling)",
                 "Vision Transformer: image cut into 14×14 patches + self-attention; pretrained without labels on 142M images"],
    "Parameters": ["25.6 M", "5.5 M (deployed)", "5.3 M", "22 M"],
}).set_index("Model"))

with st.container(border=True):
    st.markdown("**Try it: why depthwise separable convolutions are cheap**")
    c1, c2, c3 = st.columns(3)
    cin = c1.number_input("Input channels", 8, 1024, 128, step=8)
    cout = c2.number_input("Output channels", 8, 1024, 128, step=8)
    k = c3.number_input("Kernel size", 1, 7, 3, step=2)
    normal, sep = k * k * cin * cout, k * k * cin + cin * cout
    m = st.container(horizontal=True)
    m.metric("Normal convolution", f"{normal:,} weights")
    m.metric("Depthwise + pointwise", f"{sep:,} weights")
    m.metric("Saving", f"{normal / sep:.1f}× fewer")
    st.caption("Depthwise: one k×k filter per input channel. Pointwise: a 1×1 conv mixes channels.")

st.subheader("Results: lab accuracy is easy, field accuracy is the real test", anchor=False)
runs = [("Scratch CNN", "02_simple_cnn"), ("MobileNetV3 probe", "03_mnv3_probe"), ("MobileNetV3 fine-tune", "03_mnv3_finetune"),
        ("EfficientNet-B0", "03_effb0_finetune"), ("ResNet-50", "03_resnet50_finetune"), ("DINOv2 frozen", "03_dinov2_probe"),
        ("MobileNetV3 + field photos", "03_mnv3_finetune_field")]
df = pd.DataFrame([{"model": n, "test": t, "accuracy (%)": acc(r, s)} for n, r in runs
                   for t, s in [("Lab test", "lab_test"), ("Field test", "field_test")]])
bars(df, "model", "accuracy (%)", color="test", colors=[MOSS, ORANGE], y_max=112, sort=[n for n, _ in runs], height=360)

c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown(
        f"1. **Pretraining is the biggest model-side step:** {acc('02_simple_cnn')}% → 25–32% in the field.\n"
        f"2. **Best lab score ≠ best field score:** EfficientNet-B0 has the highest lab accuracy but the lowest field accuracy.\n"
        f"3. **Fine-tuning vs probe:** only ~3 more field points, but more overconfident (calibration error {ece('03_mnv3_probe')} → {ece('03_mnv3_finetune')})."
    )
with c2:
    st.markdown(
        f"4. **Self-supervised DINOv2 transfers best** ({acc('03_dinov2_probe')}%), matching published results (~41–43%).\n"
        f"5. **Real field photos matter most:** ~2,000 PlantDoc photos lift MobileNetV3 to {acc('03_mnv3_finetune_field')}%.\n"
        "6. **We deploy MobileNetV3:** 4× smaller than DINOv2 and fast on a phone CPU."
    )

remember([
    "**Transfer learning** = reuse a pretrained backbone, replace the head.",
    "**Linear probe** trains only the head; **fine-tuning** trains everything with a small learning rate.",
    "**Residual connection**: y = x + F(x). **Depthwise separable**: ~8× fewer weights. **ViT**: patches + attention.",
    f"Our ResNet-50 ({acc('03_resnet50_finetune')}%) and DINOv2 ({acc('03_dinov2_probe')}%) match published numbers, so the pipeline is sound.",
    "Choose models by **field accuracy and phone cost**, not lab accuracy.",
])
