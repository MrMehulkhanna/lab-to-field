import altair as alt
import pandas as pd
import streamlit as st

from common import GREEN, ORANGE, acc, bars, ece, figure, intro, remember, results, what_why_result

intro("Build the simplest classifier by hand, understand every layer, and measure the gap.", "Course Modules 2 and 3 · Classification, building blocks, parameters, tensor shapes")

what_why_result(
    "Built a 4-block CNN from scratch, traced every tensor shape and counted every parameter.",
    "A baseline to measure every later improvement against, and proof that we understand what a CNN is.",
    f"**{acc('02_simple_cnn', 'lab_test')}%** on lab photos but only **{acc('02_simple_cnn')}%** on field photos, while still 78% confident.",
)

st.subheader("The architecture and its tensor shapes", anchor=False)
st.graphviz_chart("""
digraph {
  rankdir=LR; bgcolor="transparent"; node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=11 margin="0.15,0.08" color="#2C5F2D" fillcolor="#EEF4E6"];
  edge [color="#7A8577" fontname="Helvetica" fontsize=10];
  in [label="Input\\n3×224×224" fillcolor="white"];
  b1 [label="Block 1\\n32×112×112"]; b2 [label="Block 2\\n64×56×56"]; b3 [label="Block 3\\n128×28×28"]; b4 [label="Block 4\\n256×14×14"];
  gap [label="Global avg pool\\n256"]; fc [label="Dropout + Linear\\n28 class scores" fillcolor="#2C5F2D" fontcolor="white"];
  in -> b1 -> b2 -> b3 -> b4 -> gap -> fc;
}
""", width="stretch")
st.caption("Each block = Conv 3×3 → BatchNorm → ReLU → MaxPool 2×2. Height and width halve, channels double: less “where”, more “what”.")

c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown("**What each building block does**")
    st.table(pd.DataFrame({
        "Block": ["Convolution 3×3", "Batch normalization", "ReLU", "Max pooling 2×2", "Global average pooling", "Dropout", "Linear"],
        "In simple words": ["slides small filters to find edges, spots, colour changes", "rescales each channel so training is stable",
                            "max(0, x): adds non-linearity", "keeps the largest value per 2×2 window, halves the size",
                            "averages each channel to one number", "randomly drops 30% of features while training (regularization)",
                            "turns 256 features into 28 class scores"],
    }).set_index("Block"))
with c2:
    st.markdown("**Counting parameters by hand**")
    chs = [3, 32, 64, 128, 256]
    rows = [{"layer": f"Conv block {i + 1}", "formula": f"3·3·{a}·{b} + 2·{b}", "parameters": 9 * a * b + 2 * b}
            for i, (a, b) in enumerate(zip(chs[:-1], chs[1:]))]
    rows.append({"layer": "Linear", "formula": "256·28 + 28", "parameters": 256 * 28 + 28})
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, width="stretch",
                 column_config={"parameters": st.column_config.NumberColumn(format="%d")})
    st.metric("Total parameters", f"{df.parameters.sum():,}", "matches PyTorch's count", delta_color="off")
    st.caption("Conv: k·k·C_in·C_out (no bias, BatchNorm follows). BatchNorm: 2·C (scale and shift). ReLU, pooling and dropout: 0.")

st.subheader("Training and the result", anchor=False)
c1, c2 = st.columns(2, gap="large")
with c1:
    h = pd.DataFrame(results()["02_simple_cnn"]["history"])
    h["val_acc"] *= 100
    base = alt.Chart(h).encode(x=alt.X("epoch:O", axis=alt.Axis(labelAngle=0)))
    loss = base.mark_line(point=True, color=ORANGE).encode(y=alt.Y("train_loss:Q", title="training loss"))
    val = base.mark_line(point=True, color=GREEN).encode(y=alt.Y("val_acc:Q", title="lab validation accuracy (%)"))
    st.altair_chart(alt.layer(loss, val).resolve_scale(y="independent").properties(height=300,
                    title="Loss falls, lab accuracy rises (orange = loss, green = accuracy)"), width="stretch")
    st.caption("Recipe: cross-entropy loss, AdamW optimizer, warm-up + cosine learning rate, mixed precision, 12 epochs.")
with c2:
    bars(pd.DataFrame({"test set": ["Lab test", "Field test"], "accuracy (%)": [acc("02_simple_cnn", "lab_test"), acc("02_simple_cnn")]}),
         "test set", "accuracy (%)", colors=[GREEN, ORANGE], y_max=110, rule=3.6, rule_label="chance",
         title="The lab-to-field gap", height=300)
    st.caption(f"On field photos the model is still **78% confident** while right only {acc('02_simple_cnn')}% of the time "
               f"(calibration error {ece('02_simple_cnn')}). It is confidently wrong.")

remember([
    "Block = **Conv → BatchNorm → ReLU → MaxPool**; then global average pooling and a linear layer.",
    "Shapes: **3×224×224 → 32×112×112 → 64×56×56 → 128×28×28 → 256×14×14 → 256 → 28**.",
    "Conv params = **k·k·C_in·C_out**; total **396,092**. ReLU and pooling have **no** parameters.",
    "Parameters = memory; MACs = compute (early layers: few parameters, many MACs).",
    f"Result: **{acc('02_simple_cnn', 'lab_test')}% lab vs {acc('02_simple_cnn')}% field**, so a small model trained from zero only memorises the lab.",
])
