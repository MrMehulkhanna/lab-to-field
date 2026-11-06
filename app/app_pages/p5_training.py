import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from common import BLUE, GREEN, GREY, MOSS, ORANGE, acc, bars, ece, figure, intro, remember, what_why_result

intro("How the model learns, and how we measure it, including whether its confidence can be trusted.",
      "Course Module 5 · Optimization, loss functions and metrics, activation functions")

what_why_result(
    "Compared optimizers, losses and activations one at a time; measured accuracy, F1, the confusion matrix and **calibration**.",
    "A model that is confidently wrong is dangerous for a farmer. We need honest confidence, not just accuracy.",
    f"**Label smoothing** cut calibration error from {ece('05_opt_adamw')} to **{ece('05_loss_label_smoothing')}** at similar accuracy.",
)

st.subheader("Optimizers and loss functions", anchor=False)
c1, c2 = st.columns(2, gap="large")
with c1:
    st.table(pd.DataFrame({
        "Choice": ["SGD + momentum", "AdamW", "Cross-entropy", "Label smoothing (0.1)", "Focal loss (γ=2)"],
        "In simple words": ["steps against the gradient; momentum keeps a running average, like a rolling ball",
                            "per-parameter step sizes + weight decay applied directly to the weights",
                            "−log p(true class), the standard loss", "target 0.9 instead of 1, so no push to 100% confidence",
                            "CE × (1 − p)^γ, focuses on hard examples"],
    }).set_index("Choice"))
    st.caption("All runs: 5% warm-up, then cosine learning-rate decay. Warm-up avoids big early steps; decay lets the model settle.")
with c2:
    runs = [("AdamW + CE", "05_opt_adamw"), ("SGD + CE", "05_opt_sgd"), ("AdamW + label smoothing", "05_loss_label_smoothing"),
            ("AdamW + focal", "05_loss_focal")]
    df = pd.DataFrame([{"experiment": n, "metric": "Field accuracy (%)", "value": acc(r)} for n, r in runs])
    bars(df, "experiment", "value", colors=[GREEN], y_max=45, sort=[n for n, _ in runs], title="Field test accuracy (%)", height=230)
    df2 = pd.DataFrame([{"experiment": n, "value": ece(r)} for n, r in runs])
    bars(df2, "experiment", "value", colors=[ORANGE], y_max=0.55, sort=[n for n, _ in runs], fmt=".2f",
         title="Calibration error (lower is better)", height=230)

st.subheader("Activation functions", anchor=False)
c1, c2 = st.columns([3, 2], gap="large")
with c1:
    x = np.linspace(-4, 4, 161)
    sig = 1 / (1 + np.exp(-x))
    gelu = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
    funcs = {"ReLU": np.maximum(0, x), "SiLU": x * sig, "GELU": gelu, "Hard-swish": x * np.clip(x + 3, 0, 6) / 6}
    show = st.pills("Show", list(funcs), default=list(funcs), selection_mode="multi")
    df = pd.DataFrame([{"x": xi, "f(x)": yi, "activation": k} for k in (show or []) for xi, yi in zip(x, funcs[k])])
    if len(df):
        st.altair_chart(alt.Chart(df).mark_line(strokeWidth=2.5).encode(
            x="x:Q", y="f(x):Q", color=alt.Color("activation:N", scale=alt.Scale(domain=list(funcs), range=[GREEN, ORANGE, BLUE, MOSS]),
                                                 legend=alt.Legend(orient="top", title=None))).properties(height=280), width="stretch")
with c2:
    st.markdown("- **ReLU** max(0, x): cheap, but its gradient is 0 for negative inputs (“dying ReLU”).\n"
                "- **SiLU** and **GELU** are smooth and let small negative values through.\n"
                "- **Hard-swish** is a cheap approximation of SiLU, used in MobileNetV3.")
    bars(pd.DataFrame({"activation": ["ReLU", "SiLU", "GELU"], "acc": [acc("02_simple_cnn"), acc("05_act_silu"), acc("05_act_gelu")]}),
         "activation", "acc", colors=[GREY], y_max=15, title="Scratch CNN field accuracy (%)", height=200)
    st.caption("Only ~1 point apart: the data problem matters far more than the activation.")

st.subheader("Metrics beyond accuracy", anchor=False)
c1, c2 = st.columns(2, gap="large")
with c1:
    st.table(pd.DataFrame({
        "Metric": ["Precision", "Recall", "F1", "Macro-F1", "Confusion matrix", "ECE (calibration error)"],
        "Answers": ["When it says “late blight”, how often is it right?", "Of real late-blight photos, how many did it find?",
                    "Harmonic mean of precision and recall", "Average F1 over classes: rare diseases count equally",
                    "Which diseases get mixed up", "Gap between stated confidence and real accuracy (0 = honest)"],
    }).set_index("Metric"))
with c2:
    with st.expander("Field confusion matrix", icon=":material/grid_on:"):
        figure("05_confusion_field.png", "Most common mix-ups: corn leaf blight vs gray leaf spot, potato early vs late blight")
    st.markdown("**Calibration: making confidence honest**")
    cal = pd.DataFrame({"model": ["Scratch CNN", "MobileNetV3 + CE", "+ label smoothing", "Final model"],
                        "ece": [ece("02_simple_cnn"), ece("05_opt_adamw"), ece("05_loss_label_smoothing"), ece("05_final_field")]})
    bars(cal, "model", "ece", colors=[BLUE], y_max=0.8, fmt=".2f", sort=list(cal.model), title="Calibration error on field photos", height=230)

st.subheader("Saying “not sure”: answering only when confident", anchor=False)
c1, c2 = st.columns(2, gap="large")
with c1:
    figure("05_reliability.png", "Reliability diagram: points on the diagonal = honest confidence")
with c2:
    figure("05_risk_coverage.png", "Answer fewer photos (lower coverage) and more of them are right")
st.markdown("**Temperature scaling** divides the scores by one number T before softmax: confidence changes, the predicted class never does. "
            "The app answers only above a confidence threshold chosen on validation data.")

remember([
    "**SGD + momentum vs AdamW**: about equal here; warm-up + cosine schedule for both.",
    "**Label smoothing** = target 0.9, so less overconfidence: calibration error 0.44 → 0.16.",
    "**Focal loss** focuses on hard examples; it didn't help (classes were already balanced at 500 each).",
    "**ReLU / SiLU / GELU** differ by ~1 point; the data matters more.",
    "**ECE** measures honest confidence; **abstention** = say “not sure” below a threshold.",
])
