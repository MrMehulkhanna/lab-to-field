"""Shared paths, data loading and small UI helpers for every page."""
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "export"
FIGS = ROOT / "results" / "figures"
RESULTS = ROOT / "results"

GREEN, ORANGE, MOSS, BLUE, GREY = "#2C5F2D", "#D9822B", "#97BC62", "#4F7CAC", "#7A8577"


@st.cache_data
def results() -> dict:
    return {p.stem: json.loads(p.read_text()) for p in RESULTS.glob("0*.json")}


def acc(run: str, split: str = "field_test") -> float:
    """Accuracy in percent for one saved experiment."""
    return round(100 * results()[run][split]["accuracy"], 1)


def ece(run: str, split: str = "field_test") -> float:
    return round(results()[run][split]["ece"], 2)


def figure(name: str, caption: str, alt_text: str | None = None):
    st.image(str(FIGS / name), caption=caption, width="stretch", alt=alt_text or caption)


def intro(one_line: str, module: str):
    st.caption(module)
    st.markdown(f"##### {one_line}")


def what_why_result(what: str, why: str, result: str):
    """The three-box summary used at the top of every explanation page."""
    row = st.container(horizontal=True)
    for title, icon, body in [("What we did", ":material/build:", what), ("Why", ":material/lightbulb:", why),
                              ("Result", ":material/insights:", result)]:
        with row.container(border=True, height="stretch"):
            st.markdown(f"**{icon} {title}**")
            st.markdown(body)


def remember(points: list[str]):
    st.success("\n".join(f"- {p}" for p in points), title="Remember for the viva", icon=":material/school:")


def bars(df: pd.DataFrame, x: str, y: str, color: str | None = None, title: str = "", height: int = 320,
         y_max: float | None = None, colors: list[str] | None = None, sort=None, rule: float | None = None,
         rule_label: str = "", fmt: str = ".1f"):
    """Bar chart with value labels; grouped when `color` is given."""
    enc_y = alt.Y(f"{y}:Q", title=None, scale=alt.Scale(domain=[0, y_max]) if y_max else alt.Undefined,
                  axis=alt.Axis(grid=False, labels=False, ticks=False, domain=False))
    enc_x = alt.X(f"{x}:N", title=None, sort=sort, axis=alt.Axis(labelAngle=0, labelLimit=140, labelFontSize=12))
    base = alt.Chart(df).encode(x=enc_x, y=enc_y)
    if color:
        domain = list(dict.fromkeys(df[color]))  # keep the order the groups appear in the data
        scale = alt.Scale(domain=domain, range=colors) if colors else alt.Scale(domain=domain)
        base = base.encode(color=alt.Color(f"{color}:N", title=None, scale=scale, sort=domain, legend=alt.Legend(orient="top")),
                           xOffset=alt.XOffset(f"{color}:N", sort=domain))
        chart = base.mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
    else:
        chart = base.mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color=(colors or [GREEN])[0])
    labels = base.mark_text(dy=-7, fontSize=11, color="#1F2A1F").encode(text=alt.Text(f"{y}:Q", format=fmt))
    layers = [chart, labels]
    if rule is not None:
        r = alt.Chart(pd.DataFrame({"v": [rule], "t": [rule_label]}))
        layers += [r.mark_rule(strokeDash=[4, 4], color=GREY).encode(y="v:Q"),
                   r.mark_text(align="left", dx=4, dy=-6, color=GREY, fontSize=11).encode(y="v:Q", text="t:N", x=alt.value(2))]
    st.altair_chart(alt.layer(*layers).properties(title=title, height=height), width="stretch")
