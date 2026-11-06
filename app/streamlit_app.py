"""Lab to Field: crop disease detection that works on real farm photos.

Run:  uv run streamlit run app/streamlit_app.py
"""
import streamlit as st

st.set_page_config(page_title="Lab to Field · Crop Disease Check", page_icon=":material/eco:", layout="wide")

pages = {
    "": [st.Page("app_pages/overview.py", title="Overview", icon=":material/home:", default=True)],
    "Explained step by step": [
        st.Page("app_pages/p1_problem.py", title="1 · Problem and data", icon=":material/search:"),
        st.Page("app_pages/p2_cnn.py", title="2 · CNN from scratch", icon=":material/layers:"),
        st.Page("app_pages/p3_transfer.py", title="3 · Transfer learning", icon=":material/swap_horiz:"),
        st.Page("app_pages/p4_augment.py", title="4 · Data augmentation", icon=":material/auto_fix_high:"),
        st.Page("app_pages/p5_training.py", title="5 · Training and metrics", icon=":material/tune:"),
        st.Page("app_pages/p6_final.py", title="6 · Final model and app", icon=":material/rocket_launch:"),
    ],
    "Use it": [
        st.Page("app_pages/diagnose.py", title="Diagnose a leaf", icon=":material/photo_camera:"),
        st.Page("app_pages/slides.py", title="Slides", icon=":material/slideshow:"),
    ],
}
page = st.navigation(pages, position="top")
st.title(page.title, anchor=False)
page.run()
