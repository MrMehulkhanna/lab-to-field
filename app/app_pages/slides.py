import json

import streamlit as st

from common import ROOT

slides = sorted((ROOT / "app" / "slides").glob("slide-*.jpg"))
notes = json.loads((ROOT / "app" / "slides" / "notes.json").read_text())

n = st.slider("Slide", 1, len(slides), 1)
st.image(str(slides[n - 1]), width="stretch", caption=notes[n - 1]["title"], alt=notes[n - 1]["title"])
with st.container(border=True):
    st.markdown("**Explanation**")
    st.write(notes[n - 1]["notes"])

deck = ROOT / "docs" / "Lab_to_Field_presentation.pptx"
st.download_button("Download the presentation (.pptx)", deck.read_bytes(), file_name=deck.name,
                   mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                   icon=":material/download:")
