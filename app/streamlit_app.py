"""Crop disease detection: take or upload a leaf photo, get the disease, see where the model looked.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference import (  # noqa: E402
    InvalidImage,
    available_models,
    gradcam_overlay,
    leaf_fraction,
    load_disease_info,
    load_model,
    predict,
    pretty,
    read_upload,
    top_k,
    unsure_reason,
)

st.set_page_config(page_title="Crop Disease Detection", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def get_model(name: str):
    return load_model(name)


@st.cache_data
def get_info() -> dict:
    return load_disease_info()


st.title("Crop disease detection")
st.caption("Photograph one leaf. The model names the disease from 38 classes across 14 crops "
           "and shows which part of the leaf drove its answer.")

names = available_models()
if not names:
    st.error("No trained models found in app/models/. Run `python scripts/fetch_models.py` "
             "(or `modal run modal_app.py::collect` and `python scripts/export_models.py`) first.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Model", names, format_func=lambda n: get_model(n).display)
    lm = get_model(model_name)
    show_cam = st.checkbox("Show Grad-CAM heatmap", value=True)
    cam_method = st.radio("Heatmap method", ["gradcam", "gradcam++"],
                          format_func=lambda m: {"gradcam": "Grad-CAM", "gradcam++": "Grad-CAM++"}[m])
    tau = st.slider("'Unsure' threshold", 0.0, 1.0, float(round(lm.unsure_threshold, 2)), 0.01,
                    help="Below this top-1 confidence the app asks for a new photo. "
                         "Default comes from the validation set.")
    with st.expander("About and limits"):
        st.markdown(
            "- Trained on PlantVillage: lab photos of single leaves on plain backgrounds.\n"
            "- Real field photos with soil, hands or many leaves will be less accurate.\n"
            "- Knows only 38 classes (14 crops). Other crops get a wrong or 'unsure' answer.\n"
            "- Disease notes are a **draft** and not yet verified. Ask a local agricultural "
            "extension officer before treating plants.\n"
            "- Images stay in memory and are never saved."
        )

tab_cam, tab_up = st.tabs(["Take photo", "Upload photo"])
with tab_cam:
    shot = st.camera_input("Point the camera at one leaf")
with tab_up:
    upload = st.file_uploader("Choose a leaf photo", type=["jpg", "jpeg", "png", "webp"])

source = shot or upload
if source is None:
    st.info("Take or upload a photo to start. Tips: one leaf, filling the frame, daylight, plain background.")
    st.stop()

try:
    img = read_upload(source.getvalue())
except InvalidImage as e:
    st.error(str(e))
    st.stop()

try:
    probs, x = predict(lm, img)
except Exception as e:  # keep the UI alive on unexpected model errors
    st.error(f"Prediction failed: {e}")
    st.stop()

top = top_k(probs, lm.classes, 3)
reason = unsure_reason(probs, tau)
if leaf_fraction(img) < 0.05:
    st.warning("No leaf detected? Very few green, yellow or brown pixels in this photo.")

left, right = st.columns(2)
with left:
    st.image(img, caption="Your photo", use_container_width=True)
with right:
    if show_cam:
        try:
            cls_idx = lm.classes.index(top[0][0])
            st.image(gradcam_overlay(lm, x, cls_idx, cam_method),
                     caption=f"Where {lm.display} looked (red = most important)", use_container_width=True)
        except Exception as e:
            st.warning(f"Heatmap unavailable: {e}")

if reason:
    st.warning(f"**Unsure, please retake the photo** ({reason}). "
               "Use one leaf, fill the frame, daylight, and a plain background.")

st.subheader("Top 3 predictions")
for cls, p in top:
    crop, disease = pretty(cls)
    st.write(f"**{crop}: {disease}** {p:.0%}")
    st.progress(p)

if not reason:
    info = get_info().get(top[0][0])
    if info:
        crop, disease = pretty(top[0][0])
        with st.container(border=True):
            st.markdown(f"### {crop}: {info['disease']}")
            if info["healthy"]:
                st.success(info["management"])
            else:
                st.markdown(f"**Cause.** {info['cause']}")
                st.markdown(f"**Symptoms.** {info['symptoms']}")
                st.markdown(f"**Management.** {info['management']}")
            if info.get("source_url"):
                st.markdown(f"[Source]({info['source_url']})")
            if not info.get("verified"):
                st.caption("Draft information, not yet verified against an agricultural extension source.")

st.caption("Trained on lab images; accuracy on field photos will be lower.")
