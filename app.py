"""Local Streamlit interface for the MediaPipe Face Landmarker."""
import hashlib
from pathlib import Path

import streamlit as st
from PIL import Image

from landmarks import detect, export_csv, export_json, export_png, load_image, overlay

st.set_page_config(page_title="Facial Landmark Explorer", layout="wide")
st.title("Facial Landmark Explorer")
st.caption("Upload a photo to inspect and export landmarks for one face. Photos are processed by the server running this app.")

upload = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png", "webp"],
                          max_upload_size=25)
if upload is None:
    st.session_state.pop("analysis", None)
    st.info("Start with a well-lit photo where one face is clearly visible.")
    st.stop()

data = upload.getvalue()
sensitive = st.sidebar.checkbox(
    "Increase detection sensitivity",
    help="Try this if a profile photo goes undetected. It may also accept false detections.",
)
key = hashlib.sha256(data).digest(), sensitive
analysis = st.session_state.get("analysis")
if analysis is None or analysis[0] != key:
    st.session_state.pop("analysis", None)
    try:
        image = load_image(data)
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        st.error(f"Could not open this image. {error}")
        st.stop()

    try:
        with st.spinner("Finding landmarks. The first run may download the MediaPipe model."):
            points, pose = detect(image, sensitive=sensitive)
    except Exception as error:
        st.error(f"Could not run MediaPipe. {error}")
        st.info("See Model setup and Troubleshooting in README.md.")
        if st.button("Retry"):
            st.rerun()
        st.stop()

    analysis = key, image, points, pose
    st.session_state.analysis = analysis

_, image, points, pose = analysis
if not points:
    st.warning("No face was detected. Try increasing detection sensitivity or using a clearer photo with a slightly smaller head turn.")
    st.image(image, caption="Uploaded photo")
    st.stop()

with st.sidebar:
    st.header("Overlay settings")
    count = st.radio("Landmarks", [468, 478], index=1,
                     format_func=lambda n: "468 face points" if n == 468 else "478 with iris points")
    radius = st.slider("Point radius (pixels)", 1, 20, 3)
    labels = st.selectbox("Index labels", ["Off", "Every 10th point", "Every point"])
    label_step = {"Off": 0, "Every 10th point": 10, "Every point": 1}[labels]
    st.caption("Iris points are orange. Face points are mint green.")

selected = points[:count]
if pose is not None:
    direction = f" · facing {pose['direction']}" if pose["direction"] else ""
    st.write(f"**Estimated view:** {pose['view']}{direction} ({pose['yaw_degrees']:+.0f}° yaw)")
st.caption("On angled photos, points on the hidden side of the face may be inferred by the model.")
annotated = overlay(image, selected, radius, label_step)
preview, original = st.tabs(["Landmarks", "Original photo"])
preview.image(annotated, caption=f"{count} landmarks · {image.width} × {image.height} pixels")
original.image(image)

st.subheader("Downloads")
stem = Path(upload.name).stem or "photo"
filename = f"{stem}-landmarks-{count}"
png, csv, json = st.columns(3)
png.download_button("Annotated PNG", export_png(annotated), f"{filename}.png",
                    "image/png", on_click="ignore")
csv.download_button("Landmarks CSV", export_csv(selected), f"{filename}.csv",
                    "text/csv", on_click="ignore")
json.download_button("Landmarks JSON", export_json(image, selected, pose), f"{filename}.json",
                     "application/json", on_click="ignore")
