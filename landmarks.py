"""Image loading, MediaPipe inference, and export helpers."""
import csv
import io
import json
import os
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen

import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageOps

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_landmarker.task"
FIELDS = ("index", "x", "y", "z", "x_px", "y_px")


def model_path():
    if MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > 0:
        return MODEL_PATH

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with urlopen(MODEL_URL, timeout=45) as response:
            with tempfile.NamedTemporaryFile(dir=MODEL_PATH.parent, delete=False) as temporary:
                temporary_path = Path(temporary.name)
                shutil.copyfileobj(response, temporary)
                size = temporary.tell()
                expected = response.headers.get("Content-Length")
                if not size or (expected and size != int(expected)):
                    raise OSError("The model download was incomplete.")
        os.replace(temporary_path, MODEL_PATH)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return MODEL_PATH


def load_image(data):
    with Image.open(io.BytesIO(data)) as source:
        if source.width * source.height > 25_000_000:
            raise ValueError("Choose an image with 25 megapixels or fewer.")
        return ImageOps.exif_transpose(source).convert("RGB")


def detect(image):
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=model_path().read_bytes()),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
    )
    frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(image))
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(frame)
    if not result.face_landmarks:
        return []
    points = result.face_landmarks[0]
    if len(points) != 478:
        raise ValueError(f"Expected 478 landmarks from the model, received {len(points)}.")
    return [
        dict(index=index, x=p.x, y=p.y, z=p.z,
             x_px=p.x * image.width, y_px=p.y * image.height)
        for index, p in enumerate(points)
    ]


def overlay(image, points, radius, label_step):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for point in points:
        x, y = point["x_px"], point["y_px"]
        color = "#ffb454" if point["index"] >= 468 else "#4fffd3"
        draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                     fill=color, outline="#101820")
    if label_step:
        for point in points[::label_step]:
            draw.text((point["x_px"] + radius + 1, point["y_px"] - radius),
                      str(point["index"]), fill="white", stroke_width=1, stroke_fill="black")
    return annotated


def export_png(image):
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def export_csv(points):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(points)
    return output.getvalue().encode("utf-8")


def export_json(image, points):
    return json.dumps({
        "image_width": image.width,
        "image_height": image.height,
        "landmark_count": len(points),
        "landmarks": points,
    }, indent=2, allow_nan=False).encode("utf-8")
