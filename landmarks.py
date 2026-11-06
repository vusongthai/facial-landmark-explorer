"""Image loading, MediaPipe inference, and export helpers."""
import csv
import io
import json
import math
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


def estimate_view(matrix):
    """Classify the direction of the canonical face's forward axis."""
    x, _, z = matrix[:3, 2]
    if not np.isfinite([x, z]).all() or np.hypot(x, z) < 1e-6:
        return None
    yaw = float(np.degrees(np.arctan2(x, z)))
    if abs(yaw) < 20:
        view = "Front"
    elif abs(yaw) < 65:
        view = "Three-quarter"
    else:
        view = "Profile"
    direction = None if view == "Front" else ("image right" if yaw > 0 else "image left")
    return {"view": view, "direction": direction, "yaw_degrees": round(yaw, 1)}


def detect(image, *, sensitive=False):
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=model_path().read_bytes()),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.35 if sensitive else 0.5,
        min_face_presence_confidence=0.35 if sensitive else 0.5,
        output_facial_transformation_matrixes=True,
    )
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        for angle in ((0, 45, -45) if sensitive else (0,)):
            candidate = image.rotate(angle, resample=Image.Resampling.BILINEAR) if angle else image
            frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(candidate))
            result = landmarker.detect(frame)
            if result.face_landmarks:
                break
    if not result.face_landmarks:
        return [], None
    points = result.face_landmarks[0]
    if len(points) != 478:
        raise ValueError(f"Expected 478 landmarks from the model, received {len(points)}.")
    width, height = image.size
    radians = math.radians(angle)
    cosine, sine = math.cos(radians), math.sin(radians)
    landmarks = []
    for index, point in enumerate(points):
        dx, dy = (point.x - 0.5) * width, (point.y - 0.5) * height
        x_px = cosine * dx - sine * dy + width / 2
        y_px = sine * dx + cosine * dy + height / 2
        landmarks.append(dict(index=index, x=x_px / width, y=y_px / height,
                              z=point.z, x_px=x_px, y_px=y_px))
    matrices = result.facial_transformation_matrixes
    if not len(matrices):
        return landmarks, None
    matrix = matrices[0].copy()
    matrix[0, 2] = cosine * matrix[0, 2] + sine * matrix[1, 2]
    pose = estimate_view(matrix)
    if pose is not None and angle:
        pose["rotation_retry_degrees"] = angle
    return landmarks, pose


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


def export_json(image, points, pose=None):
    return json.dumps({
        "image_width": image.width,
        "image_height": image.height,
        "landmark_count": len(points),
        "head_pose": pose,
        "landmarks": points,
    }, indent=2, allow_nan=False).encode("utf-8")
