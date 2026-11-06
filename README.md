# Facial Landmark Explorer

A Streamlit app for extracting facial landmarks from one photo at a time. Run it locally on Windows or deploy it to Streamlit Community Cloud.

## Install and launch on Windows

1. Install [64-bit Python 3.11 for Windows](https://www.python.org/downloads/windows/) and enable **Add Python to PATH** during setup.
2. Clone or download this repository to a writable folder such as Documents.
3. Double-click `launch.bat`. On the first launch, it creates a project-local `.venv` and installs the packages in `requirements.txt`. Leave the launcher window open while you use the app.
4. If your browser does not open automatically, visit [http://localhost:8501](http://localhost:8501).
5. To stop the app, press Ctrl+C in the launcher window.

The launcher binds to the local computer and disables Streamlit usage statistics. Initial setup needs an internet connection. Photos are processed locally and the app keeps the current photo and landmarks in its browser session's server memory.

## Deploy to Streamlit Community Cloud

1. Push the project files to a GitHub repository with `app.py` at the repository root.
2. At [Streamlit Community Cloud](https://share.streamlit.io/), connect your GitHub account and choose **Create app**.
3. Select the repository and its `main` branch. Enter `app.py` as the main file path.
4. Open **Advanced settings**, select **Python 3.11**, and deploy.

Community Cloud installs the Linux libraries listed in `packages.txt`. The app downloads the model when the first photo is submitted. On a cloud deployment, uploads are sent to and processed by the hosted Streamlit server.

## Use the app

- Upload a JPG, PNG, or WebP image up to 25 MB and 25 megapixels. Use a well-lit photo with one clearly visible face.
- Choose **468 face points** or **478 with iris points**. Both use the same MediaPipe Tasks model. The 468 option keeps indices 0–467, and the 478 option also includes indices 468–477.
- Set the dot radius in image pixels and show every index, every tenth index, or no labels. The **Original photo** tab lets you compare the overlay to the upload.
- Download a full-resolution annotated PNG or the selected points as CSV or JSON. The files use the same 468 or 478 selection shown in the preview.

## Model setup

On the first upload, the app downloads the [Google Face Landmarker model](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task) to `models/face_landmarker.task`. Later uploads reuse that file. For manual setup, download the linked model, create the `models` folder next to `app.py`, and place the model there with that exact filename. Once the dependencies and model are installed, the app works offline.

See the [MediaPipe Face Landmarker Python guide](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python) for details on the Tasks API.

## Export coordinates

Front, three-quarter, and profile photos are mapped individually. The JSON includes an estimated view and yaw in `head_pose` when available. Positive yaw points toward image right. If a profile goes undetected, try **Increase detection sensitivity** below the uploader. It lowers the detection thresholds and can retry at alternate rotations, mapping the result back to the original photo. Check the overlay before using it. The model can infer hidden points, and accuracy can drop on near-90° profiles.

| Field | Meaning |
| --- | --- |
| `index` | Zero-based MediaPipe landmark ID |
| `x`, `y` | Horizontal and vertical coordinates normalized by image width and height |
| `z` | Relative model depth, scaled like `x`, with smaller values closer to the camera |
| `x_px`, `y_px` | Floating-point image coordinates measured from the top-left, with positive y down |

The JSON includes the image dimensions and landmark count. The PNG and coordinate exports use the image orientation after applying the photo's EXIF rotation.

## Troubleshooting

- **Python is missing.** Install 64-bit Python 3.11 and reopen the launcher. If you moved the project after its first launch, remove the project's `.venv` folder and launch again to rebuild it.
- **The model fails to download or load.** Use the manual model setup above. If a previous model file is damaged, replace it with a fresh copy and click **Retry**.
- **Port 8501 is busy.** Stop the other local app or change the port in `launch.bat` and use the matching browser address.
- **No face detected.** Try a sharper, front-facing image with less occlusion.
