"""Host the Blueline API on Modal: a public HTTPS URL for the frontend, one container kept warm for the judges.

    modal deploy blueline_modal.py        -> prints https://<you>--blueline-api-serve.modal.run
    Open that URL: it serves the Blueline page itself (index.html from the frontend repo, copied next to this file)
    and the page detects the reader on the same address, so there is nothing to configure.

The Gemini key is read from .env next to this file at deploy time (stored as a Modal secret, never in the code).
Stop it after the event:  modal app stop blueline-api
"""
import os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ["index.html", "blueline_api.py", "ask.py", "schematic.py", "arcs.py", "scale_bar.py", "drawing_tools.py", "seg_data.py", "seg_post.py", "seg_predict.py"]
WEIGHTS = [f for f in ("unet_swiss.pt", "yolo_swiss.pt", "yolo_cubi.pt") if os.path.exists(os.path.join(HERE, f))]   # trained room segmenter + detectors

image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("libgl1", "libglib2.0-0")
         .pip_install("numpy", "opencv-python-headless", "scipy", "shapely", "pillow", "pymupdf", "google-genai", "rapidocr_onnxruntime"))
if WEIGHTS:
    image = image.pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cpu").pip_install("ultralytics", "segmentation-models-pytorch")
for f in FILES + WEIGHTS:
    if os.path.exists(os.path.join(HERE, f)):
        image = image.add_local_file(os.path.join(HERE, f), f"/app/{f}")

app = modal.App("blueline-api")


@app.function(image=image, secrets=[modal.Secret.from_dotenv(HERE)], cpu=4, memory=8192, min_containers=1, timeout=3600)
@modal.concurrent(max_inputs=8)
@modal.web_server(8000, startup_timeout=180)
def serve():
    import subprocess
    env = dict(os.environ, PORT="8000")
    env["FP_UNET"] = "/app/unet_swiss.pt"; env["FP_YOLO"] = "/app/yolo_swiss.pt"
    subprocess.Popen(["python3", "/app/blueline_api.py"], cwd="/app", env=env)
