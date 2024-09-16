import os
import requests

# URLs for downloading YOLO model files
YOLO_WEIGHTS_URL = "https://pjreddie.com/media/files/yolov3.weights"
YOLO_CFG_URL = "https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true"
COCO_NAMES_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

# Directory to save the model
MODEL_DIR = "./models"

def download_file(url, filepath):
    """Downloads a file from the given URL."""
    response = requests.get(url)
    with open(filepath, "wb") as f:
        f.write(response.content)
    print(f"Downloaded: {filepath}")

# Create models directory if it doesn't exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Download YOLO model weights, configuration, and labels
download_file(YOLO_WEIGHTS_URL, f"{MODEL_DIR}/yolov3.weights")
download_file(YOLO_CFG_URL, f"{MODEL_DIR}/yolov3.cfg")
download_file(COCO_NAMES_URL, f"{MODEL_DIR}/coco.names")

print("YOLO model files downloaded successfully.")
