import cv2
from typing import Union
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
IMG_PATH = BASE_DIR / "dataset" / "capturas" / "result.jpg"

app1 = FastAPI()

@app1.get("/camera")
def video_webcam(url: str):

    IMG_PATH.parent.mkdir(parents=True, exist_ok=True)

    url_camera = url

    cap = cv2.VideoCapture(url_camera)  # Open the default camera "0", si no pues metemos la url
    ret, frame = cap.read()

    if not ret:
        cap.release()
        return None
    # Resize to speed up processing
    frame_resized = cv2.resize(frame, (640, 480))
    cv2.imwrite(str(IMG_PATH), frame_resized)
    
    cap.release()

    return IMG_PATH