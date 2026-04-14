import cv2
from typing import Union
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pathlib import Path


IMG_PATH = Path("/app/dataset/capturas/result.jpg")

app = FastAPI()

@app.get("/camera")
def video_webcam(url: str):

    IMG_PATH.parent.mkdir(parents=True, exist_ok=True)

    url_camera = int(url) if url.isdigit() else url
    cap = cv2.VideoCapture(url_camera)

    ret, frame = cap.read()

    if not ret:
        cap.release()
        return None
    # Resize to speed up processing
    frame_resized = cv2.resize(frame, (640, 480))
    cv2.imwrite(str(IMG_PATH), frame_resized)
    
    cap.release()

    return {"status": "success", "message": "Imagen capturada"}