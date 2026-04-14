from ultralytics import YOLO
import cv2
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse


IMG_PATH = Path("/app/dataset/capturas/result.jpg")
RESULT_PATH = Path("/app/dataset/inferencias/result.jpg")
MODEL_PATH = Path("/app/train/weights/best.pt")

modelo_entrenado = YOLO(MODEL_PATH)

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok", "endpoints": ["/inference", "/health"]}

@app.get("/inference")
def inference():
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    resultados = modelo_entrenado.predict(IMG_PATH)
    inferencia = resultados[0].plot()
    cv2.imwrite(str(RESULT_PATH), inferencia)

    return FileResponse(str(RESULT_PATH))

@app.get("/health")
def health():
    return {"status": "healthy"}


#@app.get("/health")
#def health():
#    async def health_check() -> dict:
#        return {"status": "healthy"}