from ultralytics import YOLO
import cv2
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse

BASE_DIR = Path(__file__).resolve().parent

RESULT_PATH = BASE_DIR / "dataset" / "inferencias" / "result.jpg"
MODEL_PATH = BASE_DIR / "train" / "weights" / "best.pt"
IMG_PATH = BASE_DIR / "dataset" / "capturas" / "result.jpg"

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