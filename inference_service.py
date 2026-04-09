from ultralytics import YOLO
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

RESULT_PATH = BASE_DIR / "dataset" / "inferencias" / "result.jpg"
MODEL_PATH = BASE_DIR / "train" / "weights" / "best.pt"
IMG_PATH = BASE_DIR / "dataset" / "capturas" / "result.jpg"

modelo_entrenado = YOLO(MODEL_PATH)

def inference():
    
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    resultados = modelo_entrenado.predict(IMG_PATH)
    inferencia = resultados[0].plot()
    cv2.imwrite(str(RESULT_PATH), inferencia)

    return RESULT_PATH

