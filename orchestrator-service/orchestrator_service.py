

from typing import Union
from fastapi import FastAPI
from fastapi.responses import FileResponse
import requests
# from pathlib import Path
# import os
import cv2

from fastapi.responses import StreamingResponse


app = FastAPI()

@app.get("/vision_yolo_app")
def orchestrator():
    # 1. Disparar el servicio de cámara. 
    # Usamos "camera_service" porque es el nombre del contenedor en el docker-compose.yml
    # Le pasamos url=0 suponiendo que es la webcam local por defecto.
    try:
        res_camara = requests.get("http://camera_service:8001/camera?url=0")
        if res_camara.status_code != 200:
            return {"error": "El servicio de cámara falló al capturar la imagen."}
    except Exception as e:
        return {"error": f"No se pudo conectar con camera_service: {e}"}

    # En este punto, la cámara ya ha guardado la foto en /app/dataset/capturas/result.jpg (volumen compartido)

    # 2. Disparar el servicio de inferencia.
    # inference_service leerá la foto del disco, aplicará YOLO y nos devolverá la imagen procesada.
    try:
        res_inferencia = requests.get("http://inference_service:8002/inference")
        if res_inferencia.status_code != 200:
            return {"error": "El servicio de inferencia falló al procesar la imagen."}
        
        # 3. Devolver la imagen final directamente a quien haya llamado al orquestador
        return Response(content=res_inferencia.content, media_type="image/jpeg")
        
    except Exception as e:
        return {"error": f"No se pudo conectar con inference_service: {e}"}

