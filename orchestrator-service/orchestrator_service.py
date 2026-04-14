

from fastapi import FastAPI
from fastapi.responses import Response
import requests

app = FastAPI()

# Añadimos 'url' como parámetro. Le ponemos "0" como valor por defecto.
@app.get("/vision_yolo_app")
def orchestrator(url: str = "0"):
    
    # 1. Disparar el servicio de cámara pasándole la URL que recibimos
    try:
        # Usamos f-strings para inyectar la variable 'url' en la petición
        res_camara = requests.get(f"http://camera_service:8001/camera?url={url}")
        if res_camara.status_code != 200:
            return {"error": "El servicio de cámara falló al capturar la imagen."}
    except Exception as e:
        return {"error": f"No se pudo conectar con camera_service: {e}"}

    # 2. Disparar el servicio de inferencia
    try:
        res_inferencia = requests.get("http://inference_service:8002/inference")
        if res_inferencia.status_code != 200:
            return {"error": "El servicio de inferencia falló al procesar la imagen."}
        
        # 3. Devolver la imagen final
        return Response(content=res_inferencia.content, media_type="image/jpeg")
        
    except Exception as e:
        return {"error": f"No se pudo conectar con inference_service: {e}"}

