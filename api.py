

from typing import Union
from fastapi import FastAPI
from fastapi.responses import FileResponse
# from pathlib import Path
# import os
import cv2

from fastapi.responses import StreamingResponse
from camera_service import video_webcam
from inference_service import inference

app = FastAPI()


@app.get("/prueba")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.get("/camera")
def capture_image():
    # os.system('python camera_service.py')
    # image_path = Path("C:/Users/sersatr/Repositorios/vision-yolo-system/dataset/capturas/result.jpg")
 
    img = video_webcam()
    # img_path = Path(img)
    if img is None or not img.is_file():
        return {"error": "No se pudo conectar con la cámara IP o guardar la imagen"}
    return FileResponse(img)

    '''
    _, buffer = cv2.imencode('.jpg', img)
    if not _:
        continue
    img_data = buffer.tobytes()
    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + img_data + b'\r\n')
    return StreamingResponse(
            motor_de_streaming(), 
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
        '''
 

@app.get("/inference")
def run_inference():   
    # os.system('python inference-service
    # image_path = Path("C:/Users/sersatr/Repositorios/vision-yolo-system/runs/detect/predict8/result.jpg")
    # imagencita = capture_image()
    # image_path1 = Path(imagencita)
    resultado = inference()

    if not resultado.is_file():
        return {"error": "Image not found on the server"}
    return FileResponse(resultado)







