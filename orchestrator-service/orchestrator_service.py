

from typing import Union
from fastapi import FastAPI
from fastapi.responses import FileResponse
# from pathlib import Path
# import os
import cv2

from fastapi.responses import StreamingResponse


app = FastAPI()


@app.get("/vision_yolo_app")
def orchestrator():

    resultado = NULL

    if not resultado.is_file():
        return {"error": "Image not found on the server"}
    return FileResponse(resultado)



