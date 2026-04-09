import cv2
# import IPython.display as display
# from PIL import Image
# import io
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
IMG_PATH = BASE_DIR / "dataset" / "capturas" / "result.jpg"

'''

def webcam():

    IMG_PATH.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)  # Open the default camera
    ret, frame = cap.read()

    # Resize to speed up processing
    frame_resized = cv2.resize(frame, (640, 480))
    cv2.imwrite(str(IMG_PATH), frame_resized)

    cap.release()
    
    return IMG_PATH

# img, img_data = webcam()

# img.save("C:/Users/sersatr/Repositorios/vision-yolo-system/dataset/capturas/result.jpg")

import IPython.display as display
from IPython.display import Image as IPyImage

def show_webcam():
    cap = cv2.VideoCapture(0)  # Open the default camera

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize to speed up processing
            frame = cv2.resize(frame, (640, 480))

            # Convert frame to JPEG format for Jupyter display
            _, buffer = cv2.imencode('.jpg', frame)
            img_data = buffer.tobytes()

            # Display in Jupyter Notebook
            display.clear_output(wait=True)
            display.display(IPyImage(data=img_data))
            imagen = IPyImage(data=img_data)
            return imagen
    except KeyboardInterrupt:
        print("Webcam stopped")

    finally:
        cap.release()

'''

def video_webcam():

    IMG_PATH.parent.mkdir(parents=True, exist_ok=True)

    url_camera="http://200.46.196.243:80/mjpg/video.mjpg"

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