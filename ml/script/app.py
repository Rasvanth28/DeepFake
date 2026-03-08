from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
from deepfake_model_input import predict
from fastapi.middleware import cors

app = FastAPI()

app.add_middleware(
    cors.CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict_deepfake(frames: list[UploadFile] = File(...)):
    image_list = []
    for frame in frames:
        contents = await frame.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            image_list.append(img)
    label, confidence = predict(image_list)

    return {"prediction": label, "confidence": confidence}
