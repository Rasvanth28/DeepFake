from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
from deepfake_model_input import predict
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
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

@app.post("/predict_video")
async def predict_deepfake_video(video: UploadFile = File(...)):
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await video.read())
        temp_video_path = temp_video.name
        
    image_list = []
    cap = cv2.VideoCapture(temp_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames > 0:
        frame_indices = np.linspace(0, total_frames - 1, 5, dtype=int)
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (224, 224))
                image_list.append(frame)
    cap.release()
    os.remove(temp_video_path)
    
    if not image_list:
        return {"prediction": "Error processing video", "confidence": 0.0}
        
    label, confidence = predict(image_list)
    return {"prediction": label, "confidence": confidence}