# %%
from pathlib import Path

import cv2
import os
from tqdm import tqdm
import pandas as pd
from preTraining import is_metadata_fake, crop_face_mtcnn, base_model
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
import joblib


# %%
def save_frames(row, base_dir="../storage/temp/frames"):
    filename = row["filename"]
    label_str = "real" if row["predicted-label"] == 0 else "fake"
    video_path = f"../storage/temp/videos/{filename}"
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > 0:
        frame_indices = np.linspace(0, total_frames - 1, 5, dtype=int)
        for i, idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (224, 224))
                save_name = f"{filename.split('.')[0]}_{i}.jpg"
                cv2.imwrite(f"{base_dir}/{save_name}", frame)
    cap.release()


# %%
def extract_features(imageList):
    features = []
    for img_item in imageList:
        try:
            image_rgb = cv2.cvtColor(img_item, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image_rgb, (299, 299))
            x = image.img_to_array(image_resized)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            feature = base_model.predict(x, verbose=0)
            features.append(feature.flatten())
        except Exception as e:
            print(f"Error processing image matrix {e}")
    return np.array(features)


# %%
def predict(imageList):
    faceImages = []
    for img in imageList:
        face = crop_face_mtcnn(img)
        if face is not None:
            faceImages.append(face)
    if not faceImages:
        return "No faces detected", 0.0

    path = Path(__file__).resolve().parent

    features = extract_features(faceImages)
    clf = joblib.load(path / "../model/deepfake_svm_model.pkl")
    scaler = joblib.load(path / "../model/feature_scaler.pkl")

    X_scaled = scaler.transform(features)
    predictions = clf.predict(X_scaled)
    probabilities = clf.predict_proba(X_scaled)

    final_prediction = pd.Series(predictions).mode()[0]
    avg_confidence = probabilities[:, 1].mean()

    label = "Fake" if final_prediction == 1 else "Real"
    return label, float(avg_confidence)


# %%
