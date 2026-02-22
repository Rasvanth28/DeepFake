# %%
import cv2
import os
from tqdm import tqdm
import pandas as pd
from preTraining import is_metadata_fake, crop_and_save_face_mtcnn, base_model
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
def extract_features(filepath):
    features = []

    img_files = [f for f in os.listdir(filepath) if f.endswith(".jpg")]
    for img_file in tqdm(img_files):
        img_path = os.path.join(filepath, img_file)
        try:
            img = image.load_img(img_path, target_size=(299, 299))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            feature = base_model.predict(x, verbose=0)
            features.append(feature.flatten())
        except Exception as e:
            print(f"Error processing {img_file}:{e}")
    return np.array(features)


# %%
def predict(directory):
    data = []
    videopath = os.path.join(directory, "videos")
    framepath = os.path.join(directory, "frames")
    facepath = os.path.join(directory, "faces")

    os.makedirs(framepath, exist_ok=True)
    os.makedirs(facepath, exist_ok=True)

    if not os.path.exists(videopath):
        print(f"Error: {videopath} does not exist.")
        return None

    videofiles = os.listdir(videopath)
    framefiles = os.listdir(framepath)
    facefiles = os.listdir(facepath)
    for filename in tqdm(videofiles):
        if filename.endswith(".mp4"):
            filepath = os.path.join(videopath, filename)

            cap = cv2.VideoCapture(filepath)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                data.append({"filename": filename, "fps": fps})
            cap.release()
    df = pd.DataFrame(data)

    if df.empty:
        print("No video files found.")
        return df

    df["predicted-label"] = df["fps"].apply(is_metadata_fake)
    print(df.head())
    tqdm.pandas()
    df.progress_apply(save_frames, args=(framepath,), axis=1)
    for frames in tqdm(framefiles):
        img_in = os.path.join(framepath, frames)
        img_out = os.path.join(facepath, frames)
        crop_and_save_face_mtcnn(img_path=img_in, save_path=img_out)
    features = extract_features(facepath)
    clf = joblib.load("../model/deepfake_svm_model.pkl")
    scaler = joblib.load("../model/feature_scaler.pkl")
    X_scaled = scaler.transform(features)
    prediciton = clf.predict(X_scaled)
    probability = clf.predict_proba(X_scaled)
    fake_confidence = probability[:, 1]
    face_filenames = [f for f in os.listdir(facepath) if f.endswith(".jpg")]
    frames_df = pd.DataFrame(
        {
            "full_filename": face_filenames,
            "prediction": prediciton,
            "confidence": fake_confidence,
        }
    )

    frames_df["video_name"] = frames_df["full_filename"].apply(
        lambda x: x.split("_")[0] + ".mp4"
    )

    video_results = (
        frames_df.groupby("video_name")
        .agg({"prediction": lambda x: x.mode()[0], "confidence": "mean"})
        .reset_index()
    )

    print(video_results)


directory = "../storage/temp/"
predict(directory=directory)
