# %%
import os
import pandas as pd
from tqdm import tqdm
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mtcnn import MTCNN
import random
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib


# %%
detector_mtcnn = MTCNN()
detector_haar = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
base_model = InceptionResNetV2(weights="imagenet", include_top=False, pooling="avg")


# %%
def get_video_metadata(directory, label):
    meta_data = []
    files = [f for f in os.listdir(directory) if f.lower().endswith(".mp4")]

    for filename in tqdm(files, desc=f"Scanning {label}"):
        filepath = os.path.join(directory, filename)
        cap = cv2.VideoCapture(filepath)

        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            meta_data.append({"filename": filename, "fps": fps, "label": label})
        cap.release()
    return meta_data


# %%
def is_metadata_fake(fps):
    tol = 0.00001
    if abs(fps - 29.0) < tol:
        return 1
    if abs(fps - 23.0) < tol:
        return 1
    if abs(fps - 29.97003) < tol:
        return 1
    return 0


# %%
def save_frames(row, base_dir="../storage/frames"):
    filename = row["filename"]
    label_str = "real" if row["label"] == 0 else "fake"
    video_path = f"../dataset/train/{label_str}/{filename}"
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
                cv2.imwrite(f"{base_dir}/{label_str}/{save_name}", frame)
    cap.release()


# %%
def crop_and_save_face_mtcnn(img_path, save_path):
    img = cv2.imread(img_path)
    if img is None:
        return

    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = detector_mtcnn.detect_faces(
            img_rgb, min_face_size=15, steps_threshold=[0.4, 0.5, 0.5]
        )
    except Exception:
        results = []
    if results:
        best_face = max(results, key=lambda x: x["confidence"])
        x, y, w, h = best_face["box"]
        source = "MTCNN"
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_haar = detector_haar.detectMultiScale(gray, 1.3, 4)
        if len(faces_haar) > 0:
            x, y, w, h = max(faces_haar, key=lambda rect: rect[2] * rect[3])
            source = "Haar"
        else:
            return None
    pad_w = int(w * 0.2)
    pad_h = int(h * 0.2)
    y1 = max(0, y - pad_h)
    y2 = min(img.shape[0], y + h + pad_h)
    x1 = max(0, x - pad_w)
    x2 = min(img.shape[1], x + w + pad_w)
    face_img = img[y1:y2, x1:x2]
    if face_img.size <= 0:
        return None
    face_img = cv2.resize(face_img, (224, 224))
    cv2.imwrite(save_path, face_img)
    return source


# %%
def extract_features(directory, sample_count=1000):
    features = []
    labels = []
    classes = ["real", "fake"]
    for label_name in classes:
        class_dir = os.path.join(directory, label_name)
        if not os.path.exists(class_dir):
            continue

        print(f"Extracting features form {label_name}...")
        img_files = [f for f in os.listdir(class_dir) if f.endswith(".jpg")][
            :sample_count
        ]

        for img_file in tqdm(img_files):
            img_path = os.path.join(class_dir, img_file)

            try:
                img = image.load_img(img_path, target_size=(299, 299))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                feature = base_model.predict(x, verbose=0)
                features.append(feature.flatten())
                labels.append(0 if label_name == "real" else 1)
            except Exception as e:
                print(f"Error processing {img_file}:{e}")

    return np.array(features), np.array(labels)


# %%
if __name__ == "__main__":
    extract_path = "../dataset"
    file_path = f"{extract_path}/train_labels.csv"

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        print("Dataset loaded. First 5 rows:")
        print(df.head())
    else:
        print(f"Error: {file_path} not found.")
        exit()

    base_dir = "../storage/frames"
    os.makedirs(f"{base_dir}/real", exist_ok=True)
    os.makedirs(f"{base_dir}/fake", exist_ok=True)

    real_path = "../dataset/train/real"
    fake_path = "../dataset/train/fake"

    print("Gathering metadata from video files")
    real_meta = get_video_metadata(real_path, 0)
    fake_meta = get_video_metadata(fake_path, 1)

    df = pd.DataFrame(real_meta + fake_meta)
    print(f"\nMetadata gathered for {len(df)} videos.")
    print(df.head())

    df["pred_stage1"] = df["fps"].apply(is_metadata_fake)
    hard_cases = df[df["pred_stage1"] == 0]

    print(f"Total Videos: {len(df)}")
    print(f"Caught by Metadata: {len(df[df['pred_stage1'] == 1])}")
    print(f"Hard Cases remaining: {len(hard_cases)}")

    print("\nExtracting frames for Hard Cases...")
    tqdm.pandas()
    hard_cases.progress_apply(save_frames, args=(base_dir,), axis=1)

    input_dir = "../storage/frames"
    output_dir = "../storage/faces"
    os.makedirs(f"{output_dir}/real", exist_ok=True)
    os.makedirs(f"{output_dir}/fake", exist_ok=True)

    classes = ["real", "fake"]
    stats = {"MTCNN": 0, "Haar": 0, "Skipped": 0}

    print("\nDetecting and Cropping Faces...")
    for label in classes:
        class_path = os.path.join(input_dir, label)
        if not os.path.exists(class_path):
            continue

        files = [f for f in os.listdir(class_path) if f.endswith(".jpg")]
        for filename in tqdm(files, desc=f"Cropping {label}"):
            in_path = os.path.join(class_path, filename)
            out_path = os.path.join(output_dir, label, filename)

            res = crop_and_save_face_mtcnn(in_path, out_path)
            if res:
                stats[res] += 1
            else:
                stats["Skipped"] += 1

    print(f"Face Extraction Stats: {stats}")

    print("\nExtracting deep learning features...")
    X_f, y_l = extract_features(output_dir, sample_count=2000)

    print("\nTraining SVM Classifier...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_f, y_l, test_size=0.2, random_state=43, stratify=y_l
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = SVC(kernel="rbf", probability=True, class_weight="balanced")
    clf.fit(X_train_s, y_train)

    joblib.dump(clf, "deepfake_svm_model.pkl")
    joblib.dump(scaler, "feature_scaler.pkl")

    print("\n--- Pipeline Complete ---")
    print(f"SVM Accuracy on Test Set: {clf.score(X_test_s, y_test):.2%}")
    print("Model and Scaler saved as .pkl files.")
