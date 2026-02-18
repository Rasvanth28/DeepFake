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

# %%
# Extracting the dataset

extract_path = "../dataset"
print("Files found:", os.listdir(extract_path))

# %%
file_path = f"{extract_path}/train_labels.csv"
df = pd.read_csv(file_path)
print(df.columns.tolist())
df.info()
df.head()

# %%
real_path = "../dataset/train/real"
fake_path = "../dataset/train/fake"

data = []


def extract_video_features(directory, label):
    files = os.listdir(directory)
    print(f"Processing {len(files)} videos from {directory}")
    for filename in tqdm(files):
        if filename.endswith(".mp4"):
            filepath = os.path.join(directory, filename)
            cap = cv2.VideoCapture(filepath)
            if cap.isOpened():
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

                duration = frame_count / fps if fps > 0 else 0
                file_size = os.path.getsize(filepath) / (1024 * 1024)
                bitrate = (file_size * 8) / duration if duration > 0 else 0

                data.append(
                    {
                        "filename": filename,
                        "width": width,
                        "height": height,
                        "fps": fps,
                        "frame_count": frame_count,
                        "duration": duration,
                        "file_size_mb": file_size,
                        "bitrate": bitrate,
                        "label": label,
                    }
                )
            cap.release()


if os.path.exists(real_path) and os.path.exists(fake_path):
    extract_video_features(real_path, 0)
    extract_video_features(fake_path, 1)

    df = pd.DataFrame(data)
    print("\nFeature Extraction Complete!")
    print(df.head())
else:
    print("Error:Could not find 'train/real' or 'train/fake' folders.")


# %%
X = df.drop(["label", "filename"], axis=1)
Y = df["label"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, Y_train)

importances = rf.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Which Metadata Parameter is Most Vital for Detection?")
plt.bar(range(X.shape[1]), importances[indices], align="center", color="teal")
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45)
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()

print("\nRanking Results")
for i in indices:
    print(f"{feature_names[i]}:{importances[i]:.4f}")


# %%
sns.set_style("whitegrid")
plt.Figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
sns.kdeplot(
    data=df[df["label"] == 0]["fps"], label="Real", fill=True, color="blue", alpha=0.3
)
sns.kdeplot(
    data=df[df["label"] == 1]["fps"], label="Fake", fill=True, color="red", alpha=0.3
)
plt.title("Distribution of FPS (Real vs Fake)")
plt.xlabel("Frames Per Second")
plt.legend()

plt.subplot(1, 2, 2)
sns.kdeplot(
    data=df[df["label"] == 0]["bitrate"],
    label="Real",
    fill=True,
    color="blue",
    alpha=0.3,
)
sns.kdeplot(
    data=df[df["label"] == 1]["bitrate"],
    label="Fake",
    fill=True,
    color="red",
    alpha=0.3,
)
plt.title("Distribution of Bitrate (Real vs Fake)")
plt.xlabel("Bitrate (MB/s)")
plt.legend()

plt.tight_layout()
plt.show()

print("Average Stats\n")
print(df.groupby("label")[["fps", "bitrate"]].mean())

# %%
print("Standard Deviation\n")
print(df.groupby("label")[["fps", "bitrate"]].std())
print("FPS Counts (Real Videos)\n")
print(df[df["label"] == 0]["fps"].value_counts().head(5))
print("FPS Counts (Fake Videos)\n")
print(df[df["label"] == 1]["fps"].value_counts().head(5))


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


df["pred_stage1"] = df["fps"].apply(is_metadata_fake)
accuracy = accuracy_score(df["label"], df["pred_stage1"])
conf_matrix = confusion_matrix(df["label"], df["pred_stage1"])

print(f"Stage 1 Accuracy: {accuracy:.2%}")
print("Confusion matrix\n")
print(f"True Negatives (Real videos correctly ignored): {conf_matrix[0][0]}")
print(f"Fales Positives (Real videos wrongly flagged): {conf_matrix[0][1]}")
print(f"Fales Negative (Fakes missed by this rule): {conf_matrix[1][0]} ")
print(f"True Positives (Fakes CAUGHT instantly): {conf_matrix[1][1]}")

print("Examples of Fakes Caught by Metadata")
caught_fakes = df[(df["label"] == 1) & (df["pred_stage1"] == 1)]
print(caught_fakes[["fps", "bitrate", "file_size_mb"]].head())

# %%
required_df = df[["filename", "fps", "file_size_mb", "label", "pred_stage1"]]
hard_cases = required_df[required_df["pred_stage1"] == 0]
print(f"Total Videos: {len(required_df)}")
print(
    f"Caught by Metadata (Skipping): {len(required_df[required_df['pred_stage1'] == 1])}"
)
print(f"Hard Cases to Extract (Processing): {len(hard_cases)}")

base_dir = "../storage/frames"
os.makedirs(f"{base_dir}/real", exist_ok=True)
os.makedirs(f"{base_dir}/fake", exist_ok=True)


def save_frames(row):
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


print("Extracting frames for Hard Cases...")
tqdm.pandas()
hard_cases.progress_apply(save_frames, axis=1)
print(f"\n Done! Check the folder: {base_dir}")

# %%
input_dir = "../storage/frames"
output_dir = "../storage/faces"
os.makedirs(f"{output_dir}/real", exist_ok=True)
os.makedirs(f"{output_dir}/fake", exist_ok=True)
detector_mtcnn = MTCNN()
detector_haar = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


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
        faces_haar = detector_haar.detectMultiScale(gray, 1.1, 4)
        if len(faces_haar) == 0:
            faces_haar = detector_haar.detectMultiScale(gray, 1.05, 1)
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


classes = ["real", "fake"]
stats = {"MTCNN": 0, "Haar": 0, "Skipped": 0}
for label in classes:
    class_path = os.path.join(input_dir, label)
    files = os.listdir(class_path)
    for filename in tqdm(files):
        if filename.endswith(".jpg"):
            in_path = os.path.join(class_path, filename)
            out_path = os.path.join(output_dir, label, filename)
            res = crop_and_save_face_mtcnn(in_path, out_path)
            if res:
                stats[res] += 1
            else:
                stats["Skipped"] += 1
print(f"Final Stats: {stats}")

# %%
