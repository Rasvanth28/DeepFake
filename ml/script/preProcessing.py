# %%
import os
import pandas as pd
from tqdm import tqdm
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# %%
# Extracting the dataset

extract_path = '../dataset'
print("Files found:",os.listdir(extract_path))

# %%
file_path = f'{extract_path}/train_labels.csv'
df = pd.read_csv(file_path)
print(df.columns.tolist())
df.info()
df.head()

# %%
real_path = '../dataset/train/real'
fake_path = '../dataset/train/fake'

data = []

def extract_video_features(directory,label):
    files = os.listdir(directory)
    print(f"Processing {len(files)} videos from {directory}")
    for filename in tqdm(files):
        if filename.endswith(".mp4"):
            filepath = os.path.join(directory,filename)
            cap = cv2.VideoCapture(filepath)
            if cap.isOpened():
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

                duration = frame_count/fps if fps > 0 else 0
                file_size = os.path.getsize(filepath)/(1024*1024)
                bitrate = (file_size * 8)/duration if duration > 0 else 0

                data.append({
                    'width':width,
                    'height':height,
                    'fps':fps,
                    'frame_count':frame_count,
                    'duration':duration,
                    'file_size_mb':file_size,
                    'bitrate':bitrate,
                    'label':label
                })
            cap.release()

if os.path.exists(real_path) and os.path.exists(fake_path):
    extract_video_features(real_path,0)
    extract_video_features(fake_path,1)

    df = pd.DataFrame(data)
    print("\nFeature Extraction Complete!")
    print(df.head())
else:
    print("Error:Could not find 'train/real' or 'train/fake' folders.")

            

# %%
X = df.drop('label',axis=1)
Y = df['label']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

rf = RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(X_train,Y_train)

importances = rf.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
plt.title("Which Metadata Parameter is Most Vital for Detection?")
plt.bar(range(X.shape[1]),importances[indices],align="center",color='teal')
plt.xticks(range(X.shape[1]),[feature_names[i] for i in indices],rotation=45)
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()

print("\nRanking Results")
for i in indices:
    print(f"{feature_names[i]}:{importances[i]:.4f}")


# %%
sns.set_style("whitegrid")
plt.Figure(figsize=(14,5))

plt.subplot(1,2,1)
sns.kdeplot(data=df[df['label']==0]['fps'],label='Real',fill=True,color='blue',alpha=0.3)
sns.kdeplot(data=df[df['label']==1]['fps'],label='Fake',fill=True,color='red',alpha=0.3)
plt.title("Distribution of FPS (Real vs Fake)")
plt.xlabel("Frames Per Second")
plt.legend()

plt.subplot(1,2,2)
sns.kdeplot(data=df[df['label']==0]['bitrate'],label='Real',fill=True,color="blue",alpha=0.3)
sns.kdeplot(data=df[df['label']==1]['bitrate'],label='Fake',fill=True,color='red',alpha=0.3)
plt.title("Distribution of Bitrate (Real vs Fake)")
plt.xlabel("Bitrate (MB/s)")
plt.legend()

plt.tight_layout()
plt.show()

print("Average Stats\n")
print(df.groupby('label')[['fps','bitrate']].mean())

# %%
print("Standard Deviation\n")
print(df.groupby('label')[['fps','bitrate']].std())
print("FPS Counts (Real Videos)\n")
print(df[df['label'] == 0]['fps'].value_counts().head(5))
print("FPS Counts (Fake Videos)\n")
print(df[df['label']==1]['fps'].value_counts().head(5))

# %%



