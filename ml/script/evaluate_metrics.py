import os
import joblib
import numpy as np
import time
from pathlib import Path
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics import accuracy_score, f1_score

# Paths
base_dir = "/Users/rasvanth/Documents/Projects/DeepFake/ml"
model_path = os.path.join(base_dir, "model", "deepfake_svm_model.pkl")
scaler_path = os.path.join(base_dir, "model", "feature_scaler.pkl")
faces_dir = os.path.join(base_dir, "storage", "faces")

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    print("Error: Model or Scaler not found at expected paths.")
    exit(1)

print("Loading SVM and Scaler...")
clf = joblib.load(model_path)
scaler = joblib.load(scaler_path)

print("Loading InceptionResNetV2...")
# Keep it global to avoid re-initializing
inception_model = InceptionResNetV2(include_top=False, weights="imagenet", pooling="avg")

def extract_features(directory, sample_count=100):
    features = []
    labels = []
    classes = {"real": 0, "fake": 1}

    for label_name, label_val in classes.items():
        class_dir = os.path.join(directory, label_name)
        if not os.path.exists(class_dir):
            print(f"Directory not found: {class_dir}")
            continue
            
        files = [f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        
        # Take a subset for reasonably fast evaluation
        if len(files) > sample_count:
            files = files[:sample_count]
            
        print(f"Extracting features for {label_name}: {len(files)} items")
        
        for filename in files:
            img_path = os.path.join(class_dir, filename)
            try:
                img = image.load_img(img_path, target_size=(224, 224))
                img_data = image.img_to_array(img)
                img_data = np.expand_dims(img_data, axis=0)
                img_data = preprocess_input(img_data)

                feature = inception_model.predict(img_data, verbose=0)
                features.append(feature.flatten())
                labels.append(label_val)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return np.array(features), np.array(labels)

print("Evaluating metrics from storage/faces...")
# We use a smaller sample size to ensure it finishes within the tool call timeout
X, y = extract_features(faces_dir, sample_count=50) 

if len(X) > 0:
    X_scaled = scaler.transform(X)
    preds = clf.predict(X_scaled)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average='macro')
    
    print("\n[METRICS_START]")
    print(f"ACCURACY:{acc:.4f}")
    print(f"F1:{f1:.4f}")
    print("[METRICS_END]")
else:
    print("No faces found in storage/faces to evaluate.")
    exit(1)
