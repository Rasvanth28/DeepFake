# DeepFake Analyzer: A Hybrid Heuristic and Deep Learning Approach for Synthetic Media Detection

![TensorFlow](https://img.shields.io/badge/AI-TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/Computer_Vision-OpenCV-5C3EE8?logo=opencv&logoColor=white)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract
This repository implements a robust pipeline for the detection of synthetically manipulated video content (deepfakes). By leveraging a dual-stage heuristic and deep learning architecture, the model solves the challenge of authenticating digital media in real-time, effectively mitigating the spread of advanced digital misinformation.

## Architecture & Pipeline
The detection framework employs a modular, multi-stage pipeline designed for computational efficiency and high precision:

1. **Metadata Screening (Heuristic Filtering):** Acts as an initial low-compute sieve, isolating suspect videos based on anomalous framerates (e.g., deviations from standard 24/30 FPS) often introduced during synthetic generation.
2. **Temporal Frame Sampling:** Uniformly extracts 5 frames across the video timeline, ensuring temporal representation without the computational overhead of processing every frame.
3. **Face Detection & Alignment:** Utilizes a Multi-Task Cascaded Convolutional Neural Network (MTCNN) to precisely localize and crop facial regions, employing a Haar Cascade fallback mechanism to maximize detection recall. Extracted faces are resized to 224x224 pixels.
4. **Deep Feature Extraction:** Preprocessed frames are passed through an `InceptionResNetV2` backbone, initialized with ImageNet weights. The network acts as a robust feature extractor, producing flattened, high-dimensional latent representations of the facial regions.
5. **Classification:** Extracted embeddings are normalized via standard scaling (`StandardScaler`) and classified using a Support Vector Machine (SVM) equipped with a Radial Basis Function (RBF) kernel. The SVM provides binary categorization ("real" or "fake") utilizing a probability distribution model optimized for balanced class weights.

## Technology Stack
- **TensorFlow / Keras:** For deep feature extraction via the `InceptionResNetV2` architecture.
- **scikit-learn:** For predictive modeling (SVM) and standardization pipelines.
- **OpenCV & MTCNN:** For advanced media processing, temporal sampling, and facial feature localization.
- **FastAPI / Uvicorn:** For asynchronous, high-throughput RESTful API inference serving.
- **Pandas & NumPy:** For robust matrix operations and dataset manipulation.
- **Joblib:** For efficient serialization of statistical models and scalers.

## Getting Started

Follow these steps to replicate the environment and run the inference server.

### Prerequisites
- Python 3.9+
- Virtual Environment (recommended)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Rasvanth28/DeepFake.git
   cd DeepFake
   ```

2. Initialize and activate a virtual environment:
   ```bash
   python -m venv ml/.venv
   source ml/.venv/bin/activate  # On Windows use: ml\.venv\Scripts\activate
   ```

3. Install the core dependencies:
   ```bash
   pip install -r ml/requirements.txt
   ```

### Running Inference
To start the model inference server:
```bash
cd ml/script
uvicorn app:app --reload
```
The API will be accessible at `http://127.0.0.1:8000`. You can test the endpoints via the built-in Swagger documentation at `http://127.0.0.1:8000/docs`.
