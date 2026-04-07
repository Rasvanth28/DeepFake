# DeepFake Analyzer 🛡️

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/AI-TensorFlow-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)

An AI-driven media authentication platform designed to detect synthetic and manipulated video content with high precision.

---

## 🔗 Project Links
- **Live Frontend:** [https://rasvanth28.github.io/DeepFake-Frontend/index.html](https://rasvanth28.github.io/DeepFake-Frontend/index.html)
- **GitHub Repository:** [https://github.com/Rasvanth28/DeepFake](https://github.com/Rasvanth28/DeepFake)
- **ML Backend Hosting:** [Hugging Face Space](https://bettercallkc-deepfake.hf.space)

---

## ✨ Key Features
- **Instant Video Analysis:** Drag-and-drop or upload videos to detect deepfakes in seconds.
- **Temporal Frame Sampling:** Automatically extracts 5 key frames across the video duration for comprehensive analysis.
- **Real-Time Progress Feedback:** Visual indicators for frame extraction and AI analysis stages.
- **Glassmorphic UI/UX:** A modern, premium dark-themed interface with interactive canvas animations.
- **Metrics Dashboard:** Tracks analysis history, authenticity ratios, and processing performance.
- **Memory Optimized:** Client-side resizing (224x224) reduces network payload and improves latency.

---

## 🛠️ Technology Stack

### Frontend
- **HTML5 & CSS3:** Custom Glassmorphic design with backdrop-filters.
- **JavaScript (ES6+):** Asynchronous file processing and API integration.
- **Canvas API:** High-performance neural link background and frame preprocessing.

### Backend (ML)
- **FastAPI:** High-performance asynchronous Python framework.
- **TensorFlow:** Deep learning model for classification.
- **OpenCV:** Advanced media processing and frame extraction.
- **MTCNN:** Face detection and alignment.

### DevOps
- **Docker:** Containerized backend for consistent deployment.
- **GitHub Pages:** Static frontend hosting.
- **Hugging Face Spaces:** Cloud hosting for the machine learning backend.

---

## 🚀 Getting Started

### Local Frontend Development
1. Clone the repository:
   ```bash
   git clone https://github.com/Rasvanth28/DeepFake.git
   ```
2. Navigate to the `frontend` directory:
   ```bash
   cd DeepFake/frontend
   ```
3. Open `index.html` in your browser (or use a local server like Live Server).

### Local Backend Development
1. Navigate to the `ml` directory:
   ```bash
   cd DeepFake/ml
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the FastAPI server:
   ```bash
   uvicorn script.app:app --reload
   ```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 💡 Acknowledgements
- **Research:** Built using principles of deep learning for digital media forensics.
- **Inspiration:** Developed as a solution for modern digital misinformation challenges.
