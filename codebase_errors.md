# Codebase Error Report

This document lists all identified errors within the `DeepFake` codebase.

| File | Error Description | Category |
| :--- | :--- | :--- |
| `ml/script/deepfake_model_input.py` | `image` module is shadowed by a loop variable, leading to an AttributeError when calling `img_to_array`. | Logic/Bug |
| `ml/script/deepfake_model_input.py` | Potential typo: `predicitons` instead of `predictions`. | Typo |
| `backend/dragAndDrop.js` | Non-existent HTML tag `vid` used in `createElement` and `querySelectorAll`. | Bug |
| `backend/processAndTransfer.js` | Typo: `fileIput` instead of `fileInput`. | Typo |
| `frontend/index.html` | Typo: `<spand>` instead of `<span>` for the result display. | Syntax/Typo |
| `ml/script/app.py` | Incorrect CORS middleware import: should be `from fastapi.middleware.cors import CORSMiddleware`. | Bug |
| `ml/script/preTraining.py` | `ValueError: I/O operation on closed file` during `MTCNN()` initialization. Caused by `lz4` incompatibility with Python 3.13 garbage collection. | Environment/Bug |
| `frontend/processAndTransfer.js` | Browser `<video>` element hangs indefinitely when given unsupported container/codecs (like `mpeg4`), preventing video analysis. | Bug/Limitation 🚩|

---

## Detailed Analysis

### 1. `ml/script/deepfake_model_input.py`
...
### 4. `ml/script/preTraining.py` (LZ4 Error)
The error `ValueError: I/O operation on closed file` occurs 3 times during the instantiation of `detector_mtcnn = MTCNN()`. 
- **Cause**: The `mtcnn` library uses `joblib` to load weights from `.lz4` files. In Python 3.13, the `lz4` frame cleanup triggers a `flush()` on a file handle that has already been closed by the parent process.
- **Impact**: Harmless but noisy. It happens after weights are successfully loaded.
- **Fix**: Use "Lazy Initialization" (instantiate `MTCNN` only when first needed inside `crop_face_mtcnn`) to prevent it from triggering on every script import.

In the `extract_features` function:
```python
# Line 10
from tensorflow.keras.preprocessing import image

# ...

# Line 37
for image in imageList: # Shadows the imported module
    # ...
    # Line 41
    x = image.img_to_array(image_resized) # Error: 'numpy.ndarray' object has no attribute 'img_to_array'
```
The variable `image` (the loop component) shadows the `image` module from Keras. This will cause an `AttributeError` at runtime.

### 2. `backend/dragAndDrop.js`
In the `displayVideos` function:
```javascript
// Line 49
const vid = document.createElement("vid"); // No such HTML element
```
HTML does not have a `<vid>` element. This should be `<video>`.

### 3. `frontend/index.html`
In the `result-section`:
```html
<!-- Line 38 -->
<p>Result: <spand id="result"></p> <!-- Incorrect tag name -->
```
The tag `<spand>` is invalid. This should be `<span>`.

### 5. `frontend/processAndTransfer.js` (Unsupported Codec Bug 🚩)
When users upload video files with older codecs that the HTML5 `<video>` element does not natively support (such as `mpeg4` via drag-and-drop or file selection):
- **Cause**: The browser cannot parse the file to extract metadata or seek to timestamps. The script awaits the `video.onloadedmetadata` and `video.onseeked` events which never fire, causing the frontend to freeze indefinitely on "Extracting frames...". 
- **Impact**: Breaks the user experience by locking the application in an eternal loading state.
- **Fix**: Added a timeout and `error` event handlers to the promises in `processAndTransfer.js`. When a native codec failure is detected, the frontend bypasses local canvas extraction and sends the entire raw video file to a new `/predict_video` endpoint on the backend, where Python's OpenCV extracts the frames robustly.
