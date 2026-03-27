async function processAndSendFrames() {
    const submitBtn = document.getElementById('submit-btn');
    const btnText = submitBtn.querySelector('.btn-text');
    const loader = submitBtn.querySelector('.loader');

    const files = window.uploadedFiles && window.uploadedFiles.length > 0
        ? window.uploadedFiles
        : document.getElementById('file-input').files;

    if (files.length === 0) {
        alert("Please select at least one video!");
        return;
    }

    submitBtn.disabled = true;
    if (loader) loader.classList.remove('hidden');

    for (let v = 0; v < files.length; v++) {
        const currentFile = files[v];
        const fingerprint = `${currentFile.name}-${currentFile.size}-${currentFile.lastModified}`;
        const liId = `video-item-${fingerprint.replace(/[^a-zA-Z0-9]/g, '')}`;
        const li = document.getElementById(liId);
        const statusDiv = li ? li.querySelector('.video-status') : null;

        if (statusDiv && statusDiv.dataset.analyzed === "true") {
            continue;
        }

        if (statusDiv) {
            statusDiv.innerHTML = `<span class="badge status">Extracting frames...</span>`;
        }

        const video = document.createElement('video');
        video.src = URL.createObjectURL(currentFile);
        
        const isLoaded = await new Promise(resolve => {
            video.onloadedmetadata = () => resolve(true);
            video.onerror = () => resolve(false);
            setTimeout(() => resolve(false), 5000); // 5 sec timeout
        });

        if (!isLoaded) {
            if (statusDiv) {
                statusDiv.innerHTML = `<span class="badge status">Sending to Backend...</span>`;
            }
            if (btnText) btnText.innerText = `Analyzing Video ${v + 1}...`;
            URL.revokeObjectURL(video.src);
            video.remove();
            
            const reqData = new FormData();
            reqData.append('video', currentFile);
            
            try {
                const response = await fetch('http://127.0.0.1:8000/predict_video', {
                    method: 'POST',
                    body: reqData
                });
                if (response.ok) {
                    const result = await response.json();
                    if (statusDiv) {
                        statusDiv.dataset.analyzed = "true";
                        if (result.prediction === "No faces detected" || result.prediction.includes("Error")) {
                            statusDiv.innerHTML = `
                                <span class="badge status fake">${result.prediction}</span>
                                <span class="badge confidence">-</span>
                            `;
                        } else {
                            const isFake = result.prediction.toLowerCase().includes('fake');
                            const displayConfidence = isFake ? result.confidence : 1 - result.confidence;
                            statusDiv.innerHTML = `
                                <span class="badge status ${isFake ? 'fake' : 'real'}">${result.prediction}</span>
                                <span class="badge confidence">Confidence: ${(displayConfidence * 100).toFixed(2)}%</span>
                            `;
                        }
                    }
                } else {
                    if (statusDiv) statusDiv.innerHTML = `<span class="badge status fake">Error: ${response.statusText}</span>`;
                }
            } catch (error) {
                if (statusDiv) statusDiv.innerHTML = `<span class="badge status fake">Connection Failed</span>`;
            }
            continue;
        }

        const duration = video.duration;
        const timestamps = [0, duration * 0.25, duration * 0.5, duration * 0.75, duration * 0.98]

        const formData = new FormData();

        for (let i = 0; i < timestamps.length; i++) {
            video.currentTime = timestamps[i];
            const seekSuccess = await new Promise(resolve => {
                video.onseeked = () => resolve(true);
                video.onerror = () => resolve(false);
                setTimeout(() => resolve(false), 5000);
            });
            
            if (!seekSuccess) continue;

            const canvas = document.createElement('canvas')
            canvas.width = 224;
            canvas.height = 224;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, 224, 224);
            const blob = await new Promise(res => canvas.toBlob(res, 'image/jpeg', 0.9));
            formData.append('frames', blob, `video${v}_${i}.jpg`);

            if (statusDiv) {
                const percent = Math.round(((i + 1) / timestamps.length) * 100);
                statusDiv.innerHTML = `<span class="badge status">Extracting: ${percent}%</span>`;
            }
        }
        URL.revokeObjectURL(video.src);
        video.remove();

        if (statusDiv) {
            statusDiv.innerHTML = `<span class="badge status">AI is Analyzing...</span>`;
        }
        if (btnText) btnText.innerText = `Analyzing Video ${v + 1}...`;

        try {
            const response = await fetch('http://127.0.0.1:8000/predict', {
                method: 'POST',
                body: formData
            });
            if (response.ok) {
                const result = await response.json();
                if (statusDiv) {
                    statusDiv.dataset.analyzed = "true";
                    if (result.prediction === "No faces detected" || result.prediction.includes("Error")) {
                        statusDiv.innerHTML = `
                            <span class="badge status fake">${result.prediction}</span>
                            <span class="badge confidence">-</span>
                        `;
                    } else {
                        const isFake = result.prediction.toLowerCase().includes('fake');
                        const displayConfidence = isFake ? result.confidence : 1 - result.confidence;
                        statusDiv.innerHTML = `
                            <span class="badge status ${isFake ? 'fake' : 'real'}">${result.prediction}</span>
                            <span class="badge confidence">Confidence: ${(displayConfidence * 100).toFixed(2)}%</span>
                        `;
                    }
                }
            } else {
                if (statusDiv) statusDiv.innerHTML = `<span class="badge status fake">Error: ${response.statusText}</span>`;
            }
        } catch (error) {
            if (statusDiv) statusDiv.innerHTML = `<span class="badge status fake">Connection Failed</span>`;
        }
    }

    submitBtn.disabled = false;
    if (btnText) btnText.innerText = "Analyze Media";
    if (loader) loader.classList.add('hidden');
}