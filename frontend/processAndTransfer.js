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

    // Hide old summary if any, starting new batch
    const summaryBoard = document.getElementById("summary-board");
    if (summaryBoard) summaryBoard.classList.add('hidden');

    let batchStartTime = performance.now();
    let anyProcessed = false;

    for (let v = 0; v < files.length; v++) {
        const currentFile = files[v];
        const fingerprint = `${currentFile.name}-${currentFile.size}-${currentFile.lastModified}`;
        const liId = `video-item-${fingerprint.replace(/[^a-zA-Z0-9]/g, '')}`;
        const li = document.getElementById(liId);
        const statusDiv = li ? li.querySelector('.video-status') : null;

        if (statusDiv && statusDiv.dataset.analyzed === "true") {
            continue;
        }

        anyProcessed = true;
        const videoStartTime = performance.now();

        if (statusDiv) {
            statusDiv.innerHTML = `
                <div class="loader"></div>
                <div style="margin-top: 1rem; font-weight: 600;">Extracting frames...</div>
            `;
            statusDiv.classList.add("active");
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
                const response = await fetch('http://127.0.0.1:8082/predict_video', {
                    method: 'POST',
                    body: reqData
                });
                if (response.ok) {
                    const result = await response.json();
                    if (statusDiv) {
                        statusDiv.dataset.analyzed = "true";
                        const processTimeSec = (performance.now() - videoStartTime) / 1000;
                        const processTimeStr = processTimeSec < 60 ? `${processTimeSec.toFixed(2)}s` : `${Math.floor(processTimeSec / 60)}m ${Math.floor(processTimeSec % 60)}s`;
                        window.summaryStats.total++;

                        if (result.prediction === "No faces detected" || result.prediction.includes("Error")) {
                            if (result.prediction === "No faces detected") window.summaryStats.noFace++;
                            statusDiv.innerHTML = `
                                <div class="overlay-title fake">${result.prediction}</div>
                                <div class="badge confidence" style="margin-top:0.5rem;">-</div>
                                <div style="margin-top:0.5rem; font-size:0.9rem; opacity:0.8; font-weight: 600;">Time: ${processTimeStr}</div>
                            `;
                        } else {
                            const isFake = result.prediction.toLowerCase().includes('fake');
                            if (isFake) window.summaryStats.fake++; else window.summaryStats.real++;

                            const displayConfidence = isFake ? result.confidence : 1 - result.confidence;
                            statusDiv.innerHTML = `
                                <div class="overlay-title ${isFake ? 'fake' : 'real'}">${result.prediction}</div>
                                <div class="badge confidence" style="margin-top:0.5rem;">Conf: ${(displayConfidence * 100).toFixed(0)}%</div>
                                <div style="margin-top:0.5rem; font-size:0.9rem; opacity:0.8; font-weight: 600;">Time: ${processTimeStr}</div>
                            `;
                        }
                        statusDiv.classList.add("active");
                    }
                } else {
                    if (statusDiv) {
                        statusDiv.innerHTML = `<div class="overlay-title fake">Error</div>`;
                        statusDiv.classList.add("active");
                    }
                }
            } catch (error) {
                if (statusDiv) {
                    statusDiv.innerHTML = `<div class="overlay-title fake">Connection Failed</div>`;
                    statusDiv.classList.add("active");
                }
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
            const response = await fetch('http://127.0.0.1:8082/predict', {
                method: 'POST',
                body: formData
            });
            if (response.ok) {
                const result = await response.json();
                if (statusDiv) {
                    statusDiv.dataset.analyzed = "true";
                    const processTimeSec = (performance.now() - videoStartTime) / 1000;
                    const processTimeStr = processTimeSec < 60 ? `${processTimeSec.toFixed(2)}s` : `${Math.floor(processTimeSec / 60)}m ${Math.floor(processTimeSec % 60)}s`;
                    window.summaryStats.total++;

                    if (result.prediction === "No faces detected" || result.prediction.includes("Error")) {
                        if (result.prediction === "No faces detected") window.summaryStats.noFace++;
                        statusDiv.innerHTML = `
                            <div class="overlay-title fake">${result.prediction}</div>
                            <div class="badge confidence" style="margin-top:0.5rem;">-</div>
                            <div style="margin-top:0.5rem; font-size:0.9rem; opacity:0.8; font-weight: 600;">Time: ${processTimeStr}</div>
                        `;
                    } else {
                        const isFake = result.prediction.toLowerCase().includes('fake');
                        if (isFake) window.summaryStats.fake++; else window.summaryStats.real++;

                        const displayConfidence = isFake ? result.confidence : 1 - result.confidence;
                        statusDiv.innerHTML = `
                            <div class="overlay-title ${isFake ? 'fake' : 'real'}">${result.prediction}</div>
                            <div class="badge confidence" style="margin-top:0.5rem;">Conf: ${(displayConfidence * 100).toFixed(0)}%</div>
                            <div style="margin-top:0.5rem; font-size:0.9rem; opacity:0.8; font-weight: 600;">Time: ${processTimeStr}</div>
                        `;
                    }
                    statusDiv.classList.add("active");
                }
            } else {
                if (statusDiv) {
                    statusDiv.innerHTML = `<div class="overlay-title fake">Error</div>`;
                    statusDiv.classList.add("active");
                }
            }
        } catch (error) {
            if (statusDiv) {
                statusDiv.innerHTML = `<div class="overlay-title fake">Connection Failed</div>`;
                statusDiv.classList.add("active");
            }
        }
    }

    if (anyProcessed) {
        let batchEndTime = performance.now();
        window.summaryStats.totalTimeMs += (batchEndTime - batchStartTime);

        if (summaryBoard) {
            const stats = window.summaryStats;

            const formatTime = (ms) => {
                const totalS = ms / 1000;
                if (totalS < 60) return `${totalS.toFixed(2)}s`;
                return `${Math.floor(totalS / 60)}m ${Math.floor(totalS % 60)}s`;
            };

            const avgMs = stats.total > 0 ? (stats.totalTimeMs / stats.total) : 0;
            const totalTimeStr = formatTime(stats.totalTimeMs);
            const avgTimeStr = formatTime(avgMs);

            summaryBoard.innerHTML = `
                <h3 style="color: var(--text-primary); margin-bottom: 1.5rem; font-size: 1.5rem;">Batch Analysis Complete</h3>
                <div class="badge-container" style="gap: 1.5rem;">
                    <span class="badge" style="color: white; border-color: rgba(255,255,255,0.2);">Videos: ${stats.total}</span>
                    <span class="badge status fake" style="font-size: 1.1rem;">Fake: ${stats.fake}</span>
                    <span class="badge status real" style="font-size: 1.1rem;">Authentic: ${stats.real}</span>
                    <span class="badge" style="color: var(--text-secondary); border-color: rgba(255,255,255,0.1); font-size: 1.1rem;">No Face: ${stats.noFace}</span>
                </div>
                <div class="badge-container" style="gap: 1.5rem; margin-top: 1.5rem;">
                    <span class="badge confidence" style="font-size: 1.1rem;">Total Time: ${totalTimeStr}</span>
                    <span class="badge confidence" style="font-size: 1.1rem;">Average Time: ${avgTimeStr} / video</span>
                </div>
            `;
            summaryBoard.classList.remove("hidden");
        }
    }

    submitBtn.disabled = false;
    if (btnText) btnText.innerText = "Analyze Media";
    if (loader) loader.classList.add('hidden');
}