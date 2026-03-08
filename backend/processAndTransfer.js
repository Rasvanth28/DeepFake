async function processAndSendFrames() {

    const submitBtn = document.getElementById('submit-btn')
    const resultDisplay = document.getElementById('result');

    submitBtn.disabled = true;
    submitBtn.innerText = "Extracting Frames...";
    resultDisplay.innerHTML = "Processing...";


    const fileIput = document.getElementById('file-input');
    const files = fileIput.files;

    if (files.length === 0) {
        alert("Please select at least one video!");
        return;
    }

    // Getting 5 frames at equal distance
    const formData = new FormData();

    for (let v = 0; v < files.length; v++) {
        const currentFile = files[v];
        const video = document.createElement('video');
        video.src = URL.createObjectURL(currentFile);
        // Waiting to get duration and dimensions of the video
        await new Promise(resolve => video.onloadedmetadata = resolve)
        const duration = video.duration;
        const timestamps = [0, duration * 0.25, duration * 0.5, duration * 0.75, duration * 0.98]

        for (let i = 0; i < timestamps.length; i++) {
            video.currentTime = timestamps[i];
            // Waiting for that specific frame to render
            await new Promise(resolve => video.onseeked = resolve);
            // Resizing the image to 224x224
            const canvas = document.createElement('canvas')
            canvas.width = 224;
            canvas.height = 224;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, 224, 224);
            const blob = await new Promise(res => canvas.toBlob(res, 'image/jpeg', 0.9));
            formData.append('frames', blob, `video${v}_${i}.jpg`);
        }
        URL.revokeObjectURL(video.src);
        video.remove();
        console.log(`Finished processing video ${v + 1} of ${files.length}`);
    }

    try {
        submitBtn.innerText = "AI is Analyzing..."
        const response = await fetch('http://127.0.0.1:8000/predict', {
            method: 'POST',
            body: formData
        });
        if (response.ok) {
            const result = await response.json();
            console.log(`The video is: ${result.prediction}`);
            console.log(`Confidence: ${(result.confidence * 100).toFixed(2)}%`)
            resultDisplay.innerText = result.prediction;
        }
        else {
            console.error("Server returned an error:", response.statusText);
        }
    } catch (error) {
        console.error("Connection failed:", error);
    }
    finally {
        submitBtn.disabled = false;
        submitBtn.innerText = "Analyzed video"
    }
}