
const dropZone = document.getElementById("drop-zone");
dropZone.addEventListener("drop", dropHandler);

window.addEventListener("drop", (e) => {
    if ([...e.dataTransfer.items].some((item) => item.kind === "file")) {
        e.preventDefault();
    }
});

dropZone.addEventListener("dragover", (e) => {
    const fileItems = [...e.dataTransfer.items].filter((item) => item.kind === "file");
    if (fileItems.length > 0) {
        e.preventDefault();
        dropZone.classList.add('drag-over');
        if (fileItems.some((item) => item.type.startsWith("video/") || item.type === "")) {
            e.dataTransfer.dropEffect = "copy";
        } else {
            e.dataTransfer.dropEffect = "none";
        }
    }
});

window.addEventListener("dragover", (e) => {
    const fileItems = [...e.dataTransfer.items].filter((item) => item.kind === "file");
    if (fileItems.length > 0) {
        e.preventDefault();
        if (!dropZone.contains(e.target)) {
            e.dataTransfer.dropEffect = "none";
        }
    }
});


const preview = document.getElementById("preview");
const fileRegistry = new Set();
window.uploadedFiles = [];
window.summaryStats = { total: 0, fake: 0, real: 0, noFace: 0, totalTimeMs: 0 };

function displayVideos(files) {
    for (const file of files) {
        const fingerprint = `${file.name}-${file.size}-${file.lastModified}`;
        if (fileRegistry.has(fingerprint)) {
            console.warn(`Duplicate ignored: ${file.name}`);
            continue;
        }
        if (file.type.startsWith("video/") || file.name.match(/\.(mp4|mov|avi|wmv|flv|mkv|webm)$/i)) {
            fileRegistry.add(fingerprint);
            window.uploadedFiles.push(file);
            const li = document.createElement("li");
            li.id = `video-item-${fingerprint.replace(/[^a-zA-Z0-9]/g, '')}`;
            const wrapper = document.createElement("div");
            wrapper.className = "video-wrapper";

            const vid = document.createElement("video");
            vid.src = URL.createObjectURL(file);
            vid.controls = true;
            wrapper.appendChild(vid);
            
            const statusDiv = document.createElement("div");
            statusDiv.className = "video-status overlay-card";
            wrapper.appendChild(statusDiv);

            li.appendChild(wrapper);

            const nameSpan = document.createElement("span");
            nameSpan.className = "video-filename";
            nameSpan.textContent = file.name;
            li.appendChild(nameSpan);

            preview.appendChild(li);
        }
    }
}

function dropHandler(ev) {
    ev.preventDefault();
    const files = [...ev.dataTransfer.items]
        .map((item) => item.getAsFile()).filter((file) => file);
    displayVideos(files);
}

dropZone.addEventListener("dragleave", () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener("drop", () => dropZone.classList.remove('drag-over'));

// Adding the same behavior to the input

const fileInput = document.getElementById("file-input");
fileInput.addEventListener("change", (e) => {
    displayVideos(e.target.files);
})

// Clear Button

const clearBtn = document.getElementById("clear-btn");
clearBtn.addEventListener("click", () => {
    for (const vid of preview.querySelectorAll("video")) {
        URL.revokeObjectURL(vid.src);
    }
    preview.textContent = "";
    fileInput.value = "";
    fileRegistry.clear();
    window.uploadedFiles = [];
    window.summaryStats = { total: 0, fake: 0, real: 0, noFace: 0, totalTimeMs: 0 };
    const summaryBoard = document.getElementById("summary-board");
    if (summaryBoard) {
        summaryBoard.innerHTML = "";
        summaryBoard.classList.add("hidden");
    }
});