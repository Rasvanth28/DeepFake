
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
            const vid = document.createElement("video");
            vid.src = URL.createObjectURL(file);
            vid.controls = true;
            vid.width = 300;
            li.appendChild(vid);
            li.appendChild(document.createTextNode(file.name));
            
            const statusDiv = document.createElement("div");
            statusDiv.className = "video-status badge-container";
            statusDiv.style.marginLeft = "auto";
            li.appendChild(statusDiv);

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
});