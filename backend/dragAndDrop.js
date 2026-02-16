// Declaring the drop target

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
        if (fileItems.some((item) => item.type.startsWith("video/"))) {
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

// Processing the drop

const preview = document.getElementById("preview");
const fileRegistry = new Set();

function displayVideos(files) {
    for (const file of files) {
        const fingerprint = `${file.name}-${file.size}-${file.lastModified}`;
        if (fileRegistry.has(fingerprint)) {
            console.warn(`Duplicate ignored: ${file.name}`);
            continue;
        }
        if (file.type.startsWith("video/")) {
            fileRegistry.add(fingerprint);
            const li = document.createElement("li");
            const vid = document.createElement("vid");
            vid.src = URL.createObjectURL(file);
            vid.controls = true;
            vid.width = 300;
            li.appendChild(vid);
            li.appendChild(document.createTextNode(file.name));
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

// Adding the same behavior to the input

const fileInput = document.getElementById("file-input");
fileInput.addEventListener("change", (e) => {
    displayVideos(e.target.files);
})



// Clear Button

const clearBtn = document.getElementById("clear-btn");
clearBtn.addEventListener("click", () => {
    for (const vid of preview.querySelectorAll("vid")) {
        URL.revokeObjectURL(vid.src);
    }
    preview.textContent = "";
    fileInput.value = "";
    fileRegistry.clear();
});