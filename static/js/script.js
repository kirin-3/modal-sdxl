/**
 * SDXL Studio - Lean Reactive Frontend Client
 */

// State Management
const AppState = {
    isGenerating: false,
    timerInterval: null,
    startTime: 0,
    currentImages: [],
    history: [],
    presets: {},
    activePresetKey: "",
    loras: [
        { model_id: "civitai:1681903", weight: 2.0 },
        { model_id: "civitai:1764869", weight: 0.75 }
    ]
};

// UI Elements Cache
const UI = {};

function initElements() {
    UI.prompt = document.getElementById("prompt");
    UI.negativePrompt = document.getElementById("negative_prompt");
    UI.generateBtn = document.getElementById("generate-btn");
    UI.generateBtnText = document.getElementById("generate-btn-text");
    UI.progressContainer = document.getElementById("progress-container");
    UI.progressStatus = document.getElementById("progress-status");
    UI.progressTimer = document.getElementById("progress-timer");
    UI.galleryGrid = document.getElementById("gallery-grid");
    UI.resultSummary = document.getElementById("result-summary");
    UI.presetSelector = document.getElementById("preset-selector");
    UI.defaultNegBtn = document.getElementById("default-neg-btn");

    UI.tabHf = document.getElementById("tab-hf");
    UI.tabCivitai = document.getElementById("tab-civitai");
    UI.groupHf = document.getElementById("group-hf");
    UI.groupCivitai = document.getElementById("group-civitai");
    UI.modelId = document.getElementById("model_id");
    UI.civitaiId = document.getElementById("civitai_id");

    UI.width = document.getElementById("width");
    UI.height = document.getElementById("height");
    UI.steps = document.getElementById("steps");
    UI.stepsVal = document.getElementById("steps-val");
    UI.guidanceScale = document.getElementById("guidance_scale");
    UI.guidanceVal = document.getElementById("guidance-val");
    UI.scheduler = document.getElementById("scheduler");
    UI.seed = document.getElementById("seed");
    UI.randomizeSeedBtn = document.getElementById("randomize-seed-btn");
    UI.batchSize = document.getElementById("batch_size");
    UI.batchCount = document.getElementById("batch_count");
    UI.clipSkip = document.getElementById("clip_skip");

    UI.freeuEnabled = document.getElementById("freeu_enabled");
    UI.freeuParams = document.getElementById("freeu-params");
    UI.freeuB1 = document.getElementById("freeu_b1");
    UI.freeuB2 = document.getElementById("freeu_b2");
    UI.freeuS1 = document.getElementById("freeu_s1");
    UI.freeuS2 = document.getElementById("freeu_s2");

    UI.loraContainer = document.getElementById("lora-container");
    UI.addLoraBtn = document.getElementById("add-lora-btn");

    UI.historyToggleBtn = document.getElementById("history-toggle-btn");
    UI.historyDrawer = document.getElementById("history-drawer");
    UI.historyOverlay = document.getElementById("history-overlay");
    UI.closeHistoryBtn = document.getElementById("close-history-btn");
    UI.historyList = document.getElementById("history-list");

    UI.lightboxModal = document.getElementById("lightbox-modal");
    UI.lightboxBackdrop = document.getElementById("lightbox-backdrop");
    UI.lightboxClose = document.getElementById("lightbox-close");
    UI.lightboxImg = document.getElementById("lightbox-img");
    UI.lightboxMeta = document.getElementById("lightbox-meta");
    UI.lightboxCopyBtn = document.getElementById("lightbox-copy-btn");
    UI.lightboxReuseBtn = document.getElementById("lightbox-reuse-btn");
    UI.lightboxDownloadLink = document.getElementById("lightbox-download-link");

    UI.dropzone = document.getElementById("prompt-dropzone");
    UI.toast = document.getElementById("toast");
}

// ==============================================================================
// Toast Notifications
// ==============================================================================

function showToast(message, duration = 3000) {
    if (!UI.toast) return;
    UI.toast.textContent = message;
    UI.toast.classList.add("active");
    setTimeout(() => {
        UI.toast.classList.remove("active");
    }, duration);
}

// ==============================================================================
// Model Tabs & Dimension Presets
// ==============================================================================

function setupModelTabs() {
    UI.tabHf.addEventListener("click", () => {
        UI.tabHf.classList.add("active");
        UI.tabCivitai.classList.remove("active");
        UI.groupHf.style.display = "block";
        UI.groupCivitai.style.display = "none";
        saveFormState();
    });

    UI.tabCivitai.addEventListener("click", () => {
        UI.tabCivitai.classList.add("active");
        UI.tabHf.classList.remove("active");
        UI.groupCivitai.style.display = "block";
        UI.groupHf.style.display = "none";
        saveFormState();
    });
}

function setupDimensionPresets() {
    document.querySelectorAll(".dimensions-presets .pill-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".dimensions-presets .pill-btn").forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            UI.width.value = btn.dataset.w;
            UI.height.value = btn.dataset.h;
            saveFormState();
        });
    });
}

function setupSliders() {
    UI.steps.addEventListener("input", () => {
        UI.stepsVal.textContent = UI.steps.value;
        saveFormState();
    });
    UI.guidanceScale.addEventListener("input", () => {
        UI.guidanceVal.textContent = parseFloat(UI.guidanceScale.value).toFixed(1);
        saveFormState();
    });

    UI.freeuEnabled.addEventListener("change", () => {
        UI.freeuParams.style.display = UI.freeuEnabled.checked ? "block" : "none";
        saveFormState();
    });

    UI.randomizeSeedBtn.addEventListener("click", () => {
        UI.seed.value = "";
        showToast("Seed set to random");
        saveFormState();
    });

    UI.defaultNegBtn.addEventListener("click", () => {
        UI.negativePrompt.value = "cartoon, animation, drawing, low quality, blurry, deformed, bad anatomy, disfigured, watermark, signature";
        saveFormState();
        showToast("Inserted default negative prompt");
    });
}

// ==============================================================================
// LoRA Manager
// ==============================================================================

function renderLoras() {
    if (!UI.loraContainer) return;
    UI.loraContainer.innerHTML = "";

    AppState.loras.forEach((lora, idx) => {
        const row = document.createElement("div");
        row.className = "lora-row";

        const isCivitai = lora.model_id.startsWith("civitai:");
        const cleanId = lora.model_id.replace("civitai:", "").replace("hf:", "");

        row.innerHTML = `
            <div class="lora-header">
                <span>LoRA #${idx + 1}</span>
                <button type="button" class="remove-lora-btn" data-idx="${idx}" title="Remove LoRA">&times;</button>
            </div>
            <div class="tab-group" style="margin-bottom: 4px;">
                <button type="button" class="tab-btn ${isCivitai ? 'active' : ''}" data-source="civitai" data-idx="${idx}">CivitAI</button>
                <button type="button" class="tab-btn ${!isCivitai ? 'active' : ''}" data-source="hf" data-idx="${idx}">HF</button>
            </div>
            <div class="row-inputs">
                <div class="col" style="flex: 2;">
                    <input type="text" class="text-input lora-id-input" data-idx="${idx}" value="${cleanId}" placeholder="${isCivitai ? 'Model ID (e.g. 1681903)' : 'repo/path'}">
                </div>
                <div class="col" style="flex: 1;">
                    <input type="number" class="text-input lora-weight-input" data-idx="${idx}" value="${lora.weight}" step="0.05" min="-2" max="3" placeholder="Weight">
                </div>
            </div>
        `;
        UI.loraContainer.appendChild(row);
    });

    // Bind events
    UI.loraContainer.querySelectorAll(".remove-lora-btn").forEach((btn) => {
        btn.addEventListener("click", (e) => {
            const idx = parseInt(e.target.dataset.idx, 10);
            AppState.loras.splice(idx, 1);
            renderLoras();
            saveFormState();
        });
    });

    UI.loraContainer.querySelectorAll(".tab-btn").forEach((btn) => {
        btn.addEventListener("click", (e) => {
            const idx = parseInt(e.target.dataset.idx, 10);
            const source = e.target.dataset.source;
            const currentVal = AppState.loras[idx].model_id.replace("civitai:", "").replace("hf:", "");
            AppState.loras[idx].model_id = `${source}:${currentVal}`;
            renderLoras();
            saveFormState();
        });
    });

    UI.loraContainer.querySelectorAll(".lora-id-input").forEach((inp) => {
        inp.addEventListener("change", (e) => {
            const idx = parseInt(e.target.dataset.idx, 10);
            const isCivitai = AppState.loras[idx].model_id.startsWith("civitai:");
            const prefix = isCivitai ? "civitai:" : "hf:";
            AppState.loras[idx].model_id = `${prefix}${e.target.value.trim()}`;
            saveFormState();
        });
    });

    UI.loraContainer.querySelectorAll(".lora-weight-input").forEach((inp) => {
        inp.addEventListener("change", (e) => {
            const idx = parseInt(e.target.dataset.idx, 10);
            AppState.loras[idx].weight = parseFloat(e.target.value) || 0.75;
            saveFormState();
        });
    });
}

function setupLoraControls() {
    UI.addLoraBtn.addEventListener("click", () => {
        if (AppState.loras.length >= 5) {
            showToast("Maximum 5 LoRAs allowed");
            return;
        }
        AppState.loras.push({ model_id: "civitai:", weight: 0.75 });
        renderLoras();
        saveFormState();
    });
}

// ==============================================================================
// Presets Manager
// ==============================================================================

async function loadPresets() {
    try {
        const res = await fetch("/api/presets");
        if (res.ok) {
            AppState.presets = await res.json();
        }
    } catch (e) {
        console.warn("Could not load presets:", e);
    }
}

function setupPresets() {
    UI.presetSelector.addEventListener("change", (e) => {
        const key = e.target.value;
        if (!key || !AppState.presets[key]) return;

        const preset = AppState.presets[key];
        if (preset.prompt_suffix && !UI.prompt.value.includes(preset.prompt_suffix.trim())) {
            UI.prompt.value = (UI.prompt.value.trim() + preset.prompt_suffix).replace(/^,\s*/, "");
        }
        if (preset.negative_prompt) {
            UI.negativePrompt.value = preset.negative_prompt;
        }
        if (preset.steps) {
            UI.steps.value = preset.steps;
            UI.stepsVal.textContent = preset.steps;
        }
        if (preset.guidance_scale) {
            UI.guidanceScale.value = preset.guidance_scale;
            UI.guidanceVal.textContent = preset.guidance_scale.toFixed(1);
        }
        if (preset.scheduler) {
            UI.scheduler.value = preset.scheduler;
        }
        saveFormState();
        showToast(`Applied '${preset.name}' style preset`);
    });
}

// ==============================================================================
// Drag & Drop PNG Parameter Loader
// ==============================================================================

function setupDropzone() {
    const dropzone = UI.dropzone;

    ["dragenter", "dragover"].forEach((eventName) => {
        window.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.add("drag-active");
        });
    });

    ["dragleave", "dragend"].forEach((eventName) => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.remove("drag-active");
        });
    });

    window.addEventListener("drop", async (e) => {
        e.preventDefault();
        dropzone.classList.remove("drag-active");

        if (!e.dataTransfer || !e.dataTransfer.files || e.dataTransfer.files.length === 0) return;
        const file = e.dataTransfer.files[0];
        if (!file.name.endsWith(".png")) {
            showToast("Please drop a PNG image to read metadata.");
            return;
        }

        showToast("Reading image metadata...");
        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch("/api/metadata", { method: "POST", body: formData });
            if (!res.ok) throw new Error("Could not extract metadata");
            const meta = await res.json();
            applyExtractedMetadata(meta);
            showToast("Restored generation parameters from PNG!");
        } catch (err) {
            console.error(err);
            showToast("Failed to parse PNG metadata.");
        }
    });
}

function applyExtractedMetadata(meta) {
    if (meta.prompt) UI.prompt.value = meta.prompt;
    if (meta.negative_prompt) UI.negativePrompt.value = meta.negative_prompt;
    if (meta.steps) {
        UI.steps.value = meta.steps;
        UI.stepsVal.textContent = meta.steps;
    }
    if (meta.guidance_scale) {
        UI.guidanceScale.value = meta.guidance_scale;
        UI.guidanceVal.textContent = parseFloat(meta.guidance_scale).toFixed(1);
    }
    if (meta.seed !== undefined) UI.seed.value = meta.seed;
    if (meta.width) UI.width.value = meta.width;
    if (meta.height) UI.height.value = meta.height;
    if (meta.scheduler) UI.scheduler.value = meta.scheduler;
    if (meta.clip_skip) UI.clipSkip.value = meta.clip_skip;

    if (meta.model_id) {
        if (meta.model_id.startsWith("civitai:")) {
            UI.tabCivitai.click();
            UI.civitaiId.value = meta.model_id.replace("civitai:", "");
        } else {
            UI.tabHf.click();
            UI.modelId.value = meta.model_id;
        }
    }

    if (meta.loras && Array.isArray(meta.loras)) {
        AppState.loras = meta.loras.map((l) => ({
            model_id: l.model_id,
            weight: l.weight || 0.75
        }));
        renderLoras();
    }

    saveFormState();
}

// ==============================================================================
// Generation Workflow
// ==============================================================================

function getGenerationPayload() {
    const isCivitai = UI.tabCivitai.classList.contains("active");
    let modelId = DEFAULT_MODEL_ID;

    if (isCivitai) {
        const cid = UI.civitaiId.value.trim();
        if (!cid) throw new Error("Please enter a CivitAI SDXL Model ID");
        modelId = `civitai:${cid}`;
    } else {
        modelId = UI.modelId.value.trim() || DEFAULT_MODEL_ID;
    }

    const validLoras = AppState.loras.filter((l) => {
        const clean = l.model_id.replace("civitai:", "").replace("hf:", "").trim();
        return clean.length > 0;
    });

    const payload = {
        prompt: UI.prompt.value.trim(),
        negative_prompt: UI.negativePrompt.value.trim(),
        model_id: modelId,
        width: parseInt(UI.width.value, 10) || 1024,
        height: parseInt(UI.height.value, 10) || 1024,
        steps: parseInt(UI.steps.value, 10) || 30,
        guidance_scale: parseFloat(UI.guidanceScale.value) || 7.5,
        batch_size: parseInt(UI.batchSize.value, 10) || 1,
        batch_count: parseInt(UI.batchCount.value, 10) || 1,
        scheduler: UI.scheduler.value,
        loras: validLoras.length > 0 ? validLoras : null
    };

    if (UI.seed.value.trim()) {
        payload.seed = parseInt(UI.seed.value.trim(), 10);
    }
    if (UI.clipSkip.value.trim()) {
        payload.clip_skip = parseInt(UI.clipSkip.value.trim(), 10);
    }

    if (UI.freeuEnabled.checked) {
        payload.freeu = {
            enabled: true,
            b1: parseFloat(UI.freeuB1.value) || 1.3,
            b2: parseFloat(UI.freeuB2.value) || 1.4,
            s1: parseFloat(UI.freeuS1.value) || 0.9,
            s2: parseFloat(UI.freeuS2.value) || 0.2
        };
    }

    return payload;
}

async function triggerGeneration() {
    if (AppState.isGenerating) return;

    if (!UI.prompt.value.trim()) {
        showToast("Please enter a positive prompt.");
        UI.prompt.focus();
        return;
    }

    let payload;
    try {
        payload = getGenerationPayload();
    } catch (err) {
        showToast(err.message);
        return;
    }

    startProgress();

    try {
        const response = await fetch("/api/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(errData.detail || "Generation failed.");
        }

        const data = await response.json();
        handleGenerationSuccess(data);
    } catch (err) {
        console.error("Generation error:", err);
        showToast(`Error: ${err.message}`, 5000);
    } finally {
        stopProgress();
    }
}

function startProgress() {
    AppState.isGenerating = true;
    AppState.startTime = Date.now();
    UI.generateBtn.disabled = true;
    UI.generateBtnText.textContent = "Generating...";
    UI.progressContainer.style.display = "flex";
    UI.progressStatus.textContent = "Dispatched to Modal GPU...";

    AppState.timerInterval = setInterval(() => {
        const elapsed = ((Date.now() - AppState.startTime) / 1000).toFixed(1);
        UI.progressTimer.textContent = `${elapsed}s`;
        if (elapsed > 10) {
            UI.progressStatus.textContent = "Denoising latents...";
        }
    }, 100);
}

function stopProgress() {
    AppState.isGenerating = false;
    UI.generateBtn.disabled = false;
    UI.generateBtnText.textContent = "Generate Image";
    UI.progressContainer.style.display = "none";
    if (AppState.timerInterval) {
        clearInterval(AppState.timerInterval);
        AppState.timerInterval = null;
    }
}

function handleGenerationSuccess(data) {
    showToast(`Generated ${data.images.length} image(s) in ${data.duration_seconds}s!`);
    UI.resultSummary.textContent = `${data.images.length} image(s) • ${data.duration_seconds}s`;

    UI.galleryGrid.innerHTML = "";
    data.images.forEach((filename, idx) => {
        const item = document.createElement("div");
        item.className = "gallery-item";
        item.innerHTML = `
            <img src="/images/${filename}" alt="Generated image #${idx + 1}" loading="lazy">
            <div class="gallery-overlay">
                <span class="gallery-overlay-text">${filename}</span>
            </div>
        `;
        item.addEventListener("click", () => {
            openLightbox(`/images/${filename}`, data.parameters, filename);
        });
        UI.galleryGrid.appendChild(item);
    });

    loadHistory();
}

// ==============================================================================
// Lightbox Modal
// ==============================================================================

let currentLightboxData = null;

function openLightbox(imageUrl, parameters, filename) {
    currentLightboxData = { imageUrl, parameters, filename };
    UI.lightboxImg.src = imageUrl;

    const params = parameters || {};
    UI.lightboxMeta.innerHTML = `
        <div class="meta-field">
            <strong>Prompt</strong>
            <div class="meta-box">${params.prompt || "N/A"}</div>
        </div>
        ${params.negative_prompt ? `
        <div class="meta-field">
            <strong>Negative Prompt</strong>
            <div class="meta-box">${params.negative_prompt}</div>
        </div>` : ""}
        <div class="meta-field">
            <strong>Settings</strong>
            <div>${params.width || 1024}×${params.height || 1024} • ${params.steps || 30} steps • CFG ${params.guidance_scale || 7.5} • Seed ${params.seed || "Random"}</div>
            <div>Sampler: ${params.scheduler || "euler_ancestral"}</div>
            <div>Model: ${params.model_id || "SDXL"}</div>
        </div>
    `;

    UI.lightboxDownloadLink.href = imageUrl;
    UI.lightboxDownloadLink.download = filename || "sdxl_image.png";

    UI.lightboxModal.classList.add("active");
}

function closeLightbox() {
    UI.lightboxModal.classList.remove("active");
}

function setupLightbox() {
    UI.lightboxClose.addEventListener("click", closeLightbox);
    UI.lightboxBackdrop.addEventListener("click", closeLightbox);

    UI.lightboxCopyBtn.addEventListener("click", () => {
        if (currentLightboxData && currentLightboxData.parameters && currentLightboxData.parameters.prompt) {
            navigator.clipboard.writeText(currentLightboxData.parameters.prompt);
            showToast("Copied prompt to clipboard!");
        }
    });

    UI.lightboxReuseBtn.addEventListener("click", () => {
        if (currentLightboxData && currentLightboxData.parameters) {
            applyExtractedMetadata(currentLightboxData.parameters);
            closeLightbox();
            showToast("Parameters loaded into form!");
        }
    });

    window.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && UI.lightboxModal.classList.contains("active")) {
            closeLightbox();
        }
    });
}

// ==============================================================================
// History Drawer
// ==============================================================================

async function loadHistory() {
    try {
        const res = await fetch("/api/history");
        if (!res.ok) return;
        AppState.history = await res.json();
        renderHistory();
    } catch (e) {
        console.warn("Could not fetch history:", e);
    }
}

function renderHistory() {
    if (!UI.historyList) return;
    if (AppState.history.length === 0) {
        UI.historyList.innerHTML = '<div class="empty-text">No generation history yet.</div>';
        return;
    }

    UI.historyList.innerHTML = "";
    AppState.history.forEach((entry) => {
        const firstImg = entry.filenames && entry.filenames[0] ? entry.filenames[0] : "";
        const item = document.createElement("div");
        item.className = "history-item";
        item.innerHTML = `
            <img src="/images/${firstImg}" class="history-thumb" alt="thumb" loading="lazy">
            <div class="history-info">
                <div class="history-prompt" title="${entry.prompt}">${entry.prompt}</div>
                <div class="history-time">${entry.display_time || ""} • ${entry.filenames.length} img</div>
            </div>
        `;
        item.addEventListener("click", () => {
            if (firstImg) {
                openLightbox(`/images/${firstImg}`, entry.parameters, firstImg);
            }
        });
        UI.historyList.appendChild(item);
    });
}

function setupHistoryDrawer() {
    UI.historyToggleBtn.addEventListener("click", () => {
        UI.historyDrawer.classList.add("active");
        UI.historyOverlay.classList.add("active");
        loadHistory();
    });

    const closeDrawer = () => {
        UI.historyDrawer.classList.remove("active");
        UI.historyOverlay.classList.remove("active");
    };

    UI.closeHistoryBtn.addEventListener("click", closeDrawer);
    UI.historyOverlay.addEventListener("click", closeDrawer);
}

// ==============================================================================
// State Persistence (localStorage)
// ==============================================================================

const STORAGE_KEY = "sdxl_studio_state_v2";

function saveFormState() {
    const isCivitai = UI.tabCivitai.classList.contains("active");
    const state = {
        prompt: UI.prompt.value,
        negative_prompt: UI.negativePrompt.value,
        is_civitai: isCivitai,
        model_id: UI.modelId.value,
        civitai_id: UI.civitaiId.value,
        width: UI.width.value,
        height: UI.height.value,
        steps: UI.steps.value,
        guidance_scale: UI.guidanceScale.value,
        scheduler: UI.scheduler.value,
        seed: UI.seed.value,
        batch_size: UI.batchSize.value,
        batch_count: UI.batchCount.value,
        clip_skip: UI.clipSkip.value,
        freeu_enabled: UI.freeuEnabled.checked,
        loras: AppState.loras
    };
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch (e) {}
}

function loadFormState() {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) {
            renderLoras();
            return;
        }
        const state = JSON.parse(raw);
        if (state.prompt !== undefined) UI.prompt.value = state.prompt;
        if (state.negative_prompt !== undefined) UI.negativePrompt.value = state.negative_prompt;
        if (state.is_civitai) {
            UI.tabCivitai.click();
        } else {
            UI.tabHf.click();
        }
        if (state.model_id) UI.modelId.value = state.model_id;
        if (state.civitai_id) UI.civitaiId.value = state.civitai_id;
        if (state.width) UI.width.value = state.width;
        if (state.height) UI.height.value = state.height;
        if (state.steps) {
            UI.steps.value = state.steps;
            UI.stepsVal.textContent = state.steps;
        }
        if (state.guidance_scale) {
            UI.guidanceScale.value = state.guidance_scale;
            UI.guidanceVal.textContent = parseFloat(state.guidance_scale).toFixed(1);
        }
        if (state.scheduler) UI.scheduler.value = state.scheduler;
        if (state.seed) UI.seed.value = state.seed;
        if (state.batch_size) UI.batchSize.value = state.batch_size;
        if (state.batch_count) UI.batchCount.value = state.batch_count;
        if (state.clip_skip) UI.clipSkip.value = state.clip_skip;
        if (state.freeu_enabled) {
            UI.freeuEnabled.checked = true;
            UI.freeuParams.style.display = "block";
        }
        if (state.loras && Array.isArray(state.loras)) {
            AppState.loras = state.loras;
        }
        renderLoras();
    } catch (e) {
        renderLoras();
    }
}

// ==============================================================================
// Keyboard Shortcuts & Initialization
// ==============================================================================

function setupKeyboardShortcuts() {
    window.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
            e.preventDefault();
            triggerGeneration();
        }
    });
}

document.addEventListener("DOMContentLoaded", async () => {
    initElements();
    setupModelTabs();
    setupDimensionPresets();
    setupSliders();
    setupLoraControls();
    setupPresets();
    setupDropzone();
    setupLightbox();
    setupHistoryDrawer();
    setupKeyboardShortcuts();

    UI.generateBtn.addEventListener("click", triggerGeneration);

    [UI.prompt, UI.negativePrompt, UI.modelId, UI.civitaiId, UI.width, UI.height, UI.seed, UI.batchSize, UI.batchCount, UI.clipSkip].forEach((input) => {
        input.addEventListener("input", saveFormState);
    });

    await loadPresets();
    loadFormState();
    loadHistory();
});