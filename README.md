# SDXL Studio (Modal + FastAPI)

A modernized, high-performance text-to-image generation platform powered by **[Modal](https://modal.com/)**, **Stable Diffusion XL**, and an asynchronous **FastAPI** web interface.

---

## ⚡ Highlights

- **Modern Serverless Engine**: Powered by Modal serverless GPUs (L4, A10G, A100, H100) with automatic scale-down and persistent model caching.
- **Identical A1111-Style Prompt Chunking**: Preserves the Automatic1111 dual-CLIP text encoder token chunking and hidden-state concatenation algorithm (`_encode_prompt_chunked`) for long prompts (>77 tokens).
- **PyTorch 2.x SDPA Attention**: Native Scaled Dot-Product Attention for fast, memory-efficient generation without legacy xformers build overhead.
- **Clean Diffusers LoRA Management**: Multi-LoRA stacking (up to 5 LoRAs) from CivitAI or Hugging Face with native PEFT lifecycle management (`load_lora_weights`, `set_adapters`, `unload_lora_weights`).
- **Embedded PNG Metadata**: Outputs standard A1111 / ComfyUI compatible `tEXt` chunks (Prompt, Negative Prompt, Seed, Sampler, Steps, CFG, Model, LoRAs).
- **Drag-and-Drop Parameter Restoration**: Drag any generated PNG directly into the web UI to instantly restore all generation settings.
- **Extended Samplers & FreeU**: Includes 6 schedulers (Euler Ancestral, DPM++ 2M Karras, DPM++ SDE Karras, UniPC, Euler, DDIM) and optional FreeU frequency enhancement.
- **Lean Reactive Frontend**: Fast single-page application (ES6+, modern CSS) with live generation timers, style presets, history drawer, and full-resolution lightbox modal.
- **Asynchronous Local Server**: Built on FastAPI and Uvicorn with `httpx` async client for non-blocking execution.

---

## 📦 Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Browser UI (Lean Reactive SPA)                                          │
│   • Live timer • Style Presets • Lightbox • Drag-and-drop PNG loader    │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ Async REST API
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Local Server (FastAPI + Uvicorn + httpx)                                │
│   • /api/generate • /api/presets • /api/history • /api/metadata        │
│   • Local image saving & history indexing in generated_images/          │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ Typed JSON POST
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Modal Inference Service (text2image.py)                                 │
│   • Pydantic Request / Response Validation                              │
│   • Diffusers SDXL Pipeline + PyTorch SDPA                              │
│   • A1111-style Dual-CLIP Prompt Chunking (Preserved)                   │
│   • Multi-LoRA Fusion & CivitAI Cache Volume                            │
│   • PNG tEXt Metadata Injection                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.12+ (or 3.10+)
- A [Modal](https://modal.com) account
- (Optional) A [CivitAI](https://civitai.com) account and API token for downloading restricted models/LoRAs

### 2. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/kirin-3/modal-sdxl
cd modal-sdxl
pip install -r requirements.txt
```

### 3. Configure Modal & CivitAI Token

Authenticate the Modal CLI:

```bash
modal setup
```

Create your CivitAI API token secret in Modal (required for CivitAI downloads):

```bash
modal secret create civitai-token CIVITAI_TOKEN="your_civitai_api_token"
```

### 4. Deploy to Modal

Deploy the single-file Modal inference application:

```bash
modal deploy text2image.py
```

Copy the deployed endpoint URL (e.g., `https://<username>--text2image-inference-generate.modal.run`).

### 5. Configure Local Environment

Copy the example environment file:

```bash
cp example.env .env
```

Edit `.env` and set your `MODAL_ENDPOINT`:

```env
MODAL_ENDPOINT=https://<your-username>--text2image-inference-generate.modal.run
GPU_TYPE=L4
HOST=0.0.0.0
PORT=5000
```

### 6. Start the Local Server

```bash
python local_server.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 🎨 Web Interface Features

- **Prompt Studio**: Enter positive and negative prompts. Supports Automatic1111 token weighting syntax and chunked prompts of any length.
- **Style Presets**: One-click style defaults for *Photorealistic*, *Cinematic Film*, *Anime / Manga*, *Digital Painting*, and *Cyberpunk Neon*.
- **Drag & Drop PNG Loader**: Drag any generated PNG image onto the prompt area to instantly load its prompt, seed, sampler, steps, CFG, and LoRAs.
- **LoRA Manager**: Add up to 5 LoRAs from CivitAI (`civitai:<ID>`) or Hugging Face (`hf:<repo>/<file>`) with custom weights.
- **Lightbox Gallery**: Click any thumbnail in the gallery to open a full-resolution inspection modal with zoom, download, prompt copying, and parameter reuse.
- **History Drawer**: Access previously generated images and their full metadata directly from the slide-out history panel.

---

## 📡 Programmatic API

The Modal application exposes a typed JSON POST endpoint:

```python
import requests
import json

url = "https://yourusername--text2image-inference-generate.modal.run"

payload = {
    "prompt": "A breathtaking cinematic landscape, misty mountains, golden hour, 8k, photorealistic",
    "negative_prompt": "cartoon, low quality, blurry, watermark",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "scheduler": "euler_ancestral",
    "batch_size": 1,
    "batch_count": 1,
    "loras": [
        {"model_id": "civitai:1681903", "weight": 2.0},
        {"model_id": "civitai:1764869", "weight": 0.75}
    ],
    "freeu": {
        "enabled": False,
        "b1": 1.3,
        "b2": 1.4,
        "s1": 0.9,
        "s2": 0.2
    }
}

response = requests.post(url, json=payload)
data = response.json()

print(f"Generated {len(data['images'])} image(s) in {data['duration_seconds']}s")
```

---

## 🧪 CLI Testing

You can also generate images directly from the command line using Modal's local entrypoint:

```bash
modal run text2image.py --prompt "A photorealistic portrait of an astronaut on Mars, 8k" --steps 30
```

Images will be saved to `./generated_images/`.

---

## 📜 License

MIT