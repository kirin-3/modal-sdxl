import base64
import io
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from PIL import Image

# Load environment configuration
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
OUTPUT_DIR = BASE_DIR / "generated_images"
HISTORY_FILE = OUTPUT_DIR / "history.json"

STATIC_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Default constants
DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_SCHEDULER = "euler_ancestral"

AVAILABLE_SCHEDULERS = {
    "euler_ancestral": "Euler Ancestral (Best overall, creative)",
    "dpmpp_2m_karras": "DPM++ 2M Karras (Sharp, high quality)",
    "dpmpp_sde_karras": "DPM++ SDE Karras (Rich details)",
    "unipc": "UniPC (Fast convergence)",
    "euler": "Euler (Classic, smooth)",
    "ddim": "DDIM (Deterministic)",
}

BUILTIN_PRESETS = {
    "photorealistic": {
        "name": "Photorealistic",
        "prompt_suffix": ", 8k resolution, raw photo, highly detailed, realistic lighting, f/1.8 lens, DSLR",
        "negative_prompt": "cartoon, illustration, 3d render, painting, oversaturated, blurry, bad anatomy, deformed",
        "steps": 35,
        "guidance_scale": 7.0,
        "scheduler": "dpmpp_2m_karras",
    },
    "cinematic": {
        "name": "Cinematic Film",
        "prompt_suffix": ", 35mm photograph, film grain, cinematic lighting, masterpiece, anamorphic lens, shallow depth of field",
        "negative_prompt": "digital art, low quality, flat lighting, amateur, watermark, signature",
        "steps": 30,
        "guidance_scale": 7.5,
        "scheduler": "euler_ancestral",
    },
    "anime": {
        "name": "Anime / Manga",
        "prompt_suffix": ", anime aesthetic, vibrant colors, clean linework, studio quality, makoto shinkai style",
        "negative_prompt": "photorealistic, 3d, western comic, blurry, lowres, bad hands, missing fingers",
        "steps": 28,
        "guidance_scale": 8.0,
        "scheduler": "euler_ancestral",
    },
    "digital_art": {
        "name": "Digital Painting",
        "prompt_suffix": ", digital art, concept art, trending on artstation, detailed illustration, dynamic lighting, sharp focus",
        "negative_prompt": "photo, photorealistic, ugly, distorted, low quality, artifacting",
        "steps": 30,
        "guidance_scale": 7.5,
        "scheduler": "euler_ancestral",
    },
    "cyberpunk": {
        "name": "Cyberpunk Neon",
        "prompt_suffix": ", cyberpunk city, neon lights, volumetric fog, dark night, futuristic, ray tracing, highly detailed",
        "negative_prompt": "daylight, sunny, rustic, vintage, lowres, blurry",
        "steps": 32,
        "guidance_scale": 8.0,
        "scheduler": "dpmpp_sde_karras",
    },
}


# ==============================================================================
# Pydantic Schemas
# ==============================================================================

class LoRAItem(BaseModel):
    model_id: str
    weight: float = 0.75


class FreeUItem(BaseModel):
    enabled: bool = False
    s1: float = 0.9
    s2: float = 0.2
    b1: float = 1.3
    b2: float = 1.4


class GeneratePayload(BaseModel):
    prompt: str
    negative_prompt: str = ""
    batch_size: int = Field(default=1, ge=1, le=4)
    batch_count: int = Field(default=1, ge=1, le=4)
    seed: Optional[int] = None
    model_id: str = DEFAULT_MODEL_ID
    width: int = Field(default=DEFAULT_WIDTH, ge=512, le=2048)
    height: int = Field(default=DEFAULT_HEIGHT, ge=512, le=2048)
    steps: int = Field(default=DEFAULT_STEPS, ge=1, le=150)
    guidance_scale: float = Field(default=DEFAULT_GUIDANCE_SCALE, ge=1.0, le=20.0)
    clip_skip: Optional[int] = Field(default=None, ge=1, le=4)
    scheduler: str = DEFAULT_SCHEDULER
    loras: Optional[List[LoRAItem]] = None
    freeu: Optional[FreeUItem] = None


# ==============================================================================
# FastAPI Application & Async Client
# ==============================================================================

app = FastAPI(
    title="SDXL Image Generator",
    description="Modern Asynchronous Local Server for Modal SDXL Inference",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def slugify(text: str) -> str:
    """Convert prompt text to URL-friendly filename segment."""
    clean = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[-\s]+", "-", clean).strip("-_")[:40] or "image"


def load_history() -> List[Dict[str, Any]]:
    """Load history index from disk."""
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_history_entry(entry: Dict[str, Any]):
    """Prepend entry to history index file."""
    history = load_history()
    history.insert(0, entry)
    # Keep last 100 entries
    history = history[:100]
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"Error saving history: {e}")


def parse_png_metadata(image_bytes: bytes) -> Dict[str, Any]:
    """Extract standard A1111 / JSON generation parameters from PNG chunks."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        info = img.info or {}

        # 1. Try structured JSON metadata
        if "sdxl_metadata" in info:
            return json.loads(info["sdxl_metadata"])

        # 2. Try A1111 parameters chunk
        if "parameters" in info:
            raw = info["parameters"]
            lines = raw.strip().split("\n")
            meta: Dict[str, Any] = {"raw_parameters": raw}

            if len(lines) >= 1:
                meta["prompt"] = lines[0]
            if len(lines) >= 2 and lines[1].startswith("Negative prompt:"):
                meta["negative_prompt"] = lines[1].replace("Negative prompt:", "").strip()

            # Parse parameter key-values from last line
            last_line = lines[-1]
            for match in re.finditer(r"([A-Za-z ]+):\s*([^,]+)", last_line):
                k = match.group(1).strip().lower().replace(" ", "_")
                v = match.group(2).strip()
                if k == "steps":
                    meta["steps"] = int(v)
                elif k == "sampler":
                    meta["scheduler"] = v
                elif k == "cfg_scale":
                    meta["guidance_scale"] = float(v)
                elif k == "seed":
                    meta["seed"] = int(v)
                elif k == "size":
                    if "x" in v:
                        w, h = v.split("x")
                        meta["width"] = int(w)
                        meta["height"] = int(h)
                elif k == "model":
                    meta["model_id"] = v
                elif k == "clip_skip":
                    meta["clip_skip"] = int(v)

            return meta

        return {"prompt": info.get("prompt", ""), "negative_prompt": info.get("negative_prompt", "")}
    except Exception as e:
        return {"error": f"Failed to extract metadata: {str(e)}"}


# ==============================================================================
# Routes
# ==============================================================================

@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    """Render the primary single-page UI."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "default_model": DEFAULT_MODEL_ID,
            "default_width": DEFAULT_WIDTH,
            "default_height": DEFAULT_HEIGHT,
            "default_steps": DEFAULT_STEPS,
            "default_guidance_scale": DEFAULT_GUIDANCE_SCALE,
            "default_scheduler": DEFAULT_SCHEDULER,
            "schedulers": AVAILABLE_SCHEDULERS,
            "presets": BUILTIN_PRESETS,
            "output_directory": str(OUTPUT_DIR.resolve()),
        },
    )


@app.get("/api/presets")
async def get_presets():
    """Retrieve available prompt style presets."""
    return BUILTIN_PRESETS


@app.get("/api/history")
async def get_history():
    """Retrieve indexed generation history."""
    return load_history()


@app.post("/api/metadata")
async def extract_metadata(file: UploadFile = File(...)):
    """Extract embedded generation parameters from an uploaded PNG."""
    content = await file.read()
    return parse_png_metadata(content)


@app.get("/images/{filename}")
async def serve_image(filename: str):
    """Serve generated image from local disk with caching headers."""
    safe_name = Path(filename).name
    file_path = OUTPUT_DIR / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(
        file_path,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.post("/api/generate")
async def generate_images(payload: GeneratePayload):
    """
    Proxy generation request asynchronously to remote Modal endpoint,
    save results to disk, and index metadata.
    """
    endpoint = os.getenv("MODAL_ENDPOINT", "").strip()
    if not endpoint:
        raise HTTPException(
            status_code=500,
            detail="MODAL_ENDPOINT is not configured in .env. Please deploy your Modal app and update .env.",
        )

    # Normalize Modal endpoint URL for POST
    if not endpoint.endswith("/generate") and not endpoint.endswith(".modal.run"):
        pass

    modal_request_data = {
        "prompt": payload.prompt,
        "negative_prompt": payload.negative_prompt,
        "batch_size": payload.batch_size,
        "batch_count": payload.batch_count,
        "seed": payload.seed,
        "model_id": payload.model_id,
        "width": payload.width,
        "height": payload.height,
        "num_inference_steps": payload.steps,
        "guidance_scale": payload.guidance_scale,
        "clip_skip": payload.clip_skip,
        "scheduler": payload.scheduler,
        "loras": [l.model_dump() for l in payload.loras] if payload.loras else None,
        "freeu": payload.freeu.model_dump() if payload.freeu else None,
    }

    start_time = time.time()
    timeout = httpx.Timeout(1800.0, connect=60.0)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # Try POST to /generate endpoint or base endpoint
            target_url = endpoint
            if not target_url.endswith("/generate"):
                if target_url.endswith("/"):
                    target_url += "generate"
                else:
                    target_url += "/generate"

            response = await client.post(target_url, json=modal_request_data)

            # Fallback to base URL if /generate 404s
            if response.status_code == 404:
                response = await client.post(endpoint, json=modal_request_data)

            response.raise_for_status()

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Generation timed out on Modal. Try fewer steps or smaller batch size.",
        )
    except httpx.ConnectError:
        raise HTTPException(
            status_code=502,
            detail=f"Could not connect to Modal endpoint at {endpoint}. Ensure the app is deployed.",
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Modal endpoint returned error: {e.response.text}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error communicating with Modal: {str(e)}",
        )

    # Process response
    saved_filenames = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_slug = slugify(payload.prompt)

    content_type = response.headers.get("content-type", "")

    if "application/json" in content_type:
        try:
            data = response.json()
            images_b64 = data.get("images", [])
            for idx, b64_str in enumerate(images_b64):
                img_bytes = base64.b64decode(b64_str)
                filename = f"{timestamp}_{prompt_slug}_{idx+1}.png"
                file_path = OUTPUT_DIR / filename
                with open(file_path, "wb") as f:
                    f.write(img_bytes)
                saved_filenames.append(filename)
        except Exception as json_err:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process JSON images from Modal: {str(json_err)}",
            )
    else:
        # Binary image response
        filename = f"{timestamp}_{prompt_slug}.png"
        file_path = OUTPUT_DIR / filename
        with open(file_path, "wb") as f:
            f.write(response.content)
        saved_filenames.append(filename)

    elapsed = round(time.time() - start_time, 2)

    # Save to history index
    history_entry = {
        "timestamp": datetime.now().isoformat(),
        "display_time": datetime.now().strftime("%b %d, %H:%M:%S"),
        "filenames": saved_filenames,
        "prompt": payload.prompt,
        "negative_prompt": payload.negative_prompt,
        "parameters": payload.model_dump(),
        "duration_seconds": elapsed,
    }
    save_history_entry(history_entry)

    return {
        "success": True,
        "images": saved_filenames,
        "image_urls": [f"/images/{name}" for name in saved_filenames],
        "duration_seconds": elapsed,
        "parameters": payload.model_dump(),
    }


# ==============================================================================
# Server Entrypoint
# ==============================================================================

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))
    print(f"=== Starting Modernized SDXL FastAPI Server on http://{host}:{port} ===")
    uvicorn.run("local_server:app", host=host, port=port, reload=True)
