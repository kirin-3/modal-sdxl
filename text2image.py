import io
import os
import json
import random
import time
import requests
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Literal, TYPE_CHECKING
from pydantic import BaseModel, Field
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import modal

# Import torch for type hints only outside container
if TYPE_CHECKING:
    import torch

MINUTES = 60

app = modal.App("text2image")

CACHE_DIR = "/cache"
CIVITAI_MODELS_DIR = "/cache/civitai"
CIVITAI_LORAS_DIR = "/cache/civitai/loras"
HF_LORAS_DIR = "/cache/hf/loras"

# Available scheduler types
SchedulerType = Literal[
    "euler_ancestral",
    "dpmpp_2m_karras",
    "dpmpp_sde_karras",
    "unipc",
    "euler",
    "ddim",
]

# GPU selection from environment variable
GPU_TYPE = os.environ.get("GPU_TYPE", "L4")

# Modal container image
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "accelerate>=1.0.0",
        "diffusers>=0.31.0",
        "fastapi[standard]>=0.115.0",
        "huggingface-hub[hf_transfer]>=0.25.0",
        "pydantic>=2.9.0",
        "sentencepiece>=0.2.0",
        "torch>=2.5.1",
        "torchvision>=0.20.1",
        "transformers>=4.45.0",
        "requests>=2.31.0",
        "safetensors>=0.4.5",
        "peft>=0.13.0",
        "pillow>=10.4.0",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HUB_CACHE": CACHE_DIR,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)

civitai_secret = modal.Secret.from_name("civitai-token")
cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

# Default generation constants
DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_SCHEDULER: SchedulerType = "euler_ancestral"
MAX_TOKEN_LENGTH = 77


# ==============================================================================
# Pydantic Schemas for API Validation
# ==============================================================================

class LoRASpec(BaseModel):
    model_id: str = Field(..., description="CivitAI ID (civitai:ID) or HuggingFace repo (hf:REPO/PATH)")
    weight: float = Field(default=0.75, ge=-2.0, le=3.0, description="LoRA multiplier weight")


class FreeUSpec(BaseModel):
    enabled: bool = Field(default=False, description="Enable FreeU frequency enhancement")
    s1: float = Field(default=0.9, description="Stage 1 scaling factor")
    s2: float = Field(default=0.2, description="Stage 2 scaling factor")
    b1: float = Field(default=1.3, description="Backbone 1 scaling factor")
    b2: float = Field(default=1.4, description="Backbone 2 scaling factor")


class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Positive prompt")
    negative_prompt: str = Field(default="", description="Negative prompt")
    batch_size: int = Field(default=1, ge=1, le=4, description="Images per batch")
    batch_count: int = Field(default=1, ge=1, le=4, description="Number of sequential batches")
    seed: Optional[int] = Field(default=None, description="Random seed")
    model_id: str = Field(default=DEFAULT_MODEL_ID, description="Model ID (HuggingFace or civitai:ID)")
    width: int = Field(default=DEFAULT_WIDTH, ge=512, le=2048, description="Image width (multiple of 8)")
    height: int = Field(default=DEFAULT_HEIGHT, ge=512, le=2048, description="Image height (multiple of 8)")
    num_inference_steps: int = Field(default=DEFAULT_STEPS, ge=1, le=150, description="Denoising steps")
    guidance_scale: float = Field(default=DEFAULT_GUIDANCE_SCALE, ge=1.0, le=20.0, description="CFG guidance scale")
    clip_skip: Optional[int] = Field(default=None, ge=1, le=4, description="CLIP skip layer count")
    scheduler: SchedulerType = Field(default=DEFAULT_SCHEDULER, description="Sampling scheduler algorithm")
    loras: Optional[List[LoRASpec]] = Field(default=None, description="List of LoRA specifications")
    freeu: Optional[FreeUSpec] = Field(default=None, description="FreeU enhancement configuration")


class GenerationResponse(BaseModel):
    images: List[str] = Field(..., description="Base64 encoded PNG images with embedded metadata")
    parameters: Dict[str, Any] = Field(..., description="Parameters used for this generation run")
    duration_seconds: float = Field(..., description="Execution duration in seconds")


with image.imports():
    import diffusers
    import torch
    from diffusers import (
        AutoencoderKL,
        StableDiffusionXLPipeline,
        DDIMScheduler,
        EulerDiscreteScheduler,
        EulerAncestralDiscreteScheduler,
        DPMSolverMultistepScheduler,
        DPMSolverSDEScheduler,
        UniPCMultistepScheduler,
    )
    from huggingface_hub import hf_hub_download, list_repo_files
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer, CLIPTextConfig

    def get_gpu_memory_info() -> Dict[str, Any]:
        """Get GPU memory usage in MB."""
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            reserved = torch.cuda.memory_reserved(0) / (1024**2)
            allocated = torch.cuda.memory_allocated(0) / (1024**2)
            free = total - (reserved + allocated)
            return {
                "total_mb": round(total, 1),
                "reserved_mb": round(reserved, 1),
                "allocated_mb": round(allocated, 1),
                "free_mb": round(free, 1),
            }
        return {"error": "CUDA not available"}


# ==============================================================================
# Modal Inference Class
# ==============================================================================

@app.cls(
    image=image,
    gpu=GPU_TYPE,
    timeout=10 * MINUTES,
    scaledown_window=5 * MINUTES,
    volumes={CACHE_DIR: cache_volume},
    secrets=[civitai_secret],
)
class Inference:
    load_default_model: bool = modal.parameter(default=False)

    AVAILABLE_SCHEDULERS = {
        "euler_ancestral": "Euler Ancestral",
        "dpmpp_2m_karras": "DPM++ 2M Karras",
        "dpmpp_sde_karras": "DPM++ SDE Karras",
        "unipc": "UniPC",
        "euler": "Euler",
        "ddim": "DDIM",
    }

    @modal.enter()
    def setup(self):
        """Initialize instance state and directories on container start."""
        self.loaded_models: Dict[str, Any] = {}
        self.model_last_accessed: Dict[str, float] = {}
        self.max_loaded_models: int = 2
        self.active_loras: Dict[str, List[str]] = {}

        os.makedirs(CIVITAI_MODELS_DIR, exist_ok=True)
        os.makedirs(CIVITAI_LORAS_DIR, exist_ok=True)
        os.makedirs(HF_LORAS_DIR, exist_ok=True)

        print(f"Container initialized. GPU info: {get_gpu_memory_info()}")

        if self.load_default_model:
            print(f"Preloading default model: {DEFAULT_MODEL_ID}")
            try:
                self._load_pipeline(DEFAULT_MODEL_ID)
                print("Default model preloaded successfully.")
            except Exception as e:
                print(f"Warning: Failed to preload default model: {e}")

    # ==========================================================================
    # A1111-Style Dual-CLIP Prompt Chunking (PRESERVED IDENTICAL)
    # ==========================================================================
    def _encode_prompt_chunked(
        self,
        pipe,
        prompt: str,
        negative_prompt: str,
        device: "torch.device",
        batch_size: int = 1,
        max_length: int = MAX_TOKEN_LENGTH,
    ):
        """
        Encodes positive and negative prompts using token-based chunking and concatenates embeddings.
        Mimics Automatic1111's approach to handling long prompts by concatenating token embeddings.
        Handles dual text encoders for SDXL and batching.
        """
        tokenizer, tokenizer_2 = pipe.tokenizer, pipe.tokenizer_2
        text_encoder, text_encoder_2 = pipe.text_encoder, pipe.text_encoder_2

        # Tokenize original prompts without truncation/padding initially
        pos_input_ids_1_raw = tokenizer(prompt, add_special_tokens=False, truncation=False, return_tensors="pt").input_ids[0].tolist()
        pos_input_ids_2_raw = tokenizer_2(prompt, add_special_tokens=False, truncation=False, return_tensors="pt").input_ids[0].tolist()
        neg_input_ids_1_raw = tokenizer(negative_prompt or "", add_special_tokens=False, truncation=False, return_tensors="pt").input_ids[0].tolist()
        neg_input_ids_2_raw = tokenizer_2(negative_prompt or "", add_special_tokens=False, truncation=False, return_tensors="pt").input_ids[0].tolist()

        # Determine total max length across all tokenizations
        total_max_len = max(
            len(pos_input_ids_1_raw),
            len(pos_input_ids_2_raw),
            len(neg_input_ids_1_raw),
            len(neg_input_ids_2_raw),
        )

        num_special_tokens = tokenizer.num_special_tokens_to_add(pair=False) if hasattr(tokenizer, 'num_special_tokens_to_add') else 2
        effective_chunk_size = max_length - num_special_tokens

        if effective_chunk_size <= 0:
            raise ValueError(f"Effective chunk size is too small or zero: {effective_chunk_size}.")

        num_chunks = (total_max_len + effective_chunk_size - 1) // effective_chunk_size

        # Non-chunking case: fits within 75 tokens
        if num_chunks <= 1:
            print("Prompt fits within max effective token length, using standard encoding.")
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
            )
            return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

        # Token-based chunking case (Mimics A1111)
        print(f"Prompt requires token-based chunking into {num_chunks} chunks.")

        final_prompt_embeds = None
        final_negative_prompt_embeds = None
        first_chunk_pooled_prompt_embeds = None
        first_chunk_negative_pooled_prompt_embeds = None

        pad_token_id_1 = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        pad_token_id_2 = tokenizer_2.pad_token_id if tokenizer_2.pad_token_id is not None else tokenizer_2.eos_token_id

        text_encoder.eval()
        text_encoder_2.eval()

        with torch.no_grad():
            for i in range(num_chunks):
                overlap_tokens = 5 if i > 0 else 0
                start_idx = max(0, i * effective_chunk_size - overlap_tokens)
                end_idx = min(start_idx + effective_chunk_size, total_max_len)

                # Positive Prompt Chunk
                pos_chunk_ids_1_slice = pos_input_ids_1_raw[start_idx:end_idx]
                pos_chunk_ids_2_slice = pos_input_ids_2_raw[start_idx:end_idx]

                # Encoder 1
                pos_chunk_ids_1_padded = [tokenizer.bos_token_id] + pos_chunk_ids_1_slice + [tokenizer.eos_token_id]
                pos_chunk_attention_mask_1 = [1] * len(pos_chunk_ids_1_padded)
                padding_length_1 = max_length - len(pos_chunk_ids_1_padded)
                if padding_length_1 > 0:
                    pos_chunk_ids_1_padded += [pad_token_id_1] * padding_length_1
                    pos_chunk_attention_mask_1 += [0] * padding_length_1
                pos_chunk_ids_1_tensor = torch.tensor([pos_chunk_ids_1_padded], device=device)
                pos_chunk_attention_mask_1_tensor = torch.tensor([pos_chunk_attention_mask_1], device=device)

                # Encoder 2
                pos_chunk_ids_2_padded = [tokenizer_2.bos_token_id] + pos_chunk_ids_2_slice + [tokenizer_2.eos_token_id]
                pos_chunk_attention_mask_2 = [1] * len(pos_chunk_ids_2_padded)
                padding_length_2 = max_length - len(pos_chunk_ids_2_padded)
                if padding_length_2 > 0:
                    pos_chunk_ids_2_padded += [pad_token_id_2] * padding_length_2
                    pos_chunk_attention_mask_2 += [0] * padding_length_2
                pos_chunk_ids_2_tensor = torch.tensor([pos_chunk_ids_2_padded], device=device)
                pos_chunk_attention_mask_2_tensor = torch.tensor([pos_chunk_attention_mask_2], device=device)

                pos_chunk_embeds_1 = text_encoder(
                    pos_chunk_ids_1_tensor,
                    attention_mask=pos_chunk_attention_mask_1_tensor,
                    output_hidden_states=True,
                ).hidden_states[-2]

                pos_chunk_output_2 = text_encoder_2(
                    pos_chunk_ids_2_tensor,
                    attention_mask=pos_chunk_attention_mask_2_tensor,
                    output_hidden_states=True,
                    return_dict=True,
                )
                pos_chunk_embeds_2 = pos_chunk_output_2.hidden_states[-2]

                if hasattr(pos_chunk_output_2, 'text_embeds'):
                    pos_pooled_embeds_2 = pos_chunk_output_2.text_embeds
                elif hasattr(pos_chunk_output_2, 'pooled_output'):
                    pos_pooled_embeds_2 = pos_chunk_output_2.pooled_output
                else:
                    pos_pooled_embeds_2 = pos_chunk_output_2.last_hidden_state[:, 0]

                seq_len = max(pos_chunk_embeds_1.shape[1], pos_chunk_embeds_2.shape[1])
                if pos_chunk_embeds_1.shape[1] != seq_len:
                    pos_chunk_embeds_1 = torch.nn.functional.pad(pos_chunk_embeds_1, (0, 0, 0, seq_len - pos_chunk_embeds_1.shape[1]), value=0)
                if pos_chunk_embeds_2.shape[1] != seq_len:
                    pos_chunk_embeds_2 = torch.nn.functional.pad(pos_chunk_embeds_2, (0, 0, 0, seq_len - pos_chunk_embeds_2.shape[1]), value=0)

                chunk_combined_pos_embeds = torch.cat([pos_chunk_embeds_1, pos_chunk_embeds_2], dim=-1)

                if i == 0:
                    first_chunk_pooled_prompt_embeds = pos_pooled_embeds_2

                if final_prompt_embeds is None:
                    final_prompt_embeds = chunk_combined_pos_embeds
                else:
                    final_prompt_embeds = torch.cat([final_prompt_embeds, chunk_combined_pos_embeds], dim=1)

                # Negative Prompt Chunk
                neg_chunk_ids_1_slice = neg_input_ids_1_raw[start_idx:end_idx]
                neg_chunk_ids_2_slice = neg_input_ids_2_raw[start_idx:end_idx]

                neg_chunk_ids_1_padded = [tokenizer.bos_token_id] + neg_chunk_ids_1_slice + [tokenizer.eos_token_id]
                neg_chunk_attention_mask_1 = [1] * len(neg_chunk_ids_1_padded)
                padding_length_1_neg = max_length - len(neg_chunk_ids_1_padded)
                if padding_length_1_neg > 0:
                    neg_chunk_ids_1_padded += [pad_token_id_1] * padding_length_1_neg
                    neg_chunk_attention_mask_1 += [0] * padding_length_1_neg
                neg_chunk_ids_1_tensor = torch.tensor([neg_chunk_ids_1_padded], device=device)
                neg_chunk_attention_mask_1_tensor = torch.tensor([neg_chunk_attention_mask_1], device=device)

                neg_chunk_ids_2_padded = [tokenizer_2.bos_token_id] + neg_chunk_ids_2_slice + [tokenizer_2.eos_token_id]
                neg_chunk_attention_mask_2 = [1] * len(neg_chunk_ids_2_padded)
                padding_length_2_neg = max_length - len(neg_chunk_ids_2_padded)
                if padding_length_2_neg > 0:
                    neg_chunk_ids_2_padded += [pad_token_id_2] * padding_length_2_neg
                    neg_chunk_attention_mask_2 += [0] * padding_length_2_neg
                neg_chunk_ids_2_tensor = torch.tensor([neg_chunk_ids_2_padded], device=device)
                neg_chunk_attention_mask_2_tensor = torch.tensor([neg_chunk_attention_mask_2], device=device)

                neg_chunk_embeds_1 = text_encoder(
                    neg_chunk_ids_1_tensor,
                    attention_mask=neg_chunk_attention_mask_1_tensor,
                    output_hidden_states=True,
                ).hidden_states[-2]

                neg_chunk_output_2 = text_encoder_2(
                    neg_chunk_ids_2_tensor,
                    attention_mask=neg_chunk_attention_mask_2_tensor,
                    output_hidden_states=True,
                    return_dict=True,
                )
                neg_chunk_embeds_2 = neg_chunk_output_2.hidden_states[-2]

                if hasattr(neg_chunk_output_2, 'text_embeds'):
                    neg_pooled_embeds_2 = neg_chunk_output_2.text_embeds
                elif hasattr(neg_chunk_output_2, 'pooled_output'):
                    neg_pooled_embeds_2 = neg_chunk_output_2.pooled_output
                else:
                    neg_pooled_embeds_2 = neg_chunk_output_2.last_hidden_state[:, 0]

                seq_len_neg = max(neg_chunk_embeds_1.shape[1], neg_chunk_embeds_2.shape[1])
                if neg_chunk_embeds_1.shape[1] != seq_len_neg:
                    neg_chunk_embeds_1 = torch.nn.functional.pad(neg_chunk_embeds_1, (0, 0, 0, seq_len_neg - neg_chunk_embeds_1.shape[1]), value=0)
                if neg_chunk_embeds_2.shape[1] != seq_len_neg:
                    neg_chunk_embeds_2 = torch.nn.functional.pad(neg_chunk_embeds_2, (0, 0, 0, seq_len_neg - neg_chunk_embeds_2.shape[1]), value=0)

                chunk_combined_neg_embeds = torch.cat([neg_chunk_embeds_1, neg_chunk_embeds_2], dim=-1)

                if i == 0:
                    first_chunk_negative_pooled_prompt_embeds = neg_pooled_embeds_2

                if final_negative_prompt_embeds is None:
                    final_negative_prompt_embeds = chunk_combined_neg_embeds
                else:
                    final_negative_prompt_embeds = torch.cat([final_negative_prompt_embeds, chunk_combined_neg_embeds], dim=1)

            prompt_embeds = final_prompt_embeds
            negative_prompt_embeds = final_negative_prompt_embeds
            pooled_prompt_embeds = first_chunk_pooled_prompt_embeds
            negative_pooled_prompt_embeds = first_chunk_negative_pooled_prompt_embeds

            if negative_prompt == "" and first_chunk_negative_pooled_prompt_embeds is None:
                if pooled_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
                    _, _, _, empty_neg_pooled = pipe.encode_prompt(prompt="", negative_prompt="", device=device, num_images_per_prompt=1)
                    negative_pooled_prompt_embeds = empty_neg_pooled.to(dtype=pooled_prompt_embeds.dtype, device=device)

            if torch.isnan(prompt_embeds).any() or torch.isinf(prompt_embeds).any():
                prompt_embeds = torch.nan_to_num(prompt_embeds)
            if torch.isnan(negative_prompt_embeds).any() or torch.isinf(negative_prompt_embeds).any():
                negative_prompt_embeds = torch.nan_to_num(negative_prompt_embeds)
            if torch.isnan(pooled_prompt_embeds).any() or torch.isinf(pooled_prompt_embeds).any():
                pooled_prompt_embeds = torch.nan_to_num(pooled_prompt_embeds)
            if torch.isnan(negative_pooled_prompt_embeds).any() or torch.isinf(negative_pooled_prompt_embeds).any():
                negative_pooled_prompt_embeds = torch.nan_to_num(negative_pooled_prompt_embeds)

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    # ==========================================================================
    # Model & LoRA Downloads
    # ==========================================================================
    def _download_civitai_model(self, model_id: str) -> tuple[str, str]:
        civitai_id = model_id.split("civitai:")[1]
        model_dir = Path(f"{CIVITAI_MODELS_DIR}/{civitai_id}")
        model_dir.mkdir(exist_ok=True, parents=True)

        safetensors = list(model_dir.glob("*.safetensors"))
        if safetensors:
            return str(model_dir), safetensors[0].name

        token = os.environ.get("CIVITAI_TOKEN")
        url = f"https://civitai.com/api/download/models/{civitai_id}?token={token}"
        print(f"Downloading CivitAI model {civitai_id}...")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        if "Content-Disposition" in response.headers:
            filename = response.headers["Content-Disposition"].split("filename=")[1].strip('"')
        else:
            filename = f"model_{civitai_id}.safetensors"

        output_path = model_dir / filename
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)

        cache_volume.commit()
        print(f"Downloaded CivitAI model to {output_path}")
        return str(model_dir), filename

    def _download_civitai_lora(self, lora_id: str) -> str:
        civitai_id = lora_id.split("civitai:")[1]
        lora_dir = Path(f"{CIVITAI_LORAS_DIR}/{civitai_id}")
        lora_dir.mkdir(exist_ok=True, parents=True)

        safetensors = list(lora_dir.glob("*.safetensors"))
        if safetensors:
            return str(safetensors[0])

        token = os.environ.get("CIVITAI_TOKEN")
        url = f"https://civitai.com/api/download/models/{civitai_id}?token={token}"
        print(f"Downloading CivitAI LoRA {civitai_id}...")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        if "Content-Disposition" in response.headers:
            filename = response.headers["Content-Disposition"].split("filename=")[1].strip('"')
        else:
            filename = f"lora_{civitai_id}.safetensors"

        output_path = lora_dir / filename
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=512 * 1024):
                f.write(chunk)

        cache_volume.commit()
        print(f"Downloaded CivitAI LoRA to {output_path}")
        return str(output_path)

    def _download_hf_lora(self, lora_id: str) -> str:
        parts = lora_id.split("hf:")
        if len(parts) != 2:
            raise ValueError("HF LoRA must be in format 'hf:repo_id' or 'hf:repo_id/path'")

        repo_id_path = parts[1]
        if "/" in repo_id_path and not repo_id_path.endswith("/"):
            repo_id = repo_id_path.split("/")[0]
            file_path = "/".join(repo_id_path.split("/")[1:])
            lora_dir = Path(f"{HF_LORAS_DIR}/{repo_id}")
            lora_dir.mkdir(exist_ok=True, parents=True)

            target_file = lora_dir / file_path.split("/")[-1]
            if target_file.exists():
                return str(target_file)

            downloaded = hf_hub_download(repo_id=repo_id, filename=file_path, cache_dir=CACHE_DIR)
            cache_volume.commit()
            return downloaded
        else:
            repo_id = repo_id_path.strip("/")
            for filename in ["lora.safetensors", "pytorch_lora_weights.safetensors", "pytorch_lora_weights.bin"]:
                try:
                    downloaded = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=CACHE_DIR)
                    cache_volume.commit()
                    return downloaded
                except Exception:
                    pass

            files = list_repo_files(repo_id)
            for f in files:
                if f.endswith(".safetensors") and ("lora" in f.lower() or "weight" in f.lower()):
                    downloaded = hf_hub_download(repo_id=repo_id, filename=f, cache_dir=CACHE_DIR)
                    cache_volume.commit()
                    return downloaded

            raise ValueError(f"No LoRA safetensors file found in {repo_id}")

    def _download_lora(self, lora_spec: LoRASpec) -> str:
        model_id = lora_spec.model_id
        if model_id.startswith("civitai:"):
            return self._download_civitai_lora(model_id)
        elif model_id.startswith("hf:"):
            return self._download_hf_lora(model_id)
        else:
            raise ValueError(f"Unsupported LoRA prefix in '{model_id}'. Must start with 'civitai:' or 'hf:'")

    # ==========================================================================
    # Pipeline Management & SDPA Optimization
    # ==========================================================================
    def _manage_model_memory(self, model_key: str):
        self.model_last_accessed[model_key] = time.time()
        if len(self.loaded_models) > self.max_loaded_models:
            lru_key = min(self.model_last_accessed.items(), key=lambda x: x[1])[0]
            if lru_key != model_key and lru_key in self.loaded_models:
                print(f"Unloading least recently used model from VRAM: {lru_key}")
                try:
                    pipe = self.loaded_models[lru_key]
                    if hasattr(pipe, "unload_lora_weights"):
                        pipe.unload_lora_weights()
                    pipe.to("cpu")
                    del self.loaded_models[lru_key]
                    del self.model_last_accessed[lru_key]
                    if lru_key in self.active_loras:
                        del self.active_loras[lru_key]
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except Exception as e:
                    print(f"Error unloading model {lru_key}: {e}")

    def _load_pipeline(self, model_id: str, loras: Optional[List[LoRASpec]] = None):
        target_dtype = torch.float16
        base_model_key = model_id

        # Check if base model is already cached
        if base_model_key not in self.loaded_models:
            print(f"Loading SDXL base model into VRAM: {model_id}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            if model_id.startswith("civitai:"):
                model_path, model_filename = self._download_civitai_model(model_id)
                pipeline = StableDiffusionXLPipeline.from_single_file(
                    f"{model_path}/{model_filename}",
                    torch_dtype=target_dtype,
                    use_safetensors=True,
                )
            else:
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_id,
                    torch_dtype=target_dtype,
                    use_safetensors=True,
                    variant="fp16",
                )

            # Apply Ollin VAE fix for SDXL fp16 stability if custom model
            if model_id.startswith("civitai:") or model_id != DEFAULT_MODEL_ID:
                try:
                    vae = AutoencoderKL.from_pretrained(
                        "madebyollin/sdxl-vae-fp16-fix",
                        torch_dtype=target_dtype,
                    )
                    pipeline.vae = vae
                    print("Loaded madebyollin/sdxl-vae-fp16-fix VAE.")
                except Exception as e:
                    print(f"Using model default VAE (could not load fp16 fix: {e})")

            # Native PyTorch 2.x SDPA is standard in Diffusers 0.30+
            pipeline.to("cuda")
            self.loaded_models[base_model_key] = pipeline
            self.active_loras[base_model_key] = []
        else:
            pipeline = self.loaded_models[base_model_key]

        self._manage_model_memory(base_model_key)

        # LoRA lifecycle management via native Diffusers APIs
        current_active = self.active_loras.get(base_model_key, [])
        target_lora_ids = [l.model_id for l in loras] if loras else []

        # If current loaded adapters do not match requested, unload and re-apply
        if current_active != target_lora_ids:
            if current_active:
                print("Unloading previous LoRA adapters...")
                try:
                    pipeline.unload_lora_weights()
                    if hasattr(pipeline, "delete_adapters"):
                        for name in current_active:
                            try:
                                pipeline.delete_adapters(name)
                            except Exception:
                                pass
                except Exception as e:
                    print(f"Notice during LoRA unload: {e}")
                self.active_loras[base_model_key] = []

            if loras:
                adapter_names = []
                adapter_weights = []
                for idx, lora in enumerate(loras):
                    try:
                        lora_path = self._download_lora(lora)
                        adapter_name = f"adapter_{idx}"
                        pipeline.load_lora_weights(lora_path, adapter_name=adapter_name)
                        adapter_names.append(adapter_name)
                        adapter_weights.append(lora.weight)
                        print(f"Loaded LoRA {lora.model_id} as {adapter_name} (weight={lora.weight})")
                    except Exception as lora_err:
                        print(f"Error loading LoRA {lora.model_id}: {lora_err}")

                if adapter_names:
                    try:
                        pipeline.set_adapters(adapter_names, adapter_weights)
                        self.active_loras[base_model_key] = target_lora_ids
                        print(f"Successfully activated {len(adapter_names)} LoRA adapters.")
                    except Exception as set_err:
                        print(f"Error setting LoRA adapters: {set_err}")

        return pipeline

    # ==========================================================================
    # Sampler Configuration & FreeU
    # ==========================================================================
    def _configure_scheduler(self, pipe, scheduler_name: SchedulerType):
        config = pipe.scheduler.config
        if scheduler_name == "euler_ancestral":
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(config)
        elif scheduler_name == "dpmpp_2m_karras":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                config,
                algorithm_type="dpmsolver++",
                solver_order=2,
                use_karras_sigmas=True,
            )
        elif scheduler_name == "dpmpp_sde_karras":
            pipe.scheduler = DPMSolverSDEScheduler.from_config(
                config,
                use_karras_sigmas=True,
            )
        elif scheduler_name == "unipc":
            pipe.scheduler = UniPCMultistepScheduler.from_config(config)
        elif scheduler_name == "euler":
            pipe.scheduler = EulerDiscreteScheduler.from_config(config)
        elif scheduler_name == "ddim":
            pipe.scheduler = DDIMScheduler.from_config(config)
        else:
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(config)

    def _configure_freeu(self, pipe, freeu: Optional[FreeUSpec]):
        if freeu and freeu.enabled:
            print(f"Enabling FreeU (s1={freeu.s1}, s2={freeu.s2}, b1={freeu.b1}, b2={freeu.b2})")
            pipe.enable_freeu(s1=freeu.s1, s2=freeu.s2, b1=freeu.b1, b2=freeu.b2)
        else:
            if hasattr(pipe, "disable_freeu"):
                pipe.disable_freeu()

    # ==========================================================================
    # Metadata Embedding in PNG
    # ==========================================================================
    def _create_png_with_metadata(self, image: Image.Image, metadata_dict: Dict[str, Any]) -> bytes:
        """Embeds A1111-compatible and JSON metadata into PNG tEXt chunks."""
        pnginfo = PngInfo()

        prompt = metadata_dict.get("prompt", "")
        negative_prompt = metadata_dict.get("negative_prompt", "")
        steps = metadata_dict.get("steps", DEFAULT_STEPS)
        sampler = metadata_dict.get("sampler", DEFAULT_SCHEDULER)
        cfg = metadata_dict.get("guidance_scale", DEFAULT_GUIDANCE_SCALE)
        seed = metadata_dict.get("seed", 0)
        size = f"{metadata_dict.get('width', DEFAULT_WIDTH)}x{metadata_dict.get('height', DEFAULT_HEIGHT)}"
        model = metadata_dict.get("model_id", DEFAULT_MODEL_ID)

        lora_str = ""
        if metadata_dict.get("loras"):
            lora_str = ", LoRAs: " + ", ".join([f"{l['model_id']}:{l['weight']}" for l in metadata_dict["loras"]])

        a1111_params = (
            f"{prompt}\n"
            f"Negative prompt: {negative_prompt}\n"
            f"Steps: {steps}, Sampler: {sampler}, CFG scale: {cfg}, Seed: {seed}, Size: {size}, Model: {model}{lora_str}"
        )

        pnginfo.add_text("parameters", a1111_params)
        pnginfo.add_text("prompt", prompt)
        pnginfo.add_text("negative_prompt", negative_prompt)
        pnginfo.add_text("sdxl_metadata", json.dumps(metadata_dict))

        buf = io.BytesIO()
        image.save(buf, format="PNG", pnginfo=pnginfo)
        return buf.getvalue()

    # ==========================================================================
    # Core Execution Method
    # ==========================================================================
    @modal.method()
    def run(
        self,
        prompt: str,
        batch_size: int = 1,
        negative_prompt: str = "",
        seed: Optional[int] = None,
        model_id: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        loras: Optional[List[Dict[str, Any]]] = None,
        clip_skip: Optional[int] = None,
        scheduler: Optional[SchedulerType] = None,
        freeu: Optional[Dict[str, Any]] = None,
    ) -> List[bytes]:
        actual_model_id = model_id if model_id else DEFAULT_MODEL_ID
        lora_specs = [LoRASpec(**l) for l in loras] if loras else None
        freeu_spec = FreeUSpec(**freeu) if freeu else None

        pipe = self._load_pipeline(actual_model_id, lora_specs)

        # Scheduler & FreeU
        self._configure_scheduler(pipe, scheduler or DEFAULT_SCHEDULER)
        self._configure_freeu(pipe, freeu_spec)

        # Dimension validation (multiples of 8)
        w = max(512, min(2048, ((width or DEFAULT_WIDTH) // 8) * 8))
        h = max(512, min(2048, ((height or DEFAULT_HEIGHT) // 8) * 8))
        steps = num_inference_steps or DEFAULT_STEPS
        gs = guidance_scale if guidance_scale is not None else DEFAULT_GUIDANCE_SCALE

        actual_seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        generator = torch.Generator(device="cuda").manual_seed(actual_seed)

        # CLIP skip handling
        if not hasattr(pipe, "_original_text_encoder_config_dict"):
            pipe._original_text_encoder_config_dict = pipe.text_encoder.config.to_dict()
        if hasattr(pipe, "text_encoder_2") and not hasattr(pipe, "_original_text_encoder_2_config_dict"):
            pipe._original_text_encoder_2_config_dict = pipe.text_encoder_2.config.to_dict()

        pipe.text_encoder.config = CLIPTextConfig.from_dict(pipe._original_text_encoder_config_dict.copy())
        if hasattr(pipe, "text_encoder_2"):
            pipe.text_encoder_2.config = CLIPTextConfig.from_dict(pipe._original_text_encoder_2_config_dict.copy())

        if clip_skip and clip_skip > 1:
            skip = clip_skip - 1
            if hasattr(pipe, "text_encoder") and skip < pipe.text_encoder.config.num_hidden_layers:
                pipe.text_encoder.config.num_hidden_layers -= skip
            if hasattr(pipe, "text_encoder_2") and skip < pipe.text_encoder_2.config.num_hidden_layers:
                pipe.text_encoder_2.config.num_hidden_layers -= skip

        # A1111 Chunked Encoding
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self._encode_prompt_chunked(
            pipe=pipe,
            prompt=prompt,
            negative_prompt=negative_prompt,
            device=pipe.device,
            batch_size=1,
        )

        if batch_size > 1:
            prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)
            negative_prompt_embeds = negative_prompt_embeds.repeat(batch_size, 1, 1)
            pooled_prompt_embeds = pooled_prompt_embeds.repeat(batch_size, 1)
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(batch_size, 1)

        images = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_images_per_prompt=1,
            num_inference_steps=steps,
            guidance_scale=gs,
            width=w,
            height=h,
            generator=generator,
        ).images

        del prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": actual_seed,
            "model_id": actual_model_id,
            "width": w,
            "height": h,
            "steps": steps,
            "guidance_scale": gs,
            "sampler": scheduler or DEFAULT_SCHEDULER,
            "clip_skip": clip_skip,
            "loras": [l.model_dump() for l in lora_specs] if lora_specs else [],
            "freeu": freeu_spec.model_dump() if freeu_spec else None,
        }

        output_bytes = []
        for idx, img in enumerate(images):
            img_meta = metadata.copy()
            if batch_size > 1:
                img_meta["batch_index"] = idx
            output_bytes.append(self._create_png_with_metadata(img, img_meta))

        return output_bytes

    # ==========================================================================
    # Modern FastAPI Web Endpoint
    # ==========================================================================
    @modal.fastapi_endpoint(method="POST", docs=True)
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Typed FastAPI POST endpoint for high-throughput batch generation."""
        start_time = time.time()
        all_images_bytes = []

        base_seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)

        for batch_idx in range(request.batch_count):
            batch_seed = base_seed + batch_idx
            batch_loras = [l.model_dump() for l in request.loras] if request.loras else None
            batch_freeu = request.freeu.model_dump() if request.freeu else None

            images = self.run.local(
                prompt=request.prompt,
                batch_size=request.batch_size,
                negative_prompt=request.negative_prompt,
                seed=batch_seed,
                model_id=request.model_id,
                width=request.width,
                height=request.height,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                loras=batch_loras,
                clip_skip=request.clip_skip,
                scheduler=request.scheduler,
                freeu=batch_freeu,
            )
            all_images_bytes.extend(images)

        import base64
        b64_images = [base64.b64encode(img).decode("utf-8") for img in all_images_bytes]
        duration = round(time.time() - start_time, 2)

        return GenerationResponse(
            images=b64_images,
            parameters=request.model_dump(),
            duration_seconds=duration,
        )

    @modal.fastapi_endpoint(method="GET", docs=True)
    def health(self) -> Dict[str, Any]:
        """Health check endpoint providing VRAM status and active models."""
        return {
            "status": "healthy",
            "gpu_memory": get_gpu_memory_info(),
            "cached_models": list(self.loaded_models.keys()),
        }


# ==============================================================================
# Local CLI Entrypoint
# ==============================================================================

@app.local_entrypoint()
def entrypoint(
    prompt: str = "A photorealistic landscape, breathtaking vista, 8k, highly detailed",
    negative_prompt: str = "cartoon, animation, drawing, low quality, blurry, nsfw",
    batch_size: int = 1,
    samples: int = 1,
    seed: Optional[int] = None,
    model_id: Optional[str] = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    steps: int = DEFAULT_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    scheduler: SchedulerType = DEFAULT_SCHEDULER,
    loras: Optional[str] = None,
):
    print(f"Executing local entrypoint for prompt: {prompt}")
    lora_list = json.loads(loras) if loras else None

    inference = Inference(load_default_model=False)
    output_dir = Path("./generated_images")
    output_dir.mkdir(exist_ok=True, parents=True)

    for i in range(samples):
        current_seed = seed + i if seed is not None else None
        start = time.time()
        images = inference.run.remote(
            prompt=prompt,
            batch_size=batch_size,
            negative_prompt=negative_prompt,
            seed=current_seed,
            model_id=model_id,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            loras=lora_list,
            scheduler=scheduler,
        )
        duration = time.time() - start
        print(f"Sample {i+1}/{samples} generated {len(images)} images in {duration:.2f}s")

        for idx, img_bytes in enumerate(images):
            filepath = output_dir / f"cli_{int(time.time())}_{i}_{idx}.png"
            filepath.write_bytes(img_bytes)
            print(f"Saved: {filepath}")
