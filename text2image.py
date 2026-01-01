import io
import os
import json
import random
import time
import requests
import re # Import re for filename parsing
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Literal, TYPE_CHECKING

import modal

# Import torch for type hints only
if TYPE_CHECKING:
    import torch

MINUTES = 60

app = modal.App("text2image")

CACHE_DIR = "/cache"
CIVITAI_MODELS_DIR = "/cache/civitai"
CIVITAI_LORAS_DIR = "/cache/civitai/loras"
HF_LORAS_DIR = "/cache/hf/loras"

# Define available scheduler types
SchedulerType = Literal[
    "euler_ancestral",
    "dpmpp_2m_karras"
]

# Get GPU type from environment variable or use L4 as default
GPU_TYPE = os.environ.get("GPU_TYPE", "L4")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "accelerate==0.33.0",
        "diffusers==0.31.0",
        "fastapi[standard]==0.115.4",
        "huggingface-hub[hf_transfer]==0.25.2",
        "sentencepiece==0.2.0",
        "torch==2.5.1",
        "torchvision==0.20.1",
        "transformers~=4.44.0",
        "requests>=2.28.0",
        "safetensors>=0.4.1",
        "peft>=0.8.2",  # Required for LoRA support
        "safetensors", # Explicitly install safetensors
        "xformers>=0.0.21",  # For memory-efficient attention
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",  # faster downloads
            "HF_HUB_CACHE": CACHE_DIR,
        }
    )
)

civitai_secret = modal.Secret.from_name("civitai-token")

with image.imports():
    import diffusers
    import torch
    from fastapi import Response
    from huggingface_hub import hf_hub_download
    from diffusers.models.attention_processor import AttnProcessor2_0
    from diffusers.schedulers import (
        DDIMScheduler,
        DDPMScheduler,
        PNDMScheduler,
        LMSDiscreteScheduler,
        EulerDiscreteScheduler,
        EulerAncestralDiscreteScheduler,
        DPMSolverMultistepScheduler,
        DPMSolverSDEScheduler,
        UniPCMultistepScheduler,
        DEISMultistepScheduler,
    )
    from diffusers import AutoencoderKL  # Add import for VAE
    import peft  # Import PEFT for LoRA support
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer, CLIPTextConfig # Import necessary components
    import safetensors.torch # Import safetensors load
    
    # Add GPU memory tracking function
    def get_gpu_memory_info():
        """Get GPU memory usage information"""
        if torch.cuda.is_available():
            t = torch.cuda.get_device_properties(0).total_memory
            r = torch.cuda.memory_reserved(0)
            a = torch.cuda.memory_allocated(0)
            f = t - (r + a)  # free inside reserved
            return {
                "total": t / (1024**2),  # Convert to MB
                "reserved": r / (1024**2),
                "allocated": a / (1024**2),
                "free": f / (1024**2)
            }
        return {"error": "CUDA not available"}


# Default SDXL model
DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_REFINER_MODEL_ID = "stabilityai/stable-diffusion-xl-refiner-1.0"

# Default generation parameters
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_STEPS = 30  # Default steps for SDXL
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_SCHEDULER = "euler_ancestral"  # Default scheduler

# Max token length for CLIP models (including start/end tokens)
MAX_TOKEN_LENGTH = 77

cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)


def parse_loras(loras_json: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    """Parses a JSON string of LoRAs into a list of dictionaries."""
    if not loras_json:
        return None
    try:
        lora_list = json.loads(loras_json)
        if not isinstance(lora_list, list):
            lora_list = [lora_list]
        return lora_list
    except Exception as e:
        print(f"Error parsing LoRAs: {str(e)}")
        return None


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    timeout=10 * MINUTES,
    scaledown_window=5 * MINUTES,  # Scale down after 5 minutes of inactivity to save costs
    volumes={CACHE_DIR: cache_volume},
    secrets=[civitai_secret],
)
class Inference:
    # Use modal.parameter with type annotation
    load_default_model: bool = modal.parameter(default=False)
    
    # Track model access times for cache management
    model_last_accessed = {}
    max_loaded_models = 2  # Reduced from 3 to 2 to minimize memory usage
    loaded_loras = set()  # Track loaded LoRAs to avoid reloading

    # List of available schedulers with display names
    AVAILABLE_SCHEDULERS = {
        "euler_ancestral": "Euler Ancestral",
        "dpmpp_2m_karras": "DPM++ 2M Karras"
    }

    @modal.enter()
    def setup(self):
        # Initialize internal state
        self.loaded_models = {}
        self.loaded_loras = set()  # Track loaded LoRAs to avoid reloading
        
        # Create directories
        os.makedirs(CIVITAI_MODELS_DIR, exist_ok=True)
        os.makedirs(CIVITAI_LORAS_DIR, exist_ok=True)
        os.makedirs(HF_LORAS_DIR, exist_ok=True)
        
        # Print initial GPU memory state
        print(f"Initial GPU memory state: {get_gpu_memory_info()}")
        
        # Only load default model if explicitly requested
        if self.load_default_model:
            print("Pre-loading default SDXL model...")
            try:
                self._load_pipeline(DEFAULT_MODEL_ID)
                print("Default SDXL model loaded successfully")
                print(f"GPU memory after loading default model: {get_gpu_memory_info()}")
            except Exception as e:
                print(f"Warning: Failed to pre-load default model: {str(e)}")
                # Don't fail startup if default model can't load

    def _encode_prompt_chunked(
        self,
        pipe,
        prompt: str,
        negative_prompt: str,
        device: "torch.device", # Use string literal for type hint
        batch_size: int,
        max_length: int = MAX_TOKEN_LENGTH
    ):
        """
        Encodes positive and negative prompts using token-based chunking and concatenates embeddings.
        Mimics Automatic1111's approach to handling long prompts by concatenating token embeddings.
        Handles dual text encoders for SDXL and batching.
        """
        # Get the tokenizers and text encoders
        tokenizer, tokenizer_2 = pipe.tokenizer, pipe.tokenizer_2
        text_encoder, text_encoder_2 = pipe.text_encoder, pipe.text_encoder_2
        
        # Get the hidden size and projection dim for output shapes
        hidden_size_1 = text_encoder.config.hidden_size
        hidden_size_2 = text_encoder_2.config.hidden_size
        pooled_size_2 = text_encoder_2.config.projection_dim
        combined_hidden_size = hidden_size_1 + hidden_size_2

        # --- Tokenize Original Prompts (without truncation/padding initially) ---
        # Get raw token IDs for the full prompts
        # Use add_special_tokens=False to get just the prompt tokens
        pos_input_ids_1_raw = tokenizer(prompt, add_special_tokens=False, truncation=False, return_tensors="pt").input_ids[0].tolist()
        pos_input_ids_2_raw = tokenizer_2(prompt, add_special_tokens=False, truncation=False, return_tensors="pt").input_ids[0].tolist()
        neg_input_ids_1_raw = tokenizer(negative_prompt or "", add_special_tokens=False, truncation=False, return_tensors="pt").input_ids[0].tolist() # Handle empty string
        neg_input_ids_2_raw = tokenizer_2(negative_prompt or "", add_special_tokens=False, truncation=False, return_tensors="pt").input_ids[0].tolist() # Handle empty string


        # Determine total max length across all tokenizations
        # This is the effective length of the prompt content needing to be covered by chunks
        total_max_len = max(
            len(pos_input_ids_1_raw),
            len(pos_input_ids_2_raw),
            len(neg_input_ids_1_raw),
            len(neg_input_ids_2_raw),
        )

        # Account for BOS/EOS tokens in each chunk
        # A standard chunk includes [BOS]...[EOS]. The prompt content fits in MAX_TOKEN_LENGTH - 2 slots.
        # Use a safer calculation for num special tokens if available, otherwise default to 2
        num_special_tokens = tokenizer.num_special_tokens_to_add(pair=False) if hasattr(tokenizer, 'num_special_tokens_to_add') else 2
        effective_chunk_size = max_length - num_special_tokens

        if effective_chunk_size <= 0:
             raise ValueError(f"Effective chunk size is too small or zero: {effective_chunk_size}. Check MAX_TOKEN_LENGTH and tokenizer special tokens.")

        # Calculate number of chunks needed based on the total length
        num_chunks = (total_max_len + effective_chunk_size - 1) // effective_chunk_size


        # --- Handle Non-Chunking Case ---
        # If total length fits within one effective chunk (e.g., <= 75 prompt tokens),
        # then the standard pipeline encode_prompt is sufficient and handles batching.
        if num_chunks <= 1:
            print("Prompt fits within max effective token length, using standard encoding.")
            # Use the pipeline's internal encoding method which handles batching and dual encoders
            # Important: We're not passing batch_size, as that will be handled separately
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                device=device,
                num_images_per_prompt=1,  # Changed from batch_size to 1
                do_classifier_free_guidance=True, # Assume CFG
                # Add other relevant args like clean_caption if needed
            )
            return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds


        # --- Handle Token-Based Chunking Case (Mimics A1111) ---
        print(f"Prompt requires token-based chunking into {num_chunks} chunks.")

        # --- NEW: Variables to store embeddings for concatenation ---
        final_prompt_embeds = None
        final_negative_prompt_embeds = None
        
        # --- NEW: Variables to store first chunk's pooled output ---
        first_chunk_pooled_prompt_embeds = None
        first_chunk_negative_pooled_prompt_embeds = None

        # Get pad token IDs safely (EOS is often reused as PAD for CLIP)
        pad_token_id_1 = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        pad_token_id_2 = tokenizer_2.pad_token_id if tokenizer_2.pad_token_id is not None else tokenizer_2.eos_token_id

        # Ensure encoders are in evaluation mode
        text_encoder.eval()
        text_encoder_2.eval()

        with torch.no_grad():
            for i in range(num_chunks):
                # Calculate slice indices for the original raw token ID lists
                # Include overlap between chunks (5 tokens) except for the first chunk
                overlap_tokens = 5 if i > 0 else 0
                start_idx = max(0, i * effective_chunk_size - overlap_tokens)
                end_idx = min(start_idx + effective_chunk_size, total_max_len) # Slice up to total_max_len

                # --- Process Positive Prompt Chunk ---
                # Get the slice of raw tokens for this chunk
                pos_chunk_ids_1_slice = pos_input_ids_1_raw[start_idx:end_idx]
                pos_chunk_ids_2_slice = pos_input_ids_2_raw[start_idx:end_idx]

                # Add special tokens ([BOS]...[EOS]) and pad to max_length (77) for Encoder 1
                pos_chunk_ids_1_padded = [tokenizer.bos_token_id] + pos_chunk_ids_1_slice + [tokenizer.eos_token_id]
                pos_chunk_attention_mask_1 = [1] * len(pos_chunk_ids_1_padded)
                padding_length_1 = max_length - len(pos_chunk_ids_1_padded)
                if padding_length_1 > 0:
                    pos_chunk_ids_1_padded += [pad_token_id_1] * padding_length_1
                    pos_chunk_attention_mask_1 += [0] * padding_length_1
                pos_chunk_ids_1_tensor = torch.tensor([pos_chunk_ids_1_padded], device=device)
                pos_chunk_attention_mask_1_tensor = torch.tensor([pos_chunk_attention_mask_1], device=device)


                # Add special tokens ([BOS]...[EOS]) and pad to max_length (77) for Encoder 2
                pos_chunk_ids_2_padded = [tokenizer_2.bos_token_id] + pos_chunk_ids_2_slice + [tokenizer_2.eos_token_id]
                pos_chunk_attention_mask_2 = [1] * len(pos_chunk_ids_2_padded)
                padding_length_2 = max_length - len(pos_chunk_ids_2_padded)
                if padding_length_2 > 0:
                    pos_chunk_ids_2_padded += [pad_token_id_2] * padding_length_2
                    pos_chunk_attention_mask_2 += [0] * padding_length_2
                pos_chunk_ids_2_tensor = torch.tensor([pos_chunk_ids_2_padded], device=device)
                pos_chunk_attention_mask_2_tensor = torch.tensor([pos_chunk_attention_mask_2], device=device)


                # Encode chunk with Text Encoder 1
                pos_chunk_embeds_1 = text_encoder(
                    pos_chunk_ids_1_tensor,
                    attention_mask=pos_chunk_attention_mask_1_tensor,
                    output_hidden_states=True,
                ).hidden_states[-2] # Second to last layer

                # Encode chunk with Text Encoder 2
                pos_chunk_output_2 = text_encoder_2(
                    pos_chunk_ids_2_tensor,
                    attention_mask=pos_chunk_attention_mask_2_tensor,
                    output_hidden_states=True,
                    return_dict=True,
                )
                pos_chunk_embeds_2 = pos_chunk_output_2.hidden_states[-2] # Second to last layer
                
                # Access text_embeds for the pooled output from CLIPTextModelWithProjection
                if hasattr(pos_chunk_output_2, 'text_embeds'):
                    # CLIPTextModelWithProjection provides text_embeds
                    pos_pooled_embeds_2 = pos_chunk_output_2.text_embeds
                elif hasattr(pos_chunk_output_2, 'pooled_output'):
                    # Legacy access for backward compatibility
                    pos_pooled_embeds_2 = pos_chunk_output_2.pooled_output
                else:
                    # Fallback - use the last hidden state's first token ([CLS]/[BOS]) as pooled representation
                    print("Warning: No pooled embeddings found, falling back to first token of last hidden state")
                    pos_pooled_embeds_2 = pos_chunk_output_2.last_hidden_state[:, 0]


                # Concatenate hidden states from both encoders for this chunk
                # Ensure tensors have the same sequence length before concatenating
                seq_len = max(pos_chunk_embeds_1.shape[1], pos_chunk_embeds_2.shape[1])
                if pos_chunk_embeds_1.shape[1] != seq_len:
                    pos_chunk_embeds_1 = torch.nn.functional.pad(pos_chunk_embeds_1, (0, 0, 0, seq_len - pos_chunk_embeds_1.shape[1]), value=0)
                if pos_chunk_embeds_2.shape[1] != seq_len:
                    pos_chunk_embeds_2 = torch.nn.functional.pad(pos_chunk_embeds_2, (0, 0, 0, seq_len - pos_chunk_embeds_2.shape[1]), value=0)

                chunk_combined_pos_embeds = torch.cat([pos_chunk_embeds_1, pos_chunk_embeds_2], dim=-1)

                # --- NEW: Store first chunk's pooled embeddings ---
                if i == 0:
                    first_chunk_pooled_prompt_embeds = pos_pooled_embeds_2

                # --- NEW: Concatenate embeddings ---
                if final_prompt_embeds is None:  # First chunk
                    final_prompt_embeds = chunk_combined_pos_embeds
                else:  # Subsequent chunks
                    final_prompt_embeds = torch.cat([final_prompt_embeds, chunk_combined_pos_embeds], dim=1)

                # --- Process Negative Prompt Chunk ---
                # Get the slice of raw tokens for this chunk
                neg_chunk_ids_1_slice = neg_input_ids_1_raw[start_idx:end_idx]
                neg_chunk_ids_2_slice = neg_input_ids_2_raw[start_idx:end_idx]

                # Add special tokens ([BOS]...[EOS]) and pad to max_length (77) for Encoder 1
                neg_chunk_ids_1_padded = [tokenizer.bos_token_id] + neg_chunk_ids_1_slice + [tokenizer.eos_token_id]
                neg_chunk_attention_mask_1 = [1] * len(neg_chunk_ids_1_padded)
                padding_length_1_neg = max_length - len(neg_chunk_ids_1_padded)
                if padding_length_1_neg > 0:
                    neg_chunk_ids_1_padded += [pad_token_id_1] * padding_length_1_neg
                    neg_chunk_attention_mask_1 += [0] * padding_length_1_neg
                neg_chunk_ids_1_tensor = torch.tensor([neg_chunk_ids_1_padded], device=device)
                neg_chunk_attention_mask_1_tensor = torch.tensor([neg_chunk_attention_mask_1], device=device)

                # Add special tokens ([BOS]...[EOS]) and pad to max_length (77) for Encoder 2
                neg_chunk_ids_2_padded = [tokenizer_2.bos_token_id] + neg_chunk_ids_2_slice + [tokenizer_2.eos_token_id]
                neg_chunk_attention_mask_2 = [1] * len(neg_chunk_ids_2_padded)
                padding_length_2_neg = max_length - len(neg_chunk_ids_2_padded)
                if padding_length_2_neg > 0:
                    neg_chunk_ids_2_padded += [pad_token_id_2] * padding_length_2_neg
                    neg_chunk_attention_mask_2 += [0] * padding_length_2_neg
                neg_chunk_ids_2_tensor = torch.tensor([neg_chunk_ids_2_padded], device=device)
                neg_chunk_attention_mask_2_tensor = torch.tensor([neg_chunk_attention_mask_2], device=device)


                # Encode negative chunk with Text Encoder 1
                neg_chunk_embeds_1 = text_encoder(
                    neg_chunk_ids_1_tensor,
                    attention_mask=neg_chunk_attention_mask_1_tensor,
                    output_hidden_states=True,
                ).hidden_states[-2]

                # Encode negative chunk with Text Encoder 2
                neg_chunk_output_2 = text_encoder_2(
                    neg_chunk_ids_2_tensor,
                    attention_mask=neg_chunk_attention_mask_2_tensor,
                    output_hidden_states=True,
                    return_dict=True,
                )
                neg_chunk_embeds_2 = neg_chunk_output_2.hidden_states[-2]
                
                # Access text_embeds for the pooled output from CLIPTextModelWithProjection
                if hasattr(neg_chunk_output_2, 'text_embeds'):
                    # CLIPTextModelWithProjection provides text_embeds
                    neg_pooled_embeds_2 = neg_chunk_output_2.text_embeds
                elif hasattr(neg_chunk_output_2, 'pooled_output'):
                    # Legacy access for backward compatibility
                    neg_pooled_embeds_2 = neg_chunk_output_2.pooled_output
                else:
                    # Fallback - use the last hidden state's first token ([CLS]/[BOS]) as pooled representation
                    print("Warning: No pooled embeddings found for negative prompt, falling back to first token of last hidden state")
                    neg_pooled_embeds_2 = neg_chunk_output_2.last_hidden_state[:, 0]

                # Concatenate negative hidden states
                # Ensure tensors have the same sequence length before concatenating
                seq_len_neg = max(neg_chunk_embeds_1.shape[1], neg_chunk_embeds_2.shape[1])
                if neg_chunk_embeds_1.shape[1] != seq_len_neg:
                    neg_chunk_embeds_1 = torch.nn.functional.pad(neg_chunk_embeds_1, (0, 0, 0, seq_len_neg - neg_chunk_embeds_1.shape[1]), value=0)
                if neg_chunk_embeds_2.shape[1] != seq_len_neg:
                    neg_chunk_embeds_2 = torch.nn.functional.pad(neg_chunk_embeds_2, (0, 0, 0, seq_len_neg - neg_chunk_embeds_2.shape[1]), value=0)

                chunk_combined_neg_embeds = torch.cat([neg_chunk_embeds_1, neg_chunk_embeds_2], dim=-1)

                # --- NEW: Store first chunk's pooled negative embeddings ---
                if i == 0:
                    first_chunk_negative_pooled_prompt_embeds = neg_pooled_embeds_2

                # --- NEW: Concatenate negative embeddings ---
                if final_negative_prompt_embeds is None:  # First chunk
                    final_negative_prompt_embeds = chunk_combined_neg_embeds
                else:  # Subsequent chunks
                    final_negative_prompt_embeds = torch.cat([final_negative_prompt_embeds, chunk_combined_neg_embeds], dim=1)

            # --- MODIFIED: Use the concatenated embeddings directly ---
            prompt_embeds = final_prompt_embeds
            negative_prompt_embeds = final_negative_prompt_embeds
            
            # --- MODIFIED: Use first chunk's pooled embeddings ---
            pooled_prompt_embeds = first_chunk_pooled_prompt_embeds
            negative_pooled_prompt_embeds = first_chunk_negative_pooled_prompt_embeds
            
            # Handle case where negative prompt might be empty initially
            if negative_prompt == "" and first_chunk_negative_pooled_prompt_embeds is None:
                # Create zero pooled embeds if negative prompt was empty
                if pooled_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
                    _, _, _, empty_neg_pooled = pipe.encode_prompt(prompt="", negative_prompt="", device=device, num_images_per_prompt=1)
                    negative_pooled_prompt_embeds = empty_neg_pooled.to(dtype=pooled_prompt_embeds.dtype, device=device)

            # Check for NaN or Inf values before returning
            if torch.isnan(prompt_embeds).any() or torch.isinf(prompt_embeds).any():
                 print("WARNING: NaN or Inf values detected in final positive embeddings. Replacing with zeros.")
                 prompt_embeds = torch.nan_to_num(prompt_embeds)

            if torch.isnan(negative_prompt_embeds).any() or torch.isinf(negative_prompt_embeds).any():
                 print("WARNING: NaN or Inf values detected in final negative embeddings. Replacing with zeros.")
                 negative_prompt_embeds = torch.nan_to_num(negative_prompt_embeds)

            if torch.isnan(pooled_prompt_embeds).any() or torch.isinf(pooled_prompt_embeds).any():
                 print("WARNING: NaN or Inf values detected in final pooled positive embeddings. Replacing with zeros.")
                 pooled_prompt_embeds = torch.nan_to_num(pooled_prompt_embeds)

            if torch.isnan(negative_pooled_prompt_embeds).any() or torch.isinf(negative_pooled_prompt_embeds).any():
                 print("WARNING: NaN or Inf values detected in final pooled negative embeddings. Replacing with zeros.")
                 negative_pooled_prompt_embeds = torch.nan_to_num(negative_pooled_prompt_embeds)

        # REMOVED: Batch size expansion happens in the run method instead
        # No longer repeating embeddings for batch_size here
        
        print(f"Completed chunked encoding of prompt with {num_chunks} chunks.")
        print(f"Final embeds shape: {prompt_embeds.shape}, pooled shape: {pooled_prompt_embeds.shape}")
        print(f"Negative embeds shape: {negative_prompt_embeds.shape}, negative pooled shape: {negative_pooled_prompt_embeds.shape}")

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds


    # REMOVED: _split_prompt_into_chunks function is no longer needed for token-based chunking


    def _download_civitai_model(self, model_id):
        """
        Download a model from CivitAI using the provided model ID
        model_id format: civitai:MODEL_ID (e.g., civitai:135867)
        """
        # Extract the numeric model ID from the prefixed string
        if not model_id.startswith("civitai:"):
            raise ValueError("CivitAI model IDs must start with 'civitai:'")

        civitai_id = model_id.split("civitai:")[1]

        # Create directory for this specific model
        model_dir = Path(f"{CIVITAI_MODELS_DIR}/{civitai_id}")
        model_dir.mkdir(exist_ok=True, parents=True)

        # Check if model is already downloaded
        safetensor_files = list(model_dir.glob("*.safetensors"))
        if safetensor_files:
            print(f"CivitAI model {civitai_id} already downloaded")
            return str(model_dir), safetensor_files[0].name

        # Construct download URL with token
        token = os.environ.get("CIVITAI_TOKEN")
        download_url = f"https://civitai.com/api/download/models/{civitai_id}?token={token}"
        
        print(f"Downloading CivitAI model from {download_url}")
        
        # Download the file
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        
        # Get filename from content-disposition header
        if "Content-Disposition" in response.headers:
            content_disposition = response.headers["Content-Disposition"]
            filename = content_disposition.split("filename=")[1].strip('"')
        else:
            # Default filename if header not present
            filename = f"model_{civitai_id}.safetensors"
            
        output_path = model_dir / filename
        
        # Save file
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        print(f"Downloaded CivitAI model to {output_path}")
        return str(model_dir), filename
        
    def _download_civitai_lora(self, lora_id):
        """Download a LoRA from CivitAI"""
        # Extract the numeric ID
        civitai_id = lora_id.split("civitai:")[1]
        
        # Create directory for this specific lora
        lora_dir = Path(f"{CIVITAI_LORAS_DIR}/{civitai_id}")
        lora_dir.mkdir(exist_ok=True, parents=True)
        
        # Check if LoRA is already downloaded
        safetensor_files = list(lora_dir.glob("*.safetensors"))
        if safetensor_files:
            print(f"CivitAI LoRA {civitai_id} already downloaded")
            return str(safetensor_files[0])
            
        # Construct download URL with token
        token = os.environ.get("CIVITAI_TOKEN")
        download_url = f"https://civitai.com/api/download/models/{civitai_id}?token={token}"
        
        print(f"Downloading CivitAI LoRA from {download_url}")
        
        # Download the file
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        
        # Get filename from content-disposition header
        if "Content-Disposition" in response.headers:
            content_disposition = response.headers["Content-Disposition"]
            filename = content_disposition.split("filename=")[1].strip('"')
        else:
            # Default filename if header not present
            filename = f"lora_{civitai_id}.safetensors"
            
        output_path = lora_dir / filename
        
        # Save file
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        print(f"Downloaded CivitAI LoRA to {output_path}")
        return str(output_path)

    def _download_hf_lora(self, lora_id):
        """Download a LoRA from Hugging Face"""
        # Extract the repo_id and filename
        parts = lora_id.split("hf:")
        if len(parts) != 2:
            raise ValueError("Hugging Face LoRA IDs must be in format 'hf:repo_id/path'")
            
        repo_id_path = parts[1]
        
        # Check if a specific file is specified
        if "/" in repo_id_path and not repo_id_path.endswith("/"):
            repo_id = repo_id_path.split("/")[0]
            file_path = "/".join(repo_id_path.split("/")[1:])
            
            # Create directory for this specific lora
            lora_dir = Path(f"{HF_LORAS_DIR}/{repo_id}")
            lora_dir.mkdir(exist_ok=True, parents=True)
            
            # Check if already downloaded
            target_file = lora_dir / file_path.split("/")[-1]
            if target_file.exists():
                print(f"HF LoRA {repo_id_path} already downloaded")
                return str(target_file)
                
            # Download specific file
            print(f"Downloading Hugging Face LoRA from {repo_id_path}")
            file_path = hf_hub_download(repo_id=repo_id, filename=file_path, cache_dir=CACHE_DIR)
            return file_path
        else:
            # If no specific file, try to find one with 'lora' in the name in the root
            repo_id = repo_id_path.strip("/")
            try:
                # Find a file that is likely a LoRA
                print(f"Searching for a LoRA file in {repo_id}")
                for filename in ["lora.safetensors", "pytorch_lora_weights.safetensors", "pytorch_lora_weights.bin"]:
                    try:
                        file_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=CACHE_DIR)
                        print(f"Downloaded {filename} from {repo_id}")
                        return file_path
                    except:
                        pass
                        
                # If no known filename matched, list files and try to find one
                from huggingface_hub import list_repo_files
                files = list_repo_files(repo_id)
                
                for file in files:
                    if file.endswith(".safetensors") and ("lora" in file.lower() or "weight" in file.lower()):
                        file_path = hf_hub_download(repo_id=repo_id, filename=file, cache_dir=CACHE_DIR)
                        print(f"Found and downloaded {file} from {repo_id}")
                        return file_path
                        
                raise ValueError(f"Could not find a LoRA file in {repo_id}")
            except Exception as e:
                raise ValueError(f"Error downloading LoRA from Hugging Face: {str(e)}")

    def _download_lora(self, lora_spec):
        """Download a LoRA from either CivitAI or Hugging Face"""
        model_id = lora_spec.get("model_id", "")
        
        if not model_id:
            raise ValueError("LoRA model_id is required")
            
        if model_id.startswith("civitai:"):
            return self._download_civitai_lora(model_id)
        elif model_id.startswith("hf:"):
            return self._download_hf_lora(model_id)
        else:
            raise ValueError(f"Unsupported LoRA source: {model_id}. Must start with 'civitai:' or 'hf:'")

    def _manage_model_memory(self, model_key):
        """
        Manage model memory by unloading least recently used models
        when we hit our model cache limit
        """
        # Update access time for the current model
        self.model_last_accessed[model_key] = time.time()
        
        # Check if we need to unload models
        if len(self.loaded_models) > self.max_loaded_models:
            print(f"Cache limit reached ({len(self.loaded_models)} models). Unloading least recently used model.")
            print(f"GPU memory before unloading: {get_gpu_memory_info()}")
            
            # Find least recently used model
            lru_model_key = min(self.model_last_accessed.items(), key=lambda x: x[1])[0]
            
            # Only unload if it's not the model we just loaded
            if lru_model_key != model_key and lru_model_key in self.loaded_models:
                print(f"Unloading model {lru_model_key} from memory")
                
                # Perform deep cleanup
                try:
                    # Clean up PEFT adapters if present
                    self._clean_peft_adapters(self.loaded_models[lru_model_key])
                    
                    # Move model to CPU first to free GPU memory
                    self.loaded_models[lru_model_key].to("cpu")
                    
                    # Delete the model
                    del self.loaded_models[lru_model_key]
                    del self.model_last_accessed[lru_model_key]
                    
                    # Explicitly clear CUDA cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()  # Ensure CUDA operations are complete
                        print(f"Cleared CUDA cache, freeing GPU memory")
                        print(f"GPU memory after unloading: {get_gpu_memory_info()}")
                except Exception as e:
                    print(f"Error during model unloading: {e}")

    def _load_pipeline(self, model_id, loras=None):
        """
        Load an SDXL pipeline from a model ID and apply any LoRAs
        
        Args:
            model_id: Base model ID
            loras: List of dictionaries with 'model_id' and 'weight' keys
        """
        # Create a detailed cache key that includes the base model and all LoRAs
        lora_key = ""
        if loras:
            # Sort loras by model_id to ensure consistent keys regardless of order
            sorted_loras = sorted(loras, key=lambda x: x.get('model_id', ''))
            lora_ids = [f"{lora['model_id']}:{lora['weight']}" for lora in sorted_loras]
            lora_key = "_loras_" + "_".join(lora_ids)
            
        model_key = f"{model_id}{lora_key}"
        
        print(f"Looking for model with key: {model_key}")
        print(f"Current loaded models: {list(self.loaded_models.keys())}")
        print(f"Current GPU memory: {get_gpu_memory_info()}")
        
        if model_key not in self.loaded_models:
            print(f"Model not found in cache. Loading SDXL model: {model_id}")
            try:
                # Check if we have a similar model already loaded (same base model, different LoRAs)
                base_model_key = None
                for key in self.loaded_models.keys():
                    if key.startswith(model_id) and loras:  # Same base model with LoRAs
                        base_model_key = key
                        print(f"Found base model already loaded: {base_model_key}")
                        break
                    
                # Clear CUDA cache before loading a new model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                # Use float16 instead of bfloat16 to fix color issues
                target_dtype = torch.float16
                print(f"Using torch_dtype: {target_dtype}")
                
                # If we found a base model, unload its LoRAs and reuse it instead of loading from scratch
                if base_model_key and base_model_key in self.loaded_models:
                    print(f"Reusing base model {model_id} and applying new LoRAs")
                    pipeline = self.loaded_models[base_model_key]
                    
                    # Clean existing LoRAs thoroughly
                    self._clean_peft_adapters(pipeline)
                    
                    # Remove the old model key
                    del self.loaded_models[base_model_key]
                    if base_model_key in self.model_last_accessed:
                        del self.model_last_accessed[base_model_key]
                else:    
                    # Check if this is a CivitAI model
                    if model_id.startswith("civitai:"):
                        model_path, model_filename = self._download_civitai_model(model_id)
                        
                        # Load from local path
                        print(f"Loading CivitAI SDXL model from {model_path}/{model_filename}")
                        pipeline = diffusers.StableDiffusionXLPipeline.from_single_file(
                            f"{model_path}/{model_filename}",
                            torch_dtype=target_dtype,
                            use_safetensors=True,
                        )
                    else:
                        # Load from Hugging Face
                        pipeline = diffusers.StableDiffusionXLPipeline.from_pretrained(
                            model_id,
                            torch_dtype=target_dtype,
                            use_safetensors=True,
                            variant="fp16" if target_dtype == torch.float16 else "bf16",
                        )
                
                # VAE FIX: Attempt to load a specific VAE, especially for CivitAI models
                if model_id.startswith("civitai:") or model_id != DEFAULT_MODEL_ID:
                    try:
                        print("Attempting to load and set 'madebyollin/sdxl-vae-fp16-fix' VAE")
                        vae = AutoencoderKL.from_pretrained(
                            "madebyollin/sdxl-vae-fp16-fix",
                            torch_dtype=target_dtype
                        )
                        pipeline.vae = vae.to("cuda")
                        print("Successfully loaded and set 'madebyollin/sdxl-vae-fp16-fix' VAE.")
                    except Exception as e:
                        print(f"Warning: Could not load 'madebyollin/sdxl-vae-fp16-fix': {e}. Using model's default VAE.")
                
                # Move to GPU
                pipeline = pipeline.to("cuda")
                
                # Try to enable memory efficient attention if available
                try:
                    if hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
                        pipeline.enable_xformers_memory_efficient_attention()
                        print("Enabled xformers memory efficient attention")
                    elif hasattr(pipeline, "enable_attention_slicing"):
                        pipeline.enable_attention_slicing()
                        print("Enabled attention slicing as fallback")
                except Exception as e:
                    print(f"Could not enable optimized attention: {str(e)}")
                    print("Using default attention mechanism")
                    # Enable attention slicing as a fallback for memory efficiency
                    try:
                        pipeline.enable_attention_slicing()
                        print("Enabled attention slicing")
                    except:
                        pass
                
                # Apply LoRAs if specified
                if loras:
                    # Force a completely clean state by removing any PEFT configs
                    self._clean_peft_adapters(pipeline)
                    
                    # Track loaded adapters
                    loaded_adapter_names = []
                    adapter_weights = []
                    
                    # Track which LoRAs succeeded and failed
                    successful_loras = []
                    failed_loras = []
                    
                    for lora_index, lora_spec in enumerate(loras):
                        weight = float(lora_spec.get("weight", 0.75))
                        lora_model_id = lora_spec.get("model_id", "")
                        
                        # Skip if no model ID
                        if not lora_model_id:
                            failed_loras.append(f"LoRA #{lora_index+1}: Missing model_id")
                            continue
                        
                        # Download the LoRA file
                        try:
                            lora_path = self._download_lora(lora_spec)
                            # Add to loaded LoRAs set to track
                            self.loaded_loras.add(lora_model_id)
                        except Exception as e:
                            failed_loras.append(f"LoRA {lora_model_id}: Download failed - {str(e)}")
                            continue
                        
                        # Apply the LoRA
                        print(f"Applying LoRA {lora_index+1}/{len(loras)} from {lora_path} with weight {weight}")
                        try:
                            # Create a unique adapter name
                            adapter_name = f"lora_{lora_index}"
                            
                            # Load weights with a dedicated adapter name
                            pipeline.load_lora_weights(
                                lora_path, 
                                adapter_name=adapter_name
                            )
                            
                            # Track this adapter
                            loaded_adapter_names.append(adapter_name)
                            adapter_weights.append(weight)
                            
                            successful_loras.append(lora_model_id)
                            
                        except Exception as e:
                            print(f"Error applying LoRA {lora_model_id}: {str(e)}")
                            failed_loras.append(f"LoRA {lora_model_id}: Application failed - {str(e)}")
                    
                    # Set weights for all adapters if any were loaded
                    if loaded_adapter_names:
                        try:
                            print(f"Setting adapters: {loaded_adapter_names} with weights: {adapter_weights}")
                            pipeline.set_adapters(loaded_adapter_names, adapter_weights)
                            print(f"Successfully applied {len(successful_loras)} LoRAs with adapter fusion.")
                        except Exception as e_set_adapters:
                            print(f"Error during pipeline.set_adapters: {e_set_adapters}")
                            # Add any LoRAs that were loaded but failed in set_adapters to failed_loras list
                            failed_loras.extend([f"{lid} (set_adapters failed)" for lid in successful_loras])
                            successful_loras = [] # Clear successful if set_adapters failed for the batch
                    
                    # Summarize LoRA loading results
                    if successful_loras:
                        print(f"Successfully loaded {len(successful_loras)}/{len(loras)} LoRAs: {', '.join(successful_loras)}")
                    if failed_loras:
                        print(f"WARNING: Failed to load {len(failed_loras)}/{len(loras)} LoRAs:")
                        for fail in failed_loras:
                            print(f"  - {fail}")
                
                # Store loaded model
                self.loaded_models[model_key] = pipeline
                
                # Update model access tracking
                self._manage_model_memory(model_key)
                
                # Print memory usage after loading
                print(f"GPU memory after loading model: {get_gpu_memory_info()}")
            except Exception as e:
                print(f"Failed to load SDXL model: {str(e)}")
                raise ValueError(f"Could not load SDXL model {model_id}: {str(e)}")
                
        else:
            # Model is already loaded, update access time
            print(f"Model {model_key} already loaded, reusing from cache")
            self._manage_model_memory(model_key)
                
        return self.loaded_models[model_key]
        
    def _clean_peft_adapters(self, pipeline):
        """
        Thoroughly clean all PEFT and LoRA adapters from the pipeline.
        
        Args:
            pipeline: The diffusers pipeline to clean
        """
        print("Starting deep PEFT adapter cleanup")
        try:
            # Method 1: Standard unload_lora_weights
            if hasattr(pipeline, "unload_lora_weights"):
                pipeline.unload_lora_weights()
                print("Called unload_lora_weights")
                
            # Method 2: Remove PEFT configs explicitly from UNet
            if hasattr(pipeline, "unet"):
                for name, module in pipeline.unet.named_modules():
                    if hasattr(module, "peft_config"):
                        print(f"Removing peft_config from {name}")
                        delattr(module, "peft_config")
                        
            # Method 3: Remove adapter modules if present
            if hasattr(pipeline, "unet") and hasattr(pipeline.unet, "disable_adapter"):
                print("Disabling UNet adapter")
                pipeline.unet.disable_adapter()
                
            # Method 4: Reset all LoRA layers
            if hasattr(pipeline, "unet"):
                from diffusers.models.lora import LoRACompatibleLinear
                from peft.tuners.lora import LoraLayer
                
                # Clean UNet LoRA layers
                for module in pipeline.unet.modules():
                    if isinstance(module, LoRACompatibleLinear) and hasattr(module, "lora_layer"):
                        print("Resetting UNet LoRA layer")
                        module.lora_layer = None
                    elif hasattr(module, "_lora_layer"):
                        print("Resetting _lora_layer")
                        module._lora_layer = None
                    # Also check for PEFT's direct LoraLayer implementations
                    elif isinstance(module, LoraLayer):
                        print("Resetting PEFT LoraLayer")
                        # Reset the layer's parameters
                        if hasattr(module, "lora_A"):
                            module.lora_A.data.zero_()
                        if hasattr(module, "lora_B"):
                            module.lora_B.data.zero_()
                
            # Clean text encoder LoRA layers too
            for encoder in [pipeline.text_encoder, pipeline.text_encoder_2]:
                if encoder is not None:
                    for module in encoder.modules():
                        if isinstance(module, LoRACompatibleLinear) and hasattr(module, "lora_layer"):
                            print("Resetting text encoder LoRA layer")
                            module.lora_layer = None
                        elif hasattr(module, "_lora_layer"):
                            print("Resetting text encoder _lora_layer")
                            module._lora_layer = None
                        elif isinstance(module, LoraLayer):
                            print("Resetting text encoder PEFT LoraLayer")
                            if hasattr(module, "lora_A"):
                                module.lora_A.data.zero_()
                            if hasattr(module, "lora_B"):
                                module.lora_B.data.zero_()
                
            # Method 5: Clean any active adapters list
            if hasattr(pipeline, "active_adapters"):
                pipeline.active_adapters = None
                print("Reset active_adapters")
                
            print("Completed PEFT adapter cleanup")
            
            # Force a CUDA cache cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
        except Exception as e:
            print(f"Warning during adapter cleanup: {e}")
            
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
    ) -> list[bytes]:
        # Print memory usage at start
        print(f"GPU memory before run: {get_gpu_memory_info()}")
        
        # Use default model if none specified
        actual_model_id = model_id if model_id else DEFAULT_MODEL_ID
        
        # Clear CUDA cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        # Get the appropriate pipeline
        pipe = self._load_pipeline(actual_model_id, loras)
        
        # --- Store original text encoder configs if not already stored for this pipe instance ---
        # This ensures that modifications are not cumulative across calls using the same cached pipe.
        if not hasattr(pipe, "_original_text_encoder_config_dict"):
            pipe._original_text_encoder_config_dict = pipe.text_encoder.config.to_dict()
        if hasattr(pipe, "text_encoder_2") and not hasattr(pipe, "_original_text_encoder_2_config_dict"):
            pipe._original_text_encoder_2_config_dict = pipe.text_encoder_2.config.to_dict()

        # --- Reset text encoders to their original config before applying current clip_skip ---
        # Create new config objects from the stored dictionaries.
        # For SDXL, text_encoder is usually CLIPTextModel, text_encoder_2 is CLIPTextModelWithProjection
        
        # For text_encoder (typically CLIPTextConfig)
        current_text_encoder_config = CLIPTextConfig.from_dict(pipe._original_text_encoder_config_dict.copy())
        pipe.text_encoder.config = current_text_encoder_config
        
        # For text_encoder_2 (typically CLIPTextConfig as well)
        if hasattr(pipe, "text_encoder_2"):
            current_text_encoder_2_config = CLIPTextConfig.from_dict(pipe._original_text_encoder_2_config_dict.copy())
            pipe.text_encoder_2.config = current_text_encoder_2_config
        
        # --- Apply CLIP skip if specified ---
        if clip_skip and clip_skip > 1:
            # SDXL text_encoder (CLIP ViT-L) typically has 12 layers.
            # SDXL text_encoder_2 (OpenCLIP ViT-bigG) typically has 24 or 32 layers.
            # clip_skip=N means use output from Nth-to-last layer. N=1 means last layer (no skip).
            # So, layers_to_use = original_num_layers - (clip_skip - 1)
            
            num_layers_to_actually_skip = clip_skip - 1

            if hasattr(pipe, "text_encoder") and num_layers_to_actually_skip > 0:
                original_layers1 = pipe.text_encoder.config.num_hidden_layers # This is now the restored original
                if num_layers_to_actually_skip < original_layers1:
                    print(f"Applying CLIP skip {clip_skip} to text_encoder (original layers: {original_layers1}, using {original_layers1 - num_layers_to_actually_skip})")
                    pipe.text_encoder.config.num_hidden_layers = original_layers1 - num_layers_to_actually_skip
                else:
                    print(f"Warning: clip_skip={clip_skip} is too high for text_encoder with {original_layers1} layers. Not applying skip to text_encoder.")
            
            if hasattr(pipe, "text_encoder_2") and num_layers_to_actually_skip > 0:
                original_layers2 = pipe.text_encoder_2.config.num_hidden_layers # Restored original
                if num_layers_to_actually_skip < original_layers2:
                    print(f"Applying CLIP skip {clip_skip} to text_encoder_2 (original layers: {original_layers2}, using {original_layers2 - num_layers_to_actually_skip})")
                    pipe.text_encoder_2.config.num_hidden_layers = original_layers2 - num_layers_to_actually_skip
                else:
                    print(f"Warning: clip_skip={clip_skip} is too high for text_encoder_2 with {original_layers2} layers. Not applying skip to text_encoder_2.")
        
        # Set the random seed
        seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        print(f"seeding RNG with {seed}")
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        # Set default dimensions if not provided
        width = width if width is not None else DEFAULT_WIDTH
        height = height if height is not None else DEFAULT_HEIGHT
        
        # Validate dimensions (must be multiples of 8)
        width = max(512, min(2048, (width // 8) * 8))
        height = max(512, min(2048, (height // 8) * 8))
        
        # Set default steps if not provided
        steps = num_inference_steps if num_inference_steps is not None else DEFAULT_STEPS
        
        # Set default guidance scale if not provided
        gs = guidance_scale if guidance_scale is not None else DEFAULT_GUIDANCE_SCALE
        
        # Log LoRA information
        if loras:
            lora_info = ", ".join([f"{lora.get('model_id')}:{lora.get('weight')}" for lora in loras])
            print(f"Using LoRAs: {lora_info}")
            
        print(f"Generating SDXL image with dimensions {width}x{height}, {steps} steps")
        
        # Set up the SDXL pipeline parameters
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        # Apply scheduler if specified
        if scheduler:
            print(f"Applying scheduler: {scheduler}")
            # Create the appropriate scheduler instance based on the name
            if scheduler == "euler_ancestral":
                pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
            elif scheduler == "dpmpp_2m_karras":
                # DPM++ 2M Karras is a specific configuration of DPMSolverMultistepScheduler
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipe.scheduler.config,
                    algorithm_type="dpmsolver++",
                    solver_order=2,
                    use_karras_sigmas=True
                )
            print(f"Applied scheduler: {pipe.scheduler.__class__.__name__}")

        try:
            # --- Encode prompts using the chunking logic ---
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self._encode_prompt_chunked(
                pipe=pipe,
                prompt=prompt,
                negative_prompt=negative_prompt,
                device=pipe.device, # Use the pipeline's device
                batch_size=1  # Changed from batch_size to 1
            )
            
            # Manually repeat the embeddings for batch_size if needed, exactly once
            if batch_size > 1:
                prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)
                negative_prompt_embeds = negative_prompt_embeds.repeat(batch_size, 1, 1)
                pooled_prompt_embeds = pooled_prompt_embeds.repeat(batch_size, 1)
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(batch_size, 1)
                print(f"Expanded embeds for batch_size={batch_size}")

            # Print memory before generation
            print(f"GPU memory before generation: {get_gpu_memory_info()}")
            
            # Generate the image
            images = pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_images_per_prompt=1,  # Changed from batch_size to 1
                num_inference_steps=steps,
                guidance_scale=gs,
                width=width,
                height=height,
                generator=generator,
            ).images
            
            print(f"GPU memory after generation: {get_gpu_memory_info()}")

            # Move embeddings to CPU to free memory
            del prompt_embeds
            del negative_prompt_embeds
            del pooled_prompt_embeds
            del negative_pooled_prompt_embeds
            
            image_output = []
            for image in images:
                with io.BytesIO() as buf:
                    image.save(buf, format="PNG")
                    image_output.append(buf.getvalue())

            # Ensure CUDA cache is cleared after generation to reduce fragmentation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
            # Print memory after cleanup
            print(f"GPU memory after run and cleanup: {get_gpu_memory_info()}")
            
            return image_output
        
        except Exception as e:
            print(f"Error during image generation: {e}")
            # Ensure CUDA cache is cleared in case of error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            raise e

    @modal.fastapi_endpoint(docs=True)
    def web(
        self,
        prompt: str,
        negative_prompt: str = "",
        seed: Optional[int] = None,
        model_id: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        loras: Optional[str] = None,  # JSON string of loras
        clip_skip: Optional[int] = None,
        batch_size: int = 1,
        batch_count: int = 1,
        scheduler: Optional[SchedulerType] = None,
    ):
        # Parse the loras from JSON string if provided
        lora_list = parse_loras(loras)
        
        # Validate batch parameters to reasonable ranges
        batch_size = min(max(1, batch_size), 4)  # Limit to 1-4 images per batch
        batch_count = min(max(1, batch_count), 4)  # Limit to 1-4 batches
        
        # If we need to generate multiple batches or multiple images in a single batch
        if batch_count > 1 or batch_size > 1:
            all_images = []
            
            # For multiple batches
            for i in range(batch_count):
                # Generate a new seed for each batch
                current_seed_for_batch = None
                if seed is not None:
                    # If a master seed is provided, increment it for each batch
                    current_seed_for_batch = seed + i
                    print(f"Generating batch {i+1}/{batch_count} with seed {current_seed_for_batch} (master seed {seed} + {i})")
                else:
                    # Otherwise, generate a random seed for each batch
                    current_seed_for_batch = random.randint(0, 2**32 - 1)
                    print(f"Generating batch {i+1}/{batch_count} with random seed {current_seed_for_batch}")
                
                # Generate images for this batch
                batch_images = self.run.local(
                    prompt=prompt,
                    batch_size=batch_size,
                    negative_prompt=negative_prompt,
                    seed=current_seed_for_batch,
                    model_id=model_id,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    loras=lora_list,
                    clip_skip=clip_skip,
                    scheduler=scheduler,
                )
                all_images.extend(batch_images)
                
            # Return all images in a JSON response
            from fastapi.responses import JSONResponse
            import base64
            
            # Create base64 encoded version of all images for response
            b64_images = [base64.b64encode(img).decode('utf-8') for img in all_images]
            
            return JSONResponse(
                content={"images": b64_images},
                media_type="application/json",
            )
        else:
            # Default case - single batch with single image, return directly
            return Response(
                content=self.run.local(  # run in the same container
                    prompt=prompt, 
                    batch_size=1,
                    negative_prompt=negative_prompt,
                    seed=seed,
                    model_id=model_id,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    loras=lora_list,
                    clip_skip=clip_skip,
                    scheduler=scheduler,
                )[0],
                media_type="image/png",
            )

@app.local_entrypoint()
def entrypoint(
    samples: int = 1,
    prompt: str = "A photorealistic landscape, breathtaking vista, 8k, highly detailed",
    negative_prompt: str = "cartoon, animation, drawing, low quality, blurry, nsfw",
    batch_size: int = 1,
    seed: Optional[int] = None,
    model_id: Optional[str] = None,
    width: Optional[int] = DEFAULT_WIDTH,
    height: Optional[int] = DEFAULT_HEIGHT,
    steps: Optional[int] = DEFAULT_STEPS,
    guidance_scale: Optional[float] = DEFAULT_GUIDANCE_SCALE,
    loras: Optional[str] = None,  # JSON string of loras array
    clip_skip: Optional[int] = None,
    scheduler: Optional[SchedulerType] = None,
):
    # Parse the loras from JSON string if provided
    lora_list = parse_loras(loras)

    print(
        f"prompt => {prompt}",
        f"negative_prompt => {negative_prompt}",
        f"samples => {samples}",
        f"batch_size => {batch_size}",
        f"seed => {seed}",
        f"model_id => {model_id}",
        f"dimensions => {width}x{height}",
        f"steps => {steps}",
        f"guidance_scale => {guidance_scale}",
        f"clip_skip => {clip_skip}",
        f"scheduler => {scheduler}",
        sep="\n",
    )
    
    if lora_list:
        lora_info = ", ".join([f"{lora.get('model_id')}:{lora.get('weight')}" for lora in lora_list])
        print(f"loras => {lora_info}")

    output_dir = Path("/tmp/stable-diffusion")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Don't preload default model when using from command line
    inference_service = Inference(load_default_model=False)

    for sample_idx in range(samples):
        start = time.time()
        images = inference_service.run.remote(
            prompt=prompt, 
            batch_size=batch_size,
            negative_prompt=negative_prompt,
            seed=seed,
            model_id=model_id,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            loras=lora_list,
            clip_skip=clip_skip,
            scheduler=scheduler,
        )
        duration = time.time() - start
        print(f"Run {sample_idx + 1} took {duration:.3f}s")
        if sample_idx:
            print(
                f"\tGenerated {len(images)} image(s) at {(duration) / len(images):.3f}s / image."
            )
        for batch_idx, image_bytes in enumerate(images):
            output_path = (
                output_dir
                / f"output_{slugify(prompt)[:64]}_{str(sample_idx).zfill(2)}_{str(batch_idx).zfill(2)}.png"
            )
            if not batch_idx:
                print("Saving outputs", end="\n\t")
            print(
                output_path,
                end="\n" + ("\t" if batch_idx < len(images) - 1 else ""),
            )
            output_path.write_bytes(image_bytes)


def slugify(text):
    """Convert text to a URL-friendly format"""
    import re
    text = re.sub(r'[^\w\s-]', '', text.lower())
    return re.sub(r'[-\s]+', '-', text).strip('-_')


