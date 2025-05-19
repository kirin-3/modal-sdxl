# SDXL Image Generator

A powerful and flexible text-to-image generation system built with [Modal](https://modal.com/) and [Stable Diffusion XL](https://stability.ai/stable-diffusion). This project provides both a web interface and API for generating high-quality images from text prompts.

## Features

- **High-Quality Image Generation**: Uses Stable Diffusion XL models for state-of-the-art image generation
- **Model Flexibility**: Support for both Hugging Face and CivitAI SDXL models
- **LoRA Support**: Add up to 5 LoRAs from CivitAI or Hugging Face to customize your generations
- **Advanced Options**: Control dimensions, sampling steps, guidance scale, and more
- **Batch Generation**: Generate multiple images in parallel
- **Web Interface**: User-friendly web UI for easy image generation
- **Long Prompt Handling**: Supports very long prompts through token-based chunking
- **CLIP Skip**: Fine-tune the way the model interprets your prompts
- **Multiple Schedulers**: Choose between sampling methods for different quality/speed tradeoffs
- **Configurable GPU**: Select which GPU type to use with Modal

## Setup Instructions

### Prerequisites

- Python 3.12+ recommended
- A Modal account (https://modal.com)
- Flask for the web interface
- A CivitAI account and API token (for using CivitAI models)

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/kirin-3/modal-sdxl
   cd sdxl-generator
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up Modal CLI:
   ```
   pip install modal
   modal token new
   ```

4. Set up a CivitAI token secret in Modal:
   ```
   modal secret create civitai-token --value "YOUR_CIVITAI_API_TOKEN"
   ```
   You can get your CivitAI API token from your account page: https://civitai.com/user/account

5. Create a configuration file:
   ```
   cp example.env .env
   ```
   
   Edit the `.env` file to set your Modal endpoint URL and GPU configuration:
   ```
   # Modal endpoints
   MODAL_ENDPOINT=https://yourusername--text2image-inference-web.modal.run
   
   # GPU configuration (options include: T4, L4, A10G, A100, H100)
   GPU_TYPE=L4
   ```

6. Deploy the SDXL application to Modal:
   ```
   modal deploy text2image.py
   ```

7. Get your Modal endpoint URL:
   ```
   modal endpoints list
   ```
   
   Update your `.env` file with the actual endpoint URL.

### Running the Web Interface

Start the local web server:
```
python local_server.py
```

Visit `http://localhost:5000` in your browser to access the web interface.

## Using the Web Interface

The web interface provides access to all features through an intuitive UI:

1. **Prompts**: Enter your positive and negative prompts
2. **Model Selection**: Choose between Hugging Face models or CivitAI models
3. **Image Settings**: Control dimensions, steps, guidance scale, and other parameters
4. **LoRA Integration**: Add up to 5 LoRAs to customize your generation
5. **Batch Generation**: Generate multiple images at once

Generated images are automatically saved to the `generated_images` directory.

## API Usage

The Modal app exposes an API endpoint that can be called programmatically:

```python
import requests
import json
import base64
from PIL import Image
import io

# Prepare the request
url = "https://yourusername--text2image-web.modal.run"
params = {
    "prompt": "A photorealistic landscape, breathtaking vista, 8k, highly detailed",
    "negative_prompt": "cartoon, animation, drawing, low quality, blurry, nsfw",
    "width": 1024,
    "height": 1024,
    "steps": 30,
    "guidance_scale": 7.5,
    "batch_size": 1,
    "batch_count": 1,
    "scheduler": "euler_ancestral"
}

# Optional: Add LoRAs
loras = [
    {"model_id": "civitai:1681903", "weight": 2.0},
    {"model_id": "civitai:1764869", "weight": 0.75}
]
params["loras"] = json.dumps(loras)

# Make the request
response = requests.get(url, params=params)

# For a single image response
if response.headers.get('content-type') == 'image/png':
    img = Image.open(io.BytesIO(response.content))
    img.save("generated_image.png")
    img.show()
# For a batch of images in JSON response
elif 'application/json' in response.headers.get('content-type', ''):
    data = response.json()
    for i, b64_img in enumerate(data.get('images', [])):
        img_bytes = base64.b64decode(b64_img)
        img = Image.open(io.BytesIO(img_bytes))
        img.save(f"generated_image_{i+1}.png")
```

## Advanced Configuration

### GPU Selection

You can choose from various GPU types in your `.env` file:
- `T4`: Most affordable option, good for basic generations
- `L4`: Good balance of price and performance (default)
- `A10G`: High performance for faster generations
- `A100`: Premium performance for complex generations
- `H100`: Highest performance available

Changing GPU type affects pricing on Modal. See Modal's pricing page for details.

### Available Models

- Default model: `stabilityai/stable-diffusion-xl-base-1.0`
- You can use any SDXL model from Hugging Face or CivitAI
- For CivitAI models, use the model ID from the URL (e.g., `135867` from `https://civitai.com/models/135867`)

### LoRA Configuration

LoRAs allow you to customize the generation style without fine-tuning the entire model:

- For CivitAI LoRAs: Use format `civitai:MODEL_ID` with the model ID from the URL
- For Hugging Face LoRAs: Use format `hf:REPO_ID/PATH` for specific files or `hf:REPO_ID` to auto-detect

The `weight` parameter (0.1-2.0) controls how strongly the LoRA affects the generation.

### Scheduler Options

- `euler_ancestral`: Euler Ancestral - Best overall, default choice
- `dpmpp_2m_karras`: DPM++ 2M Karras - High quality, potentially faster

## Project Architecture

- `text2image.py`: Modal backend application with SDXL pipeline implementation
- `local_server.py`: Flask server providing the web interface
- `templates/index.html`: Web UI template
- `static/css/styles.css`: Styling for the web interface
- `static/js/script.js`: Client-side JavaScript for the web interface
- `.env`: Configuration for Modal endpoint URL and GPU type
- `example.env`: Example configuration template

## Troubleshooting

### Common Issues

1. **"Error from Modal API"**: Ensure your Modal account is set up correctly and the application is deployed
2. **Request Timeouts**: First-time generations can take longer due to model downloads. Subsequent generations will be faster.
3. **Memory Issues**: Reduce batch size, image dimensions, or the number of LoRAs if you encounter CUDA out-of-memory errors
4. **Missing LoRA Effects**: Verify the LoRA ID is correct and try increasing the weight value
5. **CivitAI Access Issues**: Make sure you've created the `civitai-token` secret in Modal with a valid API token

### Advanced Users

For advanced customization, you can modify:

- The schedulers in `text2image.py` to add more sampling options
- The chunking logic in `_encode_prompt_chunked` function to handle even longer prompts
- The pipeline parameters for different generation settings
- GPU configuration in `.env` to optimize for cost or performance

## Acknowledgments

- [Stability AI](https://stability.ai/) for Stable Diffusion XL
- [Modal](https://modal.com/) for the serverless compute platform
- [Hugging Face](https://huggingface.co/) for the Diffusers library
- [CivitAI](https://civitai.com/) for the community models repository 