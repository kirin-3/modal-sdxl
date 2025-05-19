from flask import Flask, render_template, request, send_file
import requests
import io
import os
import json
import base64
from dotenv import load_dotenv
import re
import time
from pathlib import Path
from datetime import datetime
from werkzeug.serving import WSGIRequestHandler

# Set longer timeouts for the Werkzeug server
# This helps prevent connection resets during long-running image generation
WSGIRequestHandler.protocol_version = "HTTP/1.1"  # Use HTTP/1.1 for keep-alive
WSGIRequestHandler.timeout = 1800  # 30 minutes timeout

# Load environment variables
load_dotenv()

# Ensure the templates directory exists
templates_dir = Path("templates")
templates_dir.mkdir(exist_ok=True)

# Create a directory for saved images
output_dir = Path("generated_images")
output_dir.mkdir(exist_ok=True)

app = Flask(__name__)

# Configure Flask for long-running requests
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for development
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Allow up to 50 MB requests
app.config['PROPAGATE_EXCEPTIONS'] = True  # Make sure errors are properly handled

# Add request timeout handling
@app.after_request
def add_timeout_headers(response):
    # Add response headers to help prevent timeouts in proxies/browsers
    response.headers['Connection'] = 'keep-alive'
    response.headers['Keep-Alive'] = 'timeout=1800'  # 30 minutes
    return response

# Helper function for slugifying prompt text
def slugify(text):
    """Convert text to a URL-friendly format"""
    text = re.sub(r'[^\w\s-]', '', text.lower())
    return re.sub(r'[-\s]+', '-', text).strip('-_')

# Custom template filter for base64 encoding
@app.template_filter('b64encode')
def b64encode_filter(data):
    if data:
        return base64.b64encode(data).decode('utf-8')
    return ''

# Default model for the interface
DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

# Default generation parameters
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_STEPS = 30  # Default steps for SDXL
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_SCHEDULER = "euler_ancestral"  # Default scheduler

# Available schedulers with descriptive names
AVAILABLE_SCHEDULERS = {
    "euler_ancestral": "Euler Ancestral (Best overall, default)",
    "dpmpp_2m_karras": "DPM++ 2M Karras (High quality, faster)"
}

@app.route('/', methods=['GET', 'POST'])
def index():
    image = None
    error = None
    prompt = ""
    negative_prompt = ""
    seed = None
    model_id = DEFAULT_MODEL_ID
    is_civitai = False
    civitai_id = "1637364"  # Default CivitAI ID
    width = DEFAULT_WIDTH
    height = DEFAULT_HEIGHT
    steps = DEFAULT_STEPS
    guidance_scale = DEFAULT_GUIDANCE_SCALE
    clip_skip = None
    scheduler = DEFAULT_SCHEDULER
    batch_size = 1
    batch_count = 1
    images = []  # For storing multiple images
    saved_paths = []  # Initialize empty list for saved paths
    force_civitai = True  # Force CivitAI by default
    loras = []
    
    # Pre-filled LoRA values
    loras = [
        {'model_id': 'civitai:1681903', 'weight': 2.0},
        {'model_id': 'civitai:1764869', 'weight': 0.75}
    ]
    
    if request.method == 'POST':
        try:
            # Set a 5-minute timeout for the request
            request.environ.get('werkzeug.server.shutdown')
            
            prompt = request.form.get('prompt', '')
            negative_prompt = request.form.get('negative_prompt', '')
            seed_input = request.form.get('seed', '')
            model_source = request.form.get('model_source', 'civitai')  # Default to civitai
            
            # Get batch parameters with additional validation
            try:
                batch_size = int(request.form.get('batch_size', 1))
                # Add more strict limits to prevent connection resets
                if batch_size < 1:
                    batch_size = 1
                elif batch_size > 4:
                    batch_size = 4
            except ValueError:
                batch_size = 1
                
            try:
                batch_count = int(request.form.get('batch_count', 1))
                # Add more strict limits to prevent connection resets
                if batch_count < 1:
                    batch_count = 1
                elif batch_count > 4:
                    batch_count = 4
            except ValueError:
                batch_count = 1
            
            # Calculate total images and apply safety limit
            total_images = batch_size * batch_count
            if total_images > 8:  # Cap at 8 to prevent overload
                # Adjust batch size and count to keep under limit
                if batch_size > 2:
                    batch_size = 2
                batch_count = min(batch_count, 4)
                print(f"Reduced batch parameters to prevent server overload: size={batch_size}, count={batch_count}")
            
            # Get image dimensions
            try:
                width = int(request.form.get('width', DEFAULT_WIDTH))
                if width < 512 or width > 2048:
                    width = DEFAULT_WIDTH
            except ValueError:
                width = DEFAULT_WIDTH
                
            try:
                height = int(request.form.get('height', DEFAULT_HEIGHT))
                if height < 512 or height > 2048:
                    height = DEFAULT_HEIGHT
            except ValueError:
                height = DEFAULT_HEIGHT
                
            # Get inference steps
            try:
                steps = int(request.form.get('steps', DEFAULT_STEPS))
                if steps < 1 or steps > 150:
                    steps = DEFAULT_STEPS
            except ValueError:
                steps = DEFAULT_STEPS
                
            # Get guidance scale
            try:
                guidance_scale = float(request.form.get('guidance_scale', DEFAULT_GUIDANCE_SCALE))
                if guidance_scale < 1.0 or guidance_scale > 20.0:
                    guidance_scale = DEFAULT_GUIDANCE_SCALE
            except ValueError:
                guidance_scale = DEFAULT_GUIDANCE_SCALE
                
            # Get CLIP skip
            clip_skip_input = request.form.get('clip_skip', '')
            if clip_skip_input and clip_skip_input.strip():
                try:
                    clip_skip = int(clip_skip_input)
                    # CLIP skip should be between 1 and 4
                    if clip_skip < 1 or clip_skip > 4:
                        clip_skip = None
                except ValueError:
                    clip_skip = None
            
            # Get scheduler
            scheduler_input = request.form.get('scheduler', DEFAULT_SCHEDULER)
            if scheduler_input not in AVAILABLE_SCHEDULERS:
                scheduler_input = DEFAULT_SCHEDULER
            
            # Process LoRAs
            # Support up to 5 LoRAs
            loras = []
            for i in range(1, 6):
                lora_enabled = request.form.get(f'lora{i}_enabled') == 'on'
                if lora_enabled:
                    lora_source = request.form.get(f'lora{i}_source')
                    lora_id = request.form.get(f'lora{i}_id', '').strip()
                    lora_weight = request.form.get(f'lora{i}_weight', 0.75)
                    
                    try:
                        lora_weight = float(lora_weight)
                        if lora_weight < 0.1 or lora_weight > 2.0:
                            lora_weight = 0.75
                    except ValueError:
                        lora_weight = 0.75
                    
                    if lora_id:
                        # Format the lora ID properly based on source
                        model_id_formatted = ""
                        if lora_source == 'civitai':
                            model_id_formatted = f"civitai:{lora_id}"
                        elif lora_source == 'huggingface':
                            model_id_formatted = f"hf:{lora_id}"
                        
                        if model_id_formatted:
                            loras.append({
                                'model_id': model_id_formatted,
                                'weight': lora_weight
                            })
            
            # Handle model selection based on source
            if model_source == 'huggingface':
                model_id = request.form.get('model_id', DEFAULT_MODEL_ID).strip() or DEFAULT_MODEL_ID
                is_civitai = False
            else:  # civitai
                civitai_id = request.form.get('civitai_id', '').strip()
                if civitai_id:
                    model_id = f"civitai:{civitai_id}"
                    is_civitai = True
                else:
                    error = "Please enter a CivitAI model ID"
                    return render_template('index.html', 
                                        error=error, 
                                        prompt=prompt,
                                        negative_prompt=negative_prompt,
                                        seed=seed_input,
                                        model_id=model_id,
                                        default_model=DEFAULT_MODEL_ID,
                                        is_civitai=is_civitai,
                                        civitai_id=civitai_id,
                                        width=width,
                                        height=height,
                                        steps=steps,
                                        guidance_scale=guidance_scale,
                                        clip_skip=clip_skip,
                                        scheduler=scheduler_input,
                                        loras=loras,
                                        force_civitai=force_civitai)
            
            # Check if seed is provided and valid
            if seed_input and seed_input.strip():
                try:
                    seed = int(seed_input)
                except ValueError:
                    error = "Seed must be an integer"
                    return render_template('index.html', 
                                        error=error, 
                                        prompt=prompt,
                                        negative_prompt=negative_prompt,
                                        seed=seed_input,
                                        model_id=model_id,
                                        default_model=DEFAULT_MODEL_ID,
                                        is_civitai=is_civitai,
                                        civitai_id=civitai_id,
                                        width=width,
                                        height=height,
                                        steps=steps,
                                        guidance_scale=guidance_scale,
                                        clip_skip=clip_skip,
                                        scheduler=scheduler_input,
                                        loras=loras,
                                        force_civitai=force_civitai)
            
            # Get the Modal endpoint URL from environment variable
            modal_endpoint = os.getenv('MODAL_ENDPOINT', '')
            
            if not modal_endpoint:
                error = "Modal endpoint URL not configured. Please set the MODAL_ENDPOINT environment variable."
            else:
                try:
                    # Add parameters to the URL
                    params = {
                        'prompt': prompt,
                        'negative_prompt': negative_prompt,
                        'width': width,
                        'height': height,
                        'steps': steps,
                        'guidance_scale': guidance_scale,
                        'batch_size': batch_size,
                        'batch_count': batch_count,
                        'scheduler': scheduler_input
                    }
                    
                    # Optional parameters
                    if seed is not None:
                        params['seed'] = seed
                        
                    if clip_skip is not None:
                        params['clip_skip'] = clip_skip
                    
                    # Only pass model_id if it was explicitly entered
                    if model_id and (is_civitai or model_id != DEFAULT_MODEL_ID):
                        params['model_id'] = model_id
                    
                    # Add LoRAs if any are specified
                    if loras:
                        params['loras'] = json.dumps(loras)
                    
                    # Show user which model is being used
                    model_display = civitai_id if is_civitai else (model_id or DEFAULT_MODEL_ID)
                    print(f"Generating SDXL image with model: {model_display}, dimensions: {width}x{height}, steps: {steps}, guidance: {guidance_scale}")
                    print(f"Batch settings: {batch_size} images per batch, {batch_count} batches")
                    
                    if loras:
                        lora_info = ", ".join([f"{lora.get('model_id')}:{lora.get('weight')}" for lora in loras])
                        print(f"Using LoRAs: {lora_info}")
                    
                    # Calculate an appropriate timeout based on batch size and count
                    # Base timeout is 5 minutes, add 2 minutes per additional batch or complex operation
                    timeout_seconds = 300 + (batch_count - 1) * 120 + (batch_size - 1) * 60
                    
                    # Increase timeout further if using custom models (which might need downloading)
                    if model_id != DEFAULT_MODEL_ID or is_civitai or loras:
                        timeout_seconds += 180  # Add 3 more minutes for model downloads
                    
                    print(f"Using request timeout of {timeout_seconds} seconds")
                    
                    # Make the request to Modal with increased timeout and appropriate error handling
                    session = requests.Session()
                    session.mount('https://', requests.adapters.HTTPAdapter(max_retries=3))
                    
                    try:
                        response = session.get(
                            modal_endpoint, 
                            params=params, 
                            timeout=timeout_seconds,
                            stream=True
                        )
                        response.raise_for_status()
                    except requests.exceptions.RequestException as req_err:
                        print(f"Request error: {req_err}")
                        if isinstance(req_err, requests.exceptions.Timeout):
                            error = "The request timed out. Try with a smaller batch size or fewer steps."
                        elif isinstance(req_err, requests.exceptions.ConnectionError):
                            error = "Connection error. The server may be temporarily unavailable."
                        else:
                            error = f"Error from Modal API: {str(req_err)}"
                        return render_template('index.html', 
                                            error=error, 
                                            prompt=prompt,
                                            negative_prompt=negative_prompt,
                                            seed=seed,
                                            model_id=model_id,
                                            default_model=DEFAULT_MODEL_ID,
                                            is_civitai=is_civitai,
                                            civitai_id=civitai_id,
                                            width=width,
                                            height=height,
                                            steps=steps,
                                            guidance_scale=guidance_scale,
                                            clip_skip=clip_skip,
                                            scheduler=scheduler_input,
                                            batch_size=batch_size,
                                            batch_count=batch_count,
                                            loras=loras,
                                            force_civitai=force_civitai)
                    
                    if response.status_code == 200:
                        # Check if we got a JSON response with multiple images
                        content_type = response.headers.get('content-type', '')
                        print(f"Response content type: {content_type}")
                        
                        if 'application/json' in content_type:
                            try:
                                resp_data = response.json()
                                if 'images' in resp_data:
                                    num_images = len(resp_data['images'])
                                    print(f"Received {num_images} images in JSON response")
                                    
                                    # Store all images for display
                                    images = []
                                    saved_paths = []  # Track paths of saved images
                                    
                                    # Create timestamp and base filename
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    prompt_slug = slugify(prompt[:50])  # First 50 chars of prompt
                                    
                                    for i, b64_img in enumerate(resp_data['images']):
                                        try:
                                            img_bytes = base64.b64decode(b64_img)
                                            images.append(img_bytes)
                                            
                                            # Save the image to disk
                                            filename = f"{timestamp}_{prompt_slug}_{i+1}.png"
                                            save_path = output_dir / filename
                                            with open(save_path, 'wb') as f:
                                                f.write(img_bytes)
                                            saved_paths.append(str(save_path))
                                            print(f"Saved image to {save_path}")
                                        except Exception as img_err:
                                            print(f"Error processing image {i+1}: {str(img_err)}")
                                    
                                    # Use the first image as the main display image
                                    if images:
                                        image = images[0]
                                    else:
                                        error = "No images were returned"
                            except Exception as e:
                                error = f"Error processing batch images: {str(e)}"
                        else:
                            # Single image response - process in chunks to avoid memory issues
                            image_data = b''
                            for chunk in response.iter_content(chunk_size=8192):
                                image_data += chunk
                            
                            # Process single image
                            image = image_data
                            
                            # Save single image
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            prompt_slug = slugify(prompt[:50])
                            filename = f"{timestamp}_{prompt_slug}.png"
                            save_path = output_dir / filename
                            with open(save_path, 'wb') as f:
                                f.write(image)
                            saved_paths = [str(save_path)]
                            print(f"Saved image to {save_path}")
                    else:
                        error = f"Error from Modal API: {response.status_code} - {response.text}"
                except Exception as e:
                    print(f"Exception during request processing: {str(e)}")
                    error = f"Error connecting to Modal API: {str(e)}"
        except Exception as e:
            print(f"Unhandled exception in route: {str(e)}")
            error = f"An unexpected error occurred: {str(e)}"
    
    return render_template('index.html', 
                        image=image, 
                        images=images,  # Add all images for gallery view
                        saved_paths=saved_paths if 'saved_paths' in locals() else [],  # Pass saved image paths
                        output_directory_display=str(output_dir.resolve()),  # Add absolute path of output directory
                        error=error, 
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        seed=seed,
                        model_id=model_id,
                        default_model=DEFAULT_MODEL_ID,
                        is_civitai=is_civitai,
                        civitai_id=civitai_id,
                        width=width,
                        height=height,
                        steps=steps,
                        guidance_scale=guidance_scale,
                        clip_skip=clip_skip,
                        scheduler=scheduler,
                        scheduler_options=AVAILABLE_SCHEDULERS,
                        batch_size=batch_size,
                        batch_count=batch_count,
                        loras=loras,
                        force_civitai=force_civitai)

@app.route('/image')
def get_image():
    image_data = request.args.get('data')
    if image_data:
        return send_file(io.BytesIO(image_data), mimetype='image/png')
    return "No image data", 400

if __name__ == '__main__':
    print("=== SDXL Image Generator Server ===")
    print("Server configured with 30-minute request timeout for long-running image generation")
    print("NOTE: For production use, consider using Gunicorn with:")
    print("      gunicorn --workers=2 --timeout=1800 local_server:app")
    print("Starting development server...")
    
    # Set Flask server options with best possible timeout handling for development server
    app.run(debug=True, 
            port=5000, 
            threaded=True,     # Use threading for concurrent requests
            host='0.0.0.0',    # Listen on all interfaces
            use_reloader=True) # Auto-reload on code changes 