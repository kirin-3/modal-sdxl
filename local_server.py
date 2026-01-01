from flask import Flask, render_template, request, send_file, session, redirect, url_for, flash
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
from werkzeug.exceptions import HTTPException

# Set longer timeouts for the Werkzeug server
WSGIRequestHandler.protocol_version = "HTTP/1.1"
WSGIRequestHandler.timeout = 1800  # 30 minutes timeout

# Override handle_one_request method to better handle long-running connections
original_handle = WSGIRequestHandler.handle_one_request

def patched_handle_one_request(self):
    try:
        return original_handle(self)
    except (ConnectionResetError, BrokenPipeError) as e:
        print(f"Connection error handled gracefully: {str(e)}")
        return

WSGIRequestHandler.handle_one_request = patched_handle_one_request

# Load environment variables
load_dotenv()

# Ensure the templates directory exists
templates_dir = Path("templates")
templates_dir.mkdir(exist_ok=True)

# Create a directory for saved images
output_dir = Path("generated_images")
output_dir.mkdir(exist_ok=True)

app = Flask(__name__)
# Use environment variable for secret key or fallback to a fixed development key
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-stable-for-restarts')

# Configure Flask for long-running requests
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['PROPAGATE_EXCEPTIONS'] = True

app.config.update(
    SESSION_COOKIE_SECURE=False,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=1800,
    SESSION_USE_SIGNER=False
)

@app.after_request
def add_response_headers(response):
    response.headers['Connection'] = 'keep-alive'
    response.headers['Keep-Alive'] = 'timeout=1800'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
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

# Constants
DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_SCHEDULER = "euler_ancestral"

AVAILABLE_SCHEDULERS = {
    "euler_ancestral": "Euler Ancestral (Best overall, default)",
    "dpmpp_2m_karras": "DPM++ 2M Karras (High quality, faster)"
}

def get_default_context():
    """Returns the default context variables for the template."""
    return {
        'prompt': "",
        'negative_prompt': "",
        'seed': None,
        'model_id': DEFAULT_MODEL_ID,
        'default_model': DEFAULT_MODEL_ID,
        'is_civitai': False,
        'civitai_id': "1637364",
        'width': DEFAULT_WIDTH,
        'height': DEFAULT_HEIGHT,
        'steps': DEFAULT_STEPS,
        'guidance_scale': DEFAULT_GUIDANCE_SCALE,
        'clip_skip': None,
        'scheduler': DEFAULT_SCHEDULER,
        'scheduler_options': AVAILABLE_SCHEDULERS,
        'batch_size': 1,
        'batch_count': 1,
        'loras': [],
        'force_civitai': True,
        'saved_paths': [],
        'image_filenames': [],
        'image': None,
        'images': [],
        'output_directory_display': str(output_dir.resolve()),
        'error': None
    }

def validate_and_extract_params(form):
    """Validates form input and extracts generation parameters."""
    params = {}
    context_update = {}
    
    # Basic text fields
    params['prompt'] = form.get('prompt', '')
    params['negative_prompt'] = form.get('negative_prompt', '')
    context_update['prompt'] = params['prompt']
    context_update['negative_prompt'] = params['negative_prompt']

    # Seed
    seed_input = form.get('seed', '')
    if seed_input and seed_input.strip():
        try:
            params['seed'] = int(seed_input)
            context_update['seed'] = params['seed']
        except ValueError:
            return None, None, "Seed must be an integer"
    
    # Batch parameters
    try:
        batch_size = int(form.get('batch_size', 1))
        batch_size = max(1, min(batch_size, 4))
        params['batch_size'] = batch_size
    except ValueError:
        params['batch_size'] = 1
    
    try:
        batch_count = int(form.get('batch_count', 1))
        batch_count = max(1, min(batch_count, 4))
        params['batch_count'] = batch_count
    except ValueError:
        params['batch_count'] = 1

    # Safety limit for total images
    if params['batch_size'] * params['batch_count'] > 8:
         if params['batch_size'] > 2: params['batch_size'] = 2
         params['batch_count'] = min(params['batch_count'], 4)
         print(f"Reduced batch parameters: size={params['batch_size']}, count={params['batch_count']}")
    
    context_update['batch_size'] = params['batch_size']
    context_update['batch_count'] = params['batch_count']

    # Dimensions
    try:
        width = int(form.get('width', DEFAULT_WIDTH))
        params['width'] = width if 512 <= width <= 2048 else DEFAULT_WIDTH
    except ValueError:
        params['width'] = DEFAULT_WIDTH
    
    try:
        height = int(form.get('height', DEFAULT_HEIGHT))
        params['height'] = height if 512 <= height <= 2048 else DEFAULT_HEIGHT
    except ValueError:
        params['height'] = DEFAULT_HEIGHT

    context_update['width'] = params['width']
    context_update['height'] = params['height']

    # Steps & Guidance
    try:
        steps = int(form.get('steps', DEFAULT_STEPS))
        params['steps'] = steps if 1 <= steps <= 150 else DEFAULT_STEPS
    except ValueError:
        params['steps'] = DEFAULT_STEPS
    
    try:
        scale = float(form.get('guidance_scale', DEFAULT_GUIDANCE_SCALE))
        params['guidance_scale'] = scale if 1.0 <= scale <= 20.0 else DEFAULT_GUIDANCE_SCALE
    except ValueError:
        params['guidance_scale'] = DEFAULT_GUIDANCE_SCALE

    context_update['steps'] = params['steps']
    context_update['guidance_scale'] = params['guidance_scale']

    # Clip Skip
    clip_skip_input = form.get('clip_skip', '')
    if clip_skip_input and clip_skip_input.strip():
        try:
            val = int(clip_skip_input)
            if 1 <= val <= 4:
                params['clip_skip'] = val
                context_update['clip_skip'] = val
        except ValueError:
            pass
    
    # Scheduler
    scheduler = form.get('scheduler', DEFAULT_SCHEDULER)
    if scheduler not in AVAILABLE_SCHEDULERS:
        scheduler = DEFAULT_SCHEDULER
    params['scheduler'] = scheduler
    context_update['scheduler'] = scheduler

    # Model Selection
    model_source = form.get('model_source', 'civitai')
    if model_source == 'huggingface':
        model_id = form.get('model_id', DEFAULT_MODEL_ID).strip() or DEFAULT_MODEL_ID
        params['model_id'] = model_id
        context_update['model_id'] = model_id
        context_update['is_civitai'] = False
        context_update['civitai_id'] = ""
    else:
        civitai_id = form.get('civitai_id', '').strip()
        if civitai_id:
            params['model_id'] = f"civitai:{civitai_id}"
            context_update['civitai_id'] = civitai_id
            context_update['is_civitai'] = True
            context_update['model_id'] = params['model_id'] 
        else:
            return None, context_update, "Please enter a CivitAI model ID"

    # LoRAs
    loras = []
    for i in range(1, 6):
        if form.get(f'lora{i}_enabled') == 'on':
            source = form.get(f'lora{i}_source')
            l_id = form.get(f'lora{i}_id', '').strip()
            l_weight_val = form.get(f'lora{i}_weight', 0.75)
            
            try:
                weight = float(l_weight_val)
                weight = weight if 0.1 <= weight <= 2.0 else 0.75
            except ValueError:
                weight = 0.75
            
            if l_id:
                prefix = "civitai:" if source == 'civitai' else "hf:"
                loras.append({'model_id': f"{prefix}{l_id}", 'weight': weight})
    
    if loras:
        params['loras'] = json.dumps(loras)
        context_update['loras'] = loras # Store as list (dicts) for template rendering

    return params, context_update, None

def generate_image_task(params):
    """Calls the Modal API to generate images."""
    modal_endpoint = os.getenv('MODAL_ENDPOINT', '')
    if not modal_endpoint:
        raise ValueError("Modal endpoint URL not configured.")

    # Calculate timeout
    timeout_seconds = 300 + (params.get('batch_count', 1) - 1) * 120 + (params.get('batch_size', 1) - 1) * 60
    # Add extra time for cold starts or downloads
    if params.get('model_id') != DEFAULT_MODEL_ID or 'loras' in params:
        timeout_seconds += 180
    
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=3, pool_connections=5, pool_maxsize=10)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    headers = {
        'Connection': 'keep-alive', 
        'Keep-Alive': 'timeout=600, max=1000', 
        'User-Agent': 'SDXL-Generator/1.0'
    }

    model_display = params.get('model_id')
    print(f"Generating SDXL image with model: {model_display}")
    print(f"Using request timeout of {timeout_seconds} seconds")
    
    try:
        response = session.get(modal_endpoint, params=params, timeout=timeout_seconds, headers=headers)
        response.raise_for_status()
        return response
    except requests.exceptions.Timeout:
        raise ValueError("The request timed out. Try with a smaller batch size or fewer steps.")
    except requests.exceptions.ConnectionError:
        raise ValueError("Connection error. The server may be temporarily unavailable.")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Error from Modal API: {str(e)}")

def save_images(response, prompt):
    """Processes the API response and saves images to disk."""
    content_type = response.headers.get('content-type', '')
    saved_paths = []
    filenames = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_slug = slugify(prompt[:50])

    if 'application/json' in content_type:
        try:
            data = response.json()
            images = data.get('images', [])
            print(f"Received {len(images)} images in JSON response")
            
            for i, b64_img in enumerate(images):
                try:
                    img_bytes = base64.b64decode(b64_img)
                    filename = f"{timestamp}_{prompt_slug}_{i+1}.png"
                    save_path = output_dir / filename
                    with open(save_path, 'wb') as f:
                        f.write(img_bytes)
                    saved_paths.append(str(save_path))
                    filenames.append(filename)
                    print(f"Saved image to {save_path}")
                except Exception as img_err:
                    print(f"Error processing image {i+1}: {str(img_err)}")
        except Exception as e:
            raise ValueError(f"Error processing JSON response: {str(e)}")
    else:
        # Binary response (single image)
        try:
            filename = f"{timestamp}_{prompt_slug}.png"
            save_path = output_dir / filename
            with open(save_path, 'wb') as f:
                f.write(response.content)
            saved_paths.append(str(save_path))
            filenames.append(filename)
            print(f"Saved image to {save_path}")
        except Exception as e:
            raise ValueError(f"Error saving image: {str(e)}")
    
    if not saved_paths:
         raise ValueError("No images were returned or saved.")
         
    return saved_paths, filenames

@app.route('/', methods=['GET', 'POST'])
def index():
    context = get_default_context()
    
    # Initialize default LoRAs for GET requests if no session data
    if request.method == 'GET' and not ('generation_data' in session):
         context['loras'] = [
            {'model_id': 'civitai:1681903', 'weight': 2.0},
            {'model_id': 'civitai:1764869', 'weight': 0.75}
        ]

    if request.method == 'POST':
        # Set a 5-minute timeout for the request object itself
        request.environ.get('werkzeug.server.shutdown')

        params, context_update, error = validate_and_extract_params(request.form)
        
        # Update context with form values so user doesn't lose input
        if context_update:
            context.update(context_update)
        
        if error:
            context['error'] = error
            return render_template('index.html', **context)
        
        try:
            response = generate_image_task(params)
            
            if response.status_code == 200:
                saved_paths, filenames = save_images(response, params['prompt'])
                
                context['saved_paths'] = saved_paths
                context['image_filenames'] = filenames
                flash('Images generated successfully!', 'success')
            else:
                context['error'] = f"Error from Modal API: {response.status_code} - {response.text}"
                
        except ValueError as e:
            context['error'] = str(e)
        except Exception as e:
            print(f"Unexpected error: {e}")
            context['error'] = f"An unexpected error occurred: {str(e)}"

    return render_template('index.html', **context)

@app.route('/images/<path:filename>')
def get_image(filename):
    """Serve images from the generated_images directory"""
    try:
        safe_filename = Path(filename).name
        image_path = output_dir / safe_filename
        
        if image_path.exists() and image_path.is_file():
            response = send_file(image_path, mimetype='image/png')
            response.headers['Cache-Control'] = 'public, max-age=3600'
            return response
        else:
            return "Image not found", 404
    except Exception as e:
        print(f"Error serving image {filename}: {str(e)}")
        return "Error serving image", 500

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    context = get_default_context()
    context['error'] = "Page not found. Please go back to the main page."
    return render_template('index.html', **context), 404

@app.errorhandler(500)
def server_error(e):
    context = get_default_context()
    context['error'] = "An internal server error occurred. Please try again later."
    return render_template('index.html', **context), 500

@app.errorhandler(400)
def bad_request(e):
    context = get_default_context()
    context['error'] = "Bad request. Please check your input."
    return render_template('index.html', **context), 400

@app.errorhandler(413)
def request_entity_too_large(e):
    context = get_default_context()
    context['error'] = "The file or input is too large."
    return render_template('index.html', **context), 413

@app.errorhandler(Exception)
def handle_exception(e):
    print(f"Unhandled exception: {str(e)}")
    if isinstance(e, HTTPException):
        return app.handle_http_exception(e)
    
    context = get_default_context()
    context['error'] = f"An unexpected error occurred: {str(e)}"
    return render_template('index.html', **context), 500

if __name__ == '__main__':
    print("=== SDXL Image Generator Server ===")
    print("Server configured with 30-minute request timeout for long-running image generation")
    print("Starting development server...")
    
    app.run(debug=True, port=5000, threaded=True, host='0.0.0.0', use_reloader=True)
