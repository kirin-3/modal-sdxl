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
# This helps prevent connection resets during long-running image generation
WSGIRequestHandler.protocol_version = "HTTP/1.1"  # Use HTTP/1.1 for keep-alive
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
app.secret_key = os.urandom(24)  # Add a secret key for session support

# Configure Flask for long-running requests
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for development
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Allow up to 50 MB requests
app.config['PROPAGATE_EXCEPTIONS'] = True  # Make sure errors are properly handled

# Enhance session security but ensure cookies work in development
app.config.update(
    SESSION_COOKIE_SECURE=False,  # Set to False for development to ensure cookies work on localhost
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=1800,  # 30 minutes
    SESSION_USE_SIGNER=False  # Disable signing for simpler debugging in development
)

# Add request timeout handling (and no-cache headers)
@app.after_request
def add_response_headers(response):
    # Add response headers to help prevent timeouts in proxies/browsers
    response.headers['Connection'] = 'keep-alive'
    response.headers['Keep-Alive'] = 'timeout=1800'  # 30 minutes
    # Add no-cache headers to prevent browser caching issues with redirects/session data
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
    images = []  # For storing multiple images (this is the one we need to populate correctly)
    saved_paths = []  # Initialize empty list for saved paths
    image_filenames = []  # Initialize empty list for image filenames
    force_civitai = True  # Force CivitAI by default
    loras = []
    
    # Pre-filled LoRA values (moved this down slightly for clarity, no functional change)
    loras_from_form = request.form.getlist('loras') # If loras are passed differently
    if not loras_from_form and request.method == 'GET' and not ('generation_data' in session): # only default if not POST and no session restore
        loras = [
            {'model_id': 'civitai:1681903', 'weight': 2.0},
            {'model_id': 'civitai:1764869', 'weight': 0.75}
        ]
    # If loras were restored from session or will be from POST, they'll be handled there.

    if request.method == 'POST':
        try:
            # Set a 5-minute timeout for the request
            request.environ.get('werkzeug.server.shutdown')
            
            prompt = request.form.get('prompt', '')
            negative_prompt = request.form.get('negative_prompt', '')
            seed_input = request.form.get('seed', '')
            model_source = request.form.get('model_source', 'civitai')
            
            # Get batch parameters with additional validation
            try:
                batch_size = int(request.form.get('batch_size', 1))
                if batch_size < 1: batch_size = 1
                elif batch_size > 4: batch_size = 4
            except ValueError: batch_size = 1
                
            try:
                batch_count = int(request.form.get('batch_count', 1))
                if batch_count < 1: batch_count = 1
                elif batch_count > 4: batch_count = 4
            except ValueError: batch_count = 1
            
            # Calculate total images and apply safety limit
            total_images = batch_size * batch_count
            if total_images > 8:
                if batch_size > 2: batch_size = 2
                batch_count = min(batch_count, 4)
                print(f"Reduced batch parameters: size={batch_size}, count={batch_count}")
            
            # Get image dimensions
            try:
                width = int(request.form.get('width', DEFAULT_WIDTH))
                if width < 512 or width > 2048: width = DEFAULT_WIDTH
            except ValueError: width = DEFAULT_WIDTH
                
            try:
                height = int(request.form.get('height', DEFAULT_HEIGHT))
                if height < 512 or height > 2048: height = DEFAULT_HEIGHT
            except ValueError: height = DEFAULT_HEIGHT
                
            # Get inference steps
            try:
                steps = int(request.form.get('steps', DEFAULT_STEPS))
                if steps < 1 or steps > 150: steps = DEFAULT_STEPS
            except ValueError: steps = DEFAULT_STEPS
                
            # Get guidance scale
            try:
                guidance_scale = float(request.form.get('guidance_scale', DEFAULT_GUIDANCE_SCALE))
                if guidance_scale < 1.0 or guidance_scale > 20.0: guidance_scale = DEFAULT_GUIDANCE_SCALE
            except ValueError: guidance_scale = DEFAULT_GUIDANCE_SCALE
                
            # Get CLIP skip
            clip_skip_input = request.form.get('clip_skip', '')
            if clip_skip_input and clip_skip_input.strip():
                try:
                    clip_skip_val = int(clip_skip_input)
                    if 1 <= clip_skip_val <= 4:
                        clip_skip = clip_skip_val
                except ValueError:
                    pass # clip_skip remains None
            
            scheduler_input = request.form.get('scheduler', DEFAULT_SCHEDULER) # Renamed to avoid conflict with outer `scheduler`
            if scheduler_input not in AVAILABLE_SCHEDULERS:
                scheduler_input = DEFAULT_SCHEDULER
            scheduler = scheduler_input # Assign to outer scope variable for template rendering

            # Process LoRAs
            current_loras = [] # Use a temporary list for POST processing
            for i in range(1, 6):
                lora_enabled = request.form.get(f'lora{i}_enabled') == 'on'
                if lora_enabled:
                    lora_source = request.form.get(f'lora{i}_source')
                    lora_id_val = request.form.get(f'lora{i}_id', '').strip()
                    lora_weight_val = request.form.get(f'lora{i}_weight', 0.75)
                    
                    try:
                        lora_weight = float(lora_weight_val)
                        if not (0.1 <= lora_weight <= 2.0): lora_weight = 0.75
                    except ValueError: lora_weight = 0.75
                    
                    if lora_id_val:
                        model_id_formatted = ""
                        if lora_source == 'civitai': model_id_formatted = f"civitai:{lora_id_val}"
                        elif lora_source == 'huggingface': model_id_formatted = f"hf:{lora_id_val}"
                        
                        if model_id_formatted:
                            current_loras.append({'model_id': model_id_formatted, 'weight': lora_weight})
            loras = current_loras # Assign processed LoRAs to the outer scope `loras`

            if model_source == 'huggingface':
                model_id_input = request.form.get('model_id', DEFAULT_MODEL_ID).strip() # Renamed
                model_id = model_id_input or DEFAULT_MODEL_ID
                is_civitai = False
                civitai_id = "" # Clear civitai_id if HF is selected
            else:
                civitai_id_input = request.form.get('civitai_id', '').strip() # Renamed
                if civitai_id_input:
                    model_id = f"civitai:{civitai_id_input}"
                    civitai_id = civitai_id_input # Assign to outer scope
                    is_civitai = True
                else:
                    error = "Please enter a CivitAI model ID"
                    return render_template('index.html', error=error, prompt=prompt, negative_prompt=negative_prompt, seed=seed_input, model_id=model_id, default_model=DEFAULT_MODEL_ID, is_civitai=is_civitai, civitai_id=civitai_id, width=width, height=height, steps=steps, guidance_scale=guidance_scale, clip_skip=clip_skip, scheduler=scheduler, loras=loras, force_civitai=force_civitai, scheduler_options=AVAILABLE_SCHEDULERS, batch_size=batch_size, batch_count=batch_count)

            # Check seed (assign to outer scope `seed` if valid)
            seed_val = None
            if seed_input and seed_input.strip():
                try:
                    seed_val = int(seed_input)
                except ValueError:
                    error = "Seed must be an integer"
                    return render_template('index.html', error=error, prompt=prompt, negative_prompt=negative_prompt, seed=seed_input, model_id=model_id, default_model=DEFAULT_MODEL_ID, is_civitai=is_civitai, civitai_id=civitai_id, width=width, height=height, steps=steps, guidance_scale=guidance_scale, clip_skip=clip_skip, scheduler=scheduler, loras=loras, force_civitai=force_civitai, scheduler_options=AVAILABLE_SCHEDULERS, batch_size=batch_size, batch_count=batch_count)
            seed = seed_val # Assign to outer scope `seed`

            modal_endpoint = os.getenv('MODAL_ENDPOINT', '')
            if not modal_endpoint:
                error = "Modal endpoint URL not configured."
            else:
                try:
                    # Create a shorter-lived session for the API request to avoid server timeouts
                    request_session = requests.Session()
                    
                    # Set up the parameters for the request
                    params = {
                        'prompt': prompt, 'negative_prompt': negative_prompt,
                        'width': width, 'height': height, 'steps': steps,
                        'guidance_scale': guidance_scale, 'batch_size': batch_size,
                        'batch_count': batch_count, 'scheduler': scheduler
                    }
                    
                    # Add optional parameters
                    if seed is not None: params['seed'] = seed
                    if clip_skip is not None: params['clip_skip'] = clip_skip
                    if model_id and (is_civitai or model_id != DEFAULT_MODEL_ID):
                        params['model_id'] = model_id
                    if loras: params['loras'] = json.dumps(loras)
                    
                    # Log what we're doing
                    model_display = civitai_id if is_civitai else (model_id or DEFAULT_MODEL_ID)
                    print(f"Generating SDXL image with model: {model_display}, dimensions: {width}x{height}, steps: {steps}, guidance: {guidance_scale}")
                    print(f"Batch settings: {batch_size} images per batch, {batch_count} batches")
                    if loras:
                        lora_info = []
                        for lora in loras:
                            model_id = lora.get('model_id', '')
                            weight = lora.get('weight', 0)
                            lora_info.append(f"{model_id}:{weight}")
                        print(f"Using LoRAs: {', '.join(lora_info)}")
                    
                    # Calculate an appropriate timeout based on the complexity of the request
                    timeout_seconds = 300 + (batch_count - 1) * 120 + (batch_size - 1) * 60
                    if model_id != DEFAULT_MODEL_ID or is_civitai or loras:
                        timeout_seconds += 180
                    print(f"Using request timeout of {timeout_seconds} seconds")
                    
                    # Set up request headers and session
                    request_session = requests.Session()
                    adapter = requests.adapters.HTTPAdapter(max_retries=3, pool_connections=5, pool_maxsize=10)
                    request_session.mount('http://', adapter)
                    request_session.mount('https://', adapter)
                    headers = {'Connection': 'keep-alive', 'Keep-Alive': 'timeout=600, max=1000', 'User-Agent': 'SDXL-Generator/1.0'}
                    
                    # Make the API request with proper error handling
                    try:
                        print(f"Starting request to Modal endpoint with {timeout_seconds}s timeout")
                        response = request_session.get(modal_endpoint, params=params, timeout=timeout_seconds, headers=headers)
                        response.raise_for_status()
                        print(f"Modal API responded with status code: {response.status_code}")
                    except requests.exceptions.RequestException as req_err:
                        error_message = f"Error from Modal API: {str(req_err)}"
                        if isinstance(req_err, requests.exceptions.Timeout): 
                            error_message = "The request timed out. Try with a smaller batch size or fewer steps."
                        elif isinstance(req_err, requests.exceptions.ConnectionError): 
                            error_message = "Connection error. The server may be temporarily unavailable."
                        error = error_message
                        return render_template('index.html', error=error, prompt=prompt, negative_prompt=negative_prompt, seed=seed, model_id=model_id, default_model=DEFAULT_MODEL_ID, is_civitai=is_civitai, civitai_id=civitai_id, width=width, height=height, steps=steps, guidance_scale=guidance_scale, clip_skip=clip_skip, scheduler=scheduler, batch_size=batch_size, batch_count=batch_count, loras=loras, force_civitai=force_civitai, scheduler_options=AVAILABLE_SCHEDULERS)

                    if response.status_code == 200:
                        print("Received successful 200 response, processing content...")
                        
                        # Check content type
                        content_type = response.headers.get('content-type', '')
                        print(f"Response content type: {content_type}")
                        
                        # Lists to store processed image data
                        processed_saved_paths = []
                        processed_image_filenames = []

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        prompt_slug = slugify(prompt[:50])
                        
                        # Process response based on content type
                        if 'application/json' in content_type:
                            try:
                                # For JSON responses (multiple images)
                                resp_data = response.json()
                                if 'images' in resp_data:
                                    num_images_resp = len(resp_data['images'])
                                    print(f"Received {num_images_resp} images in JSON response")
                                    
                                    for i, b64_img in enumerate(resp_data['images']):
                                        try:
                                            img_bytes = base64.b64decode(b64_img)
                                            # Create unique filename for each image
                                            filename = f"{timestamp}_{prompt_slug}_{i+1}.png"
                                            save_path = output_dir / filename
                                            
                                            # Save the image to disk
                                            with open(save_path, 'wb') as f:
                                                f.write(img_bytes)
                                            
                                            # Add path and filename to lists
                                            processed_saved_paths.append(str(save_path))
                                            processed_image_filenames.append(filename)
                                            print(f"Saved image to {save_path}")
                                        except Exception as img_err:
                                            print(f"Error processing image {i+1}: {str(img_err)}")
                                    
                                    if not processed_saved_paths:
                                        error = "No images were returned or processed correctly from JSON"
                            except Exception as e:
                                error = f"Error processing batch images: {str(e)}"
                        else:
                            # For binary responses (single image)
                            try:
                                image_data = response.content
                                
                                # Create filename for the single image
                                filename = f"{timestamp}_{prompt_slug}.png"
                                save_path = output_dir / filename
                                
                                # Save the image to disk
                                with open(save_path, 'wb') as f:
                                    f.write(image_data)
                                
                                # Add path and filename to lists
                                processed_saved_paths.append(str(save_path))
                                processed_image_filenames.append(filename)
                                print(f"Saved image to {save_path}")
                            except Exception as e:
                                error = f"Error saving image: {str(e)}"

                        if error:
                            # If error during processing, render error template
                            return render_template('index.html', error=error, prompt=prompt, negative_prompt=negative_prompt, seed=seed, model_id=model_id, default_model=DEFAULT_MODEL_ID, is_civitai=is_civitai, civitai_id=civitai_id, width=width, height=height, steps=steps, guidance_scale=guidance_scale, clip_skip=clip_skip, scheduler=scheduler, batch_size=batch_size, batch_count=batch_count, loras=loras, force_civitai=force_civitai, scheduler_options=AVAILABLE_SCHEDULERS)

                        # Success - show success message
                        flash('Images generated successfully!', 'success')
                        
                        # Render the template with the results
                        return render_template('index.html', 
                                               image=None,  # No binary data
                                               images=None,  # No binary data
                                               saved_paths=processed_saved_paths,
                                               image_filenames=processed_image_filenames,
                                               output_directory_display=str(output_dir.resolve()),
                                               error=None,
                                               prompt=prompt, negative_prompt=negative_prompt,
                                               seed=seed, model_id=model_id, default_model=DEFAULT_MODEL_ID,
                                               is_civitai=is_civitai, civitai_id=civitai_id,
                                               width=width, height=height, steps=steps,
                                               guidance_scale=guidance_scale, clip_skip=clip_skip,
                                               scheduler=scheduler, scheduler_options=AVAILABLE_SCHEDULERS,
                                               batch_size=batch_size, batch_count=batch_count,
                                               loras=loras, force_civitai=force_civitai)
                    else:
                        error = f"Error from Modal API: {response.status_code} - {response.text}"
                except Exception as e:
                    print(f"Exception during request processing: {str(e)}")
                    error = f"Error connecting to Modal API: {str(e)}"
        except Exception as e:
            print(f"Unhandled exception in route: {str(e)}")
            error = f"An unexpected error occurred: {str(e)}"
    
    # Common render path for GET or POST with errors
    return render_template('index.html', 
                          image=image, 
                          images=images, 
                          saved_paths=saved_paths,
                          image_filenames=image_filenames,
                          output_directory_display=str(output_dir.resolve()),
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

@app.route('/images/<path:filename>')
def get_image(filename):
    """Serve images from the generated_images directory"""
    try:
        # Sanitize filename to prevent path traversal
        safe_filename = Path(filename).name
        image_path = output_dir / safe_filename
        
        if image_path.exists() and image_path.is_file():
            # Set higher cache max-age for faster browsing
            response = send_file(image_path, mimetype='image/png')
            response.headers['Cache-Control'] = 'public, max-age=3600'
            return response
        else:
            print(f"Image not found: {safe_filename}")
            return "Image not found", 404
    except Exception as e:
        print(f"Error serving image {filename}: {str(e)}")
        return "Error serving image", 500

# Error handlers for common HTTP exceptions
@app.errorhandler(404)
def page_not_found(e):
    # Provide all necessary default values to prevent template errors
    return render_template('index.html', 
                          error="Page not found. Please go back to the main page.", 
                          prompt="", 
                          negative_prompt="", 
                          seed=None, 
                          model_id=DEFAULT_MODEL_ID,
                          default_model=DEFAULT_MODEL_ID,
                          is_civitai=False,
                          civitai_id="1637364", 
                          width=DEFAULT_WIDTH, 
                          height=DEFAULT_HEIGHT, 
                          steps=DEFAULT_STEPS, 
                          guidance_scale=DEFAULT_GUIDANCE_SCALE,
                          clip_skip=None, 
                          scheduler=DEFAULT_SCHEDULER, 
                          scheduler_options=AVAILABLE_SCHEDULERS,
                          batch_size=1, 
                          batch_count=1,
                          loras=[], 
                          force_civitai=True,
                          saved_paths=[],
                          image_filenames=[],
                          image=None,
                          images=[],
                          output_directory_display=str(output_dir.resolve())), 404

@app.errorhandler(500)
def server_error(e):
    # Provide all necessary default values to prevent template errors
    return render_template('index.html', 
                          error="An internal server error occurred. Please try again later.", 
                          prompt="", 
                          negative_prompt="", 
                          seed=None, 
                          model_id=DEFAULT_MODEL_ID,
                          default_model=DEFAULT_MODEL_ID,
                          is_civitai=False,
                          civitai_id="1637364", 
                          width=DEFAULT_WIDTH, 
                          height=DEFAULT_HEIGHT, 
                          steps=DEFAULT_STEPS, 
                          guidance_scale=DEFAULT_GUIDANCE_SCALE,
                          clip_skip=None, 
                          scheduler=DEFAULT_SCHEDULER, 
                          scheduler_options=AVAILABLE_SCHEDULERS,
                          batch_size=1, 
                          batch_count=1,
                          loras=[], 
                          force_civitai=True,
                          saved_paths=[],
                          image_filenames=[],
                          image=None,
                          images=[],
                          output_directory_display=str(output_dir.resolve())), 500

@app.errorhandler(400)
def bad_request(e):
    # Provide all necessary default values to prevent template errors
    return render_template('index.html', 
                          error="Bad request. Please check your input.", 
                          prompt="", 
                          negative_prompt="", 
                          seed=None, 
                          model_id=DEFAULT_MODEL_ID,
                          default_model=DEFAULT_MODEL_ID,
                          is_civitai=False,
                          civitai_id="1637364", 
                          width=DEFAULT_WIDTH, 
                          height=DEFAULT_HEIGHT, 
                          steps=DEFAULT_STEPS, 
                          guidance_scale=DEFAULT_GUIDANCE_SCALE,
                          clip_skip=None, 
                          scheduler=DEFAULT_SCHEDULER, 
                          scheduler_options=AVAILABLE_SCHEDULERS,
                          batch_size=1, 
                          batch_count=1,
                          loras=[], 
                          force_civitai=True,
                          saved_paths=[],
                          image_filenames=[],
                          image=None,
                          images=[],
                          output_directory_display=str(output_dir.resolve())), 400

@app.errorhandler(413)
def request_entity_too_large(e):
    # Provide all necessary default values to prevent template errors
    return render_template('index.html', 
                          error="The file or input is too large.", 
                          prompt="", 
                          negative_prompt="", 
                          seed=None, 
                          model_id=DEFAULT_MODEL_ID,
                          default_model=DEFAULT_MODEL_ID,
                          is_civitai=False,
                          civitai_id="1637364", 
                          width=DEFAULT_WIDTH, 
                          height=DEFAULT_HEIGHT, 
                          steps=DEFAULT_STEPS, 
                          guidance_scale=DEFAULT_GUIDANCE_SCALE,
                          clip_skip=None, 
                          scheduler=DEFAULT_SCHEDULER, 
                          scheduler_options=AVAILABLE_SCHEDULERS,
                          batch_size=1, 
                          batch_count=1,
                          loras=[], 
                          force_civitai=True,
                          saved_paths=[],
                          image_filenames=[],
                          image=None,
                          images=[],
                          output_directory_display=str(output_dir.resolve())), 413

# Add a catch-all error handler
@app.errorhandler(Exception)
def handle_exception(e):
    # Print the exception to console for debugging
    print(f"Unhandled exception: {str(e)}")
    
    # If it's an HTTP exception, pass it to the specific handler
    if isinstance(e, HTTPException):
        return app.handle_http_exception(e)
    
    # For all other exceptions, return a generic 500 error
    return render_template('index.html', 
                          error=f"An unexpected error occurred: {str(e)}", 
                          prompt="", 
                          negative_prompt="", 
                          seed=None, 
                          model_id=DEFAULT_MODEL_ID,
                          default_model=DEFAULT_MODEL_ID,
                          is_civitai=False,
                          civitai_id="1637364", 
                          width=DEFAULT_WIDTH, 
                          height=DEFAULT_HEIGHT, 
                          steps=DEFAULT_STEPS, 
                          guidance_scale=DEFAULT_GUIDANCE_SCALE,
                          clip_skip=None, 
                          scheduler=DEFAULT_SCHEDULER, 
                          scheduler_options=AVAILABLE_SCHEDULERS,
                          batch_size=1, 
                          batch_count=1,
                          loras=[], 
                          force_civitai=True,
                          saved_paths=[],
                          image_filenames=[],
                          image=None,
                          images=[],
                          output_directory_display=str(output_dir.resolve())), 500

if __name__ == '__main__':
    print("=== SDXL Image Generator Server ===")
    print("Server configured with 30-minute request timeout for long-running image generation")
    print("NOTE: For production use, consider using Gunicorn with:")
    print("      gunicorn --workers=2 --timeout=1800 local_server:app")
    print("Starting development server...")
    
    app.run(debug=True, port=5000, threaded=True, host='0.0.0.0', use_reloader=True)