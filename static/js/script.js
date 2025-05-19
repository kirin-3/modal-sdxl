// Modular JS components
const UIComponents = {
    promptHistory: {
        init() {
            this.renderHistory();
            this.bindEvents();
        },
        
        bindEvents() {
            // Add event listeners for saving prompts to history
            document.querySelectorAll('.generate-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    const prompt = document.getElementById('prompt').value;
                    const negativePrompt = document.getElementById('negative_prompt').value;
                    if (prompt) {
                        this.addToHistory(prompt, negativePrompt);
                    }
                });
            });
            
            // Add event listener for the history panel toggle
            document.getElementById('history-toggle')?.addEventListener('click', () => {
                const panel = document.getElementById('history-panel');
                if (panel) {
                    panel.classList.toggle('history-panel-open');
                }
            });
            
            // Add event listener for the close button
            document.getElementById('history-close-btn')?.addEventListener('click', () => {
                const panel = document.getElementById('history-panel');
                if (panel) {
                    panel.classList.remove('history-panel-open');
                }
            });
            
            // Add event listener to close history panel when clicking anywhere else
            document.addEventListener('click', (e) => {
                const panel = document.getElementById('history-panel');
                const toggle = document.getElementById('history-toggle');
                
                if (panel && panel.classList.contains('history-panel-open') &&
                    !panel.contains(e.target) && 
                    toggle && !toggle.contains(e.target)) {
                    panel.classList.remove('history-panel-open');
                }
            });
        },
        
        addToHistory(prompt, negativePrompt) {
            let history = this.getHistory();
            const timestamp = new Date().toISOString();
            
            // Add to the beginning of the array
            history.unshift({
                prompt,
                negativePrompt,
                timestamp,
                favorite: false
            });
            
            // Keep only the latest 20 items
            if (history.length > 20) {
                history = history.slice(0, 20);
            }
            
            localStorage.setItem('prompt-history', JSON.stringify(history));
            this.renderHistory();
        },
        
        getHistory() {
            const history = localStorage.getItem('prompt-history');
            return history ? JSON.parse(history) : [];
        },
        
        toggleFavorite(index) {
            const history = this.getHistory();
            if (history[index]) {
                history[index].favorite = !history[index].favorite;
                localStorage.setItem('prompt-history', JSON.stringify(history));
                this.renderHistory();
            }
        },
        
        usePrompt(index) {
            const history = this.getHistory();
            if (history[index]) {
                document.getElementById('prompt').value = history[index].prompt;
                document.getElementById('negative_prompt').value = history[index].negativePrompt || '';
                saveFormData(); // Save to form persistence
            }
        },
        
        renderHistory() {
            const historyContainer = document.getElementById('prompt-history-container');
            if (!historyContainer) return;
            
            // Clear container
            historyContainer.innerHTML = '';
            
            const history = this.getHistory();
            const favorites = history.filter(item => item.favorite);
            
            // Add favorites section if there are favorites
            if (favorites.length > 0) {
                const favoritesSection = document.createElement('div');
                favoritesSection.classList.add('history-section');
                
                const favoritesHeader = document.createElement('h4');
                favoritesHeader.textContent = 'Favorites';
                favoritesSection.appendChild(favoritesHeader);
                
                favorites.forEach((item, originalIndex) => {
                    const index = history.findIndex(h => h.timestamp === item.timestamp);
                    favoritesSection.appendChild(this.createHistoryItem(item, index));
                });
                
                historyContainer.appendChild(favoritesSection);
            }
            
            // Add recent prompts section
            const recentSection = document.createElement('div');
            recentSection.classList.add('history-section');
            
            const recentHeader = document.createElement('h4');
            recentHeader.textContent = 'Recent Prompts';
            recentSection.appendChild(recentHeader);
            
            if (history.length === 0) {
                const emptyMessage = document.createElement('p');
                emptyMessage.textContent = 'No prompt history yet. Generate some images!';
                recentSection.appendChild(emptyMessage);
            } else {
                history.forEach((item, index) => {
                    recentSection.appendChild(this.createHistoryItem(item, index));
                });
            }
            
            historyContainer.appendChild(recentSection);
        },
        
        createHistoryItem(item, index) {
            const itemElement = document.createElement('div');
            itemElement.classList.add('history-item');
            
            const promptText = document.createElement('p');
            promptText.classList.add('history-prompt');
            promptText.textContent = item.prompt.length > 60 ? 
                item.prompt.substring(0, 60) + '...' : 
                item.prompt;
            itemElement.appendChild(promptText);
            
            const date = new Date(item.timestamp);
            const dateText = document.createElement('small');
            dateText.textContent = date.toLocaleString();
            itemElement.appendChild(dateText);
            
            const buttonsContainer = document.createElement('div');
            buttonsContainer.classList.add('history-buttons');
            
            const useButton = document.createElement('button');
            useButton.classList.add('history-btn');
            useButton.textContent = 'Use';
            useButton.addEventListener('click', () => this.usePrompt(index));
            buttonsContainer.appendChild(useButton);
            
            const favoriteButton = document.createElement('button');
            favoriteButton.classList.add('history-btn', 'favorite-btn');
            favoriteButton.innerHTML = item.favorite ? '★' : '☆';
            favoriteButton.addEventListener('click', () => this.toggleFavorite(index));
            buttonsContainer.appendChild(favoriteButton);
            
            itemElement.appendChild(buttonsContainer);
            
            return itemElement;
        }
    },
    
    // LoRA preview has been removed
};

// Function to open the image modal
function enlargeImage(img) {
    const modal = document.getElementById('imageModal');
    const enlargedImg = document.getElementById('enlargedImage');
    modal.style.display = 'block';
    enlargedImg.src = img.src;
}

// Function to close the image modal
function closeModal() {
    document.getElementById('imageModal').style.display = 'none';
}

// Close modal when clicking outside the image
window.addEventListener('click', function(event) {
    const modal = document.getElementById('imageModal');
    if (event.target === modal) {
        closeModal();
    }
});

// Handle escape key to close modal
window.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        closeModal();
    }
});

// Copy prompt to clipboard function
function copyToClipboard(text) {
    // Create a temporary textarea element to hold the text
    const textarea = document.createElement('textarea');
    textarea.value = text;
    document.body.appendChild(textarea);
    
    // Select and copy the text
    textarea.select();
    document.execCommand('copy');
    
    // Remove the textarea
    document.body.removeChild(textarea);
    
    // Show a toast notification
    showToast('Copied to clipboard!');
}

// Simple toast notification function
function showToast(message) {
    // Create toast element if it doesn't exist
    let toast = document.getElementById('toast-notification');
    if (!toast) {
        toast = document.createElement('div');
        toast.id = 'toast-notification';
        document.body.appendChild(toast);
    }
    
    // Set message and show toast
    toast.textContent = message;
    toast.classList.add('show-toast');
    
    // Hide toast after 3 seconds
    setTimeout(() => {
        toast.classList.remove('show-toast');
    }, 3000);
}

// Form persistence using localStorage
const formInputs = document.querySelectorAll('input, textarea, select');
const formId = 'sdxl-generator-form';
const clearFormBtn = document.getElementById('clear-form-btn');

// Connection status handler
const ConnectionStatus = {
    init() {
        this.statusElement = null;
        this.createStatusElement();
    },
    
    createStatusElement() {
        // Create status element if it doesn't exist
        if (!this.statusElement) {
            this.statusElement = document.createElement('div');
            this.statusElement.className = 'connection-status';
            this.statusElement.innerHTML = '<span class="connection-status-icon"></span><span class="connection-status-text"></span>';
            document.body.appendChild(this.statusElement);
        }
    },
    
    show(message, type = 'info') {
        this.createStatusElement();
        
        // Clear any existing classes
        this.statusElement.classList.remove('connecting', 'error', 'success');
        
        // Set icon and class based on type
        let icon = '';
        if (type === 'connecting') {
            icon = '⏳';
            this.statusElement.classList.add('connecting');
        } else if (type === 'error') {
            icon = '❌';
            this.statusElement.classList.add('error');
        } else if (type === 'success') {
            icon = '✓';
            this.statusElement.classList.add('success');
        }
        
        // Update content
        this.statusElement.querySelector('.connection-status-icon').textContent = icon;
        this.statusElement.querySelector('.connection-status-text').textContent = message;
        
        // Show the element
        this.statusElement.classList.add('visible');
        
        // Auto-hide success and info messages after 3 seconds
        if (type === 'success' || type === 'info') {
            setTimeout(() => {
                this.hide();
            }, 3000);
        }
    },
    
    hide() {
        if (this.statusElement) {
            this.statusElement.classList.remove('visible');
        }
    }
};

// Enhanced error handling
const ErrorHandler = {
    init() {
        // Add event listeners to close error panels
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('error-panel-close')) {
                const panel = e.target.closest('.error-panel');
                if (panel) {
                    panel.remove();
                }
            }
        });
    },
    
    showError(message, title = 'Error', tip = null) {
        // Create error panel
        const errorPanel = document.createElement('div');
        errorPanel.className = 'error-panel';
        
        let html = `
            <button class="error-panel-close" aria-label="Dismiss error">&times;</button>
            <h4 class="error-panel-title">${title}</h4>
            <div class="error-panel-message">${message}</div>
        `;
        
        if (tip) {
            html += `<div class="error-panel-tip">${tip}</div>`;
        }
        
        errorPanel.innerHTML = html;
        
        // Add retry button for connection errors
        if (title.includes('Connection') || message.includes('connection') || message.includes('reset')) {
            const retryButton = document.createElement('button');
            retryButton.className = 'retry-button';
            retryButton.textContent = 'Try Again with Reduced Settings';
            retryButton.addEventListener('click', () => {
                // Reduce batch settings
                const batchSizeInput = document.getElementById('batch_size');
                const batchCountInput = document.getElementById('batch_count');
                
                if (batchSizeInput && parseInt(batchSizeInput.value) > 1) {
                    batchSizeInput.value = 1;
                }
                
                if (batchCountInput && parseInt(batchCountInput.value) > 1) {
                    batchCountInput.value = 1;
                }
                
                // Update total images count
                updateTotalImages();
                
                // Remove the error panel
                errorPanel.remove();
                
                // Submit the form
                document.getElementById('generate-form').submit();
            });
            
            errorPanel.appendChild(retryButton);
        }
        
        // Insert at the top of the page
        const container = document.querySelector('.container');
        if (container && container.firstChild) {
            container.insertBefore(errorPanel, container.querySelector('h1').nextSibling);
        } else {
            document.body.prepend(errorPanel);
        }
    },
    
    handleNetworkError(error) {
        console.error('Network error:', error);
        
        let title = 'Connection Error';
        let message = 'Could not connect to the server.';
        let tip = 'The server might be temporarily unavailable. Try reducing batch size or using simpler settings.';
        
        if (error.message.includes('reset')) {
            message = 'The connection was reset by the server.';
            tip = 'This typically happens when generating large batches or complex images. Try reducing batch size, using fewer steps, or simplifying your prompt.';
        } else if (error.message.includes('timeout')) {
            message = 'The request timed out.';
            tip = 'The server is taking too long to respond. Try reducing batch size or steps.';
        }
        
        this.showError(message, title, tip);
    }
};

// Add network and error monitoring to detect connection resets
const NetworkMonitor = {
    init() {
        // Watch for online/offline events
        window.addEventListener('online', () => {
            ConnectionStatus.show('Back online', 'success');
        });
        
        window.addEventListener('offline', () => {
            ConnectionStatus.show('Connection lost', 'error');
        });
        
        // Detect aborted connections that might lead to ERR_CONNECTION_RESET
        window.addEventListener('unhandledrejection', (event) => {
            if (event.reason && 
                typeof event.reason.message === 'string' &&
                (event.reason.message.includes('abort') || 
                event.reason.message.includes('reset') || 
                event.reason.message.includes('network'))) {
                
                ErrorHandler.handleNetworkError(event.reason);
                event.preventDefault(); // Prevent default error handling
            }
        });
    }
};

// Initialize additional components when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    loadFormData();
    updateTotalImages(); // Initialize total images counter
    
    // Hide clear button if no saved data
    if (!localStorage.getItem(formId)) {
        clearFormBtn.style.display = 'none';
    }
    
    // Initialize UI components
    UIComponents.promptHistory.init();
    
    // Initialize new components
    ConnectionStatus.init();
    ErrorHandler.init();
    NetworkMonitor.init();
});

// Save form data as user types/changes values
formInputs.forEach(input => {
    input.addEventListener('change', saveFormData);
    if (input.tagName === 'TEXTAREA' || input.type === 'text' || input.type === 'number') {
        input.addEventListener('input', saveFormData);
    }
});

// Clear saved form data
clearFormBtn.addEventListener('click', function() {
    localStorage.removeItem(formId);
    clearFormBtn.style.display = 'none';
    alert('Saved form data cleared!');
});

function saveFormData() {
    const formData = {};
    formInputs.forEach(input => {
        const name = input.name;
        if (!name) return;
        
        if (input.type === 'checkbox') {
            formData[name] = input.checked;
        } else if (input.type === 'radio') {
            if (input.checked) {
                formData[name] = input.value;
            }
        } else {
            formData[name] = input.value;
        }
    });
    
    localStorage.setItem(formId, JSON.stringify(formData));
    clearFormBtn.style.display = 'block';
}

function loadFormData() {
    const savedData = localStorage.getItem(formId);
    if (!savedData) return;
    
    const formData = JSON.parse(savedData);
    formInputs.forEach(input => {
        const name = input.name;
        if (!name || !(name in formData)) return;
        
        if (input.type === 'checkbox') {
            input.checked = formData[name];
        } else if (input.type === 'radio') {
            input.checked = (input.value === formData[name]);
        } else {
            input.value = formData[name];
        }
    });
    
    // Make sure to update UI after loading saved data
    toggleModelSource(document.querySelector('input[name="model_source"]:checked').value);
}

// Total images counter functionality
const batchSizeInput = document.getElementById('batch_size');
const batchCountInput = document.getElementById('batch_count');
const totalImagesDisplay = document.getElementById('total-images-info');

function updateTotalImages() {
    const size = parseInt(batchSizeInput.value) || 1;
    const count = parseInt(batchCountInput.value) || 1;
    // Each batch generates exactly 'size' images, and we run 'count' batches
    // The total is simply size * count (not size^2 * count)
    totalImagesDisplay.textContent = `Total images to generate: ${size * count}`;
}

batchSizeInput.addEventListener('input', updateTotalImages);
batchCountInput.addEventListener('input', updateTotalImages);

// Form submission handling
document.getElementById('generate-form').addEventListener('submit', function() {
    // Show loading indicator before traditional form submit
    document.getElementById('loading').style.display = 'block';
    
    // Change button text to "Generating..."
    const generateBtns = document.querySelectorAll('.generate-btn');
    generateBtns.forEach(btn => {
        btn.textContent = "Generating...";
        btn.disabled = true;
    });
    
    // Save form data to localStorage before submission
    saveFormData();
    
    // Get batch parameters to show appropriate message
    const batchSize = parseInt(document.getElementById('batch_size').value) || 1;
    const batchCount = parseInt(document.getElementById('batch_count').value) || 1;
    const isBatch = batchSize > 1 || batchCount > 1;
    const timeoutMinutes = isBatch ? 20 : 10;
    
    // Show timeout information
    // Do not edit wording here
    ConnectionStatus.show(`Processing request... (up to ${timeoutMinutes} minutes for large batches)`, 'connecting');
    
    // Allow form to submit naturally - no preventDefault()
    // Browser will handle the redirect to the results page
    return true;
});



// Toggle model source display
function toggleModelSource(source) {
    if (source === 'huggingface') {
        document.getElementById('huggingface-options').style.display = 'block';
        document.getElementById('civitai-options').style.display = 'none';
        document.getElementById('civitai_id').removeAttribute('required');
    } else {
        document.getElementById('huggingface-options').style.display = 'none';
        document.getElementById('civitai-options').style.display = 'block';
        document.getElementById('civitai_id').setAttribute('required', 'required');
    }
}

// Set dimensions from presets
function setDimensions(width, height) {
    document.getElementById('width').value = width;
    document.getElementById('height').value = height;
    saveFormData(); // Save form data after setting dimensions
}

// Scroll to result on page load and reset UI elements
window.onload = function() {
    // Always hide loading indicator
    const loadingDiv = document.getElementById('loading');
    if (loadingDiv) {
        loadingDiv.style.display = 'none';
    }

    // Always reset generate buttons
    const generateBtns = document.querySelectorAll('.generate-btn');
    generateBtns.forEach(btn => {
        if (btn) { // Check if button exists
            btn.textContent = "Generate Image";
            btn.disabled = false;
        }
    });

    // Always clear any progress timer
    if (window.progressTimer) {
        clearInterval(window.progressTimer);
        window.progressTimer = null; // Clear the variable itself
    }

    // If there's a result image, scroll to it
    const resultDiv = document.getElementById('result');
    if (resultDiv) {
        resultDiv.scrollIntoView({behavior: 'smooth'});
        // Success notification is handled by Flask flash messages
    }
} 