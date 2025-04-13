from fastapi import FastAPI
from pydantic import BaseModel
import torch
import os
from typing import Optional

# Import custom modules from your project.
# BiasClassifier: The custom model wrapper for bias classification.
# predict_bias: Function that performs inference using the trained model.
# swap_demographics_spacy and associated dictionaries: Functions & mappings for text swapping.
from ml.scripts.classifier import BiasClassifier
from ml.scripts.inference import predict_bias
from ml.scripts.advanced_swaps import (
    swap_demographics_spacy,
    NAME_SWAPS_MALE_FEMALE,
    PRONOUN_SWAPS_M2F,
    LASTNAME_SWAPS,
)
# BiasDataset import is here if you need it later (for any data-related endpoints).
from ml.scripts.train import BiasDataset

############################################
#        CONFIGURATION & MODEL LOADING     #
############################################

# Path to the saved model checkpoint. Can be overridden by an environment variable.
MODEL_PATH = os.getenv("MODEL_PATH", "ml/models/bias_classifier.pt")
# Model name must match the one used in training.
MODEL_NAME = "distilbert-base-uncased"

# Initialize the FastAPI app with metadata.
app = FastAPI(
    title="BiasGPT Backend",
    description="FastAPI for demographic swapping and bias classification",
    version="0.1.0"
)

# Instantiate a global classifier model.
classifier_model = BiasClassifier(model_name=MODEL_NAME, num_labels=2)
# Load the model weights from the specified file.
classifier_model.load(MODEL_PATH)
# Move model to available device (GPU or CPU).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier_model.to(device)
classifier_model.eval()  # Set model to evaluation mode.

# Add Cross-Origin Resource Sharing (CORS) middleware to allow requests from the frontend.
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

############################################
#        Pydantic Models for Requests      #
############################################

class SwapRequest(BaseModel):
    """
    Request model for swapping text.
    - text: The input text for which demographic swaps will be applied.
    """
    text: str

class ClassifyRequest(BaseModel):
    """
    Request model for bias classification.
    - original_text: The original (pre-swap) text.
    - swapped_text: The demographically swapped text.
    """
    original_text: str
    swapped_text: str

class FullAnalyzeRequest(BaseModel):
    """
    Request model for performing a full analysis:
      - Performs advanced swapping.
      - Optionally performs a GPT generation step (if implemented).
      - Classifies the bias of the text change.
    """
    text: str
    do_gpt: bool = False  # Set to True to run additional GPT-based generation (currently a placeholder).

############################################
#               API ROUTES                 #
############################################

@app.get("/")
def root():
    """
    Root endpoint that returns a welcome message.
    """
    return {"message": "Welcome to BiasGPT FastAPI backend!"}

@app.post("/swap_text")
def swap_text(request: SwapRequest):
    """
    Apply advanced demographic swaps to the input text.
    Uses predefined swap dictionaries (male-to-female name/pronoun, last name swaps).
    
    Request Body:
      - text: the text to swap.

    Returns:
      - The original text and its swapped version.
    """
    swapped = swap_demographics_spacy(
        text=request.text,
        name_swaps=NAME_SWAPS_MALE_FEMALE,
        pronoun_swaps=PRONOUN_SWAPS_M2F,
        last_name_swaps=LASTNAME_SWAPS
    )
    return {"original_text": request.text, "swapped_text": swapped}

@app.post("/classify_bias")
def classify_bias(request: ClassifyRequest):
    """
    Classify bias between two texts (original and swapped).
    
    Request Body:
      - original_text: the original input text.
      - swapped_text: the demographically swapped text.
    
    Returns:
      - A label (0: Neutral, 1: Biased) and a human-readable prediction.
    """
    label = predict_bias(
        original_text=request.original_text,
        swapped_text=request.swapped_text,
        model_path=MODEL_PATH
    )
    return {
        "label": label,
        "prediction": "Biased" if label == 1 else "Neutral"
    }

@app.post("/swap_and_analyze")
def swap_and_analyze(request: FullAnalyzeRequest):
    """
    Full analysis endpoint that:
      1. Applies demographic swapping.
      2. Optionally (if enabled) performs GPT generation (placeholder).
      3. Classifies the bias between the original and swapped text.

    Request Body:
      - text: the input text.
      - do_gpt: (optional) boolean flag to perform GPT generation.

    Returns:
      - Original and swapped texts.
      - The bias classification and prediction.
      - If GPT generation is enabled, includes the GPT output.
    """
    # 1. Apply swapping.
    swapped = swap_demographics_spacy(
        text=request.text,
        name_swaps=NAME_SWAPS_MALE_FEMALE,
        pronoun_swaps=PRONOUN_SWAPS_M2F,
        last_name_swaps=LASTNAME_SWAPS
    )

    # 2. (Optional) GPT generation placeholder.
    gpt_output = ""
    if request.do_gpt:
        # Implement GPT call if needed; using placeholder text for now.
        gpt_output = "[GPT response placeholder]"

    # 3. Perform bias classification.
    label = predict_bias(
        original_text=request.text,
        swapped_text=swapped,
        model_path=MODEL_PATH
    )

    # Assemble the complete result.
    result = {
        "original_text": request.text,
        "swapped_text": swapped,
        "label": label,
        "prediction": "Biased" if label == 1 else "Neutral"
    }
    if request.do_gpt:
        result["gpt_output"] = gpt_output

    return result

@app.post("/reload_model")
def reload_model():
    """
    Reload the classifier model.
    
    This endpoint re-instantiates the model and loads the latest saved weights from disk.
    Useful for hot reloading when a new model is deployed.
    
    Returns:
      - A status message indicating if the reload was successful.
    """
    global classifier_model  # Declare as global to modify the global instance.
    
    # Re-instantiate the classifier.
    classifier_model = BiasClassifier(model_name=MODEL_NAME, num_labels=2)
    # Load the same model checkpoint used during training.
    classifier_model.load("ml/models/bias_classifier.pt")
    
    return {"status": "reloaded"}

############################################
#       INSTRUCTIONS TO RUN THE API        #
############################################

"""
To run the FastAPI backend:
1. Install dependencies:
    pip install fastapi uvicorn
2. Start the server:
    uvicorn fastapi_backend:app --reload
3. Open your browser to:
    http://127.0.0.1:8000/docs
   to explore the interactive API documentation.
"""
