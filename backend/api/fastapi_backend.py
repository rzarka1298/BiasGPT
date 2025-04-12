from fastapi import FastAPI, Body
from pydantic import BaseModel
import torch
import os
from typing import Optional

# We'll import the custom modules we created
from ml.scripts.classifier import BiasClassifier
from ml.scripts.inference import predict_bias
from ml.scripts.advanced_swaps import swap_demographics_spacy, NAME_SWAPS_MALE_FEMALE, PRONOUN_SWAPS_M2F, LASTNAME_SWAPS
from ml.scripts.train import BiasDataset  # If we need dataset structures


"""
FastAPI backend to:
1. Swap text with advanced demographic changes.
2. Classify bias between original + swapped text.
3. Optionally integrate GPT calls (if we want to generate text on the fly).
"""

# ============================================================
#        1. Setup & Model Loading
# ============================================================

MODEL_PATH = os.getenv("MODEL_PATH", "ml/models/bias_classifier.pt")
MODEL_NAME = "distilbert-base-uncased"  # Should match your training

# Initialize the app
app = FastAPI(
    title="BiasGPT Backend",
    description="FastAPI for demographic swapping and bias classification",
    version="0.1.0"
)

# Create a global classifier instance
classifier_model = BiasClassifier(model_name=MODEL_NAME, num_labels=2)

# Load the saved weights
classifier_model.load(MODEL_PATH)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier_model.to(device)
classifier_model.eval()

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
#        2. Pydantic Models for Request & Response
# ============================================================

class SwapRequest(BaseModel):
    text: str
    # Optionally specify which type of swap or custom dict
    # For now, let's keep it simple: we do male->female as an example

class ClassifyRequest(BaseModel):
    original_text: str
    swapped_text: str


# ============================================================
#        3. Routes
# ============================================================

@app.get("/")
def root():
    return {"message": "Welcome to BiasGPT FastAPI backend!"}


@app.post("/swap_text")
def swap_text(request: SwapRequest):
    """Apply advanced demographic swaps to the input text."""
    swapped = swap_demographics_spacy(
        text=request.text,
        name_swaps=NAME_SWAPS_MALE_FEMALE,
        pronoun_swaps=PRONOUN_SWAPS_M2F,
        last_name_swaps=LASTNAME_SWAPS
    )
    return {"original_text": request.text, "swapped_text": swapped}


@app.post("/classify_bias")
def classify_bias(request: ClassifyRequest):
    """Use the trained classifier to determine if text changes indicate bias (1) or neutral (0)."""
    label = predict_bias(
        original_text=request.original_text,
        swapped_text=request.swapped_text,
        model_path=MODEL_PATH
    )
    return {
        "label": label,
        "prediction": "Biased" if label == 1 else "Neutral"
    }


"""
Optionally, if you want an endpoint that does everything in one go:
1. Takes original text
2. Performs advanced swap
3. (Optionally) calls GPT to get a generation if needed
4. Classifies the difference
"""

class FullAnalyzeRequest(BaseModel):
    text: str
    do_gpt: bool = False

@app.post("/swap_and_analyze")
def swap_and_analyze(request: FullAnalyzeRequest):
    # 1. Swap text
    swapped = swap_demographics_spacy(
        text=request.text,
        name_swaps=NAME_SWAPS_MALE_FEMALE,
        pronoun_swaps=PRONOUN_SWAPS_M2F,
        last_name_swaps=LASTNAME_SWAPS
    )

    # 2. (Optional) GPT generation step
    gpt_output = ""  # placeholder
    if request.do_gpt:
        # You could implement GPT calls here if needed
        # For demonstration, we just skip it.
        gpt_output = "[GPT response placeholder]"

    # 3. Classify
    label = predict_bias(
        original_text=request.text,
        swapped_text=swapped,
        model_path=MODEL_PATH
    )

    result = {
        "original_text": request.text,
        "swapped_text": swapped,
        "label": label,
        "prediction": "Biased" if label == 1 else "Neutral"
    }

    if request.do_gpt:
        result["gpt_output"] = gpt_output

    return result


"""
Now, to run this:
1. pip install fastapi uvicorn
2. uvicorn fastapi_backend:app --reload

Open http://127.0.0.1:8000/docs to explore!
"""
