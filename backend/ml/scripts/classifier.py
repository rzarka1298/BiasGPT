import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

class BiasClassifier(nn.Module):
    """
    A simple wrapper around Hugging Face's AutoModelForSequenceClassification.
    This class can load a pre-trained DistilBERT or any other HF transformer model.
    """

    def __init__(self, model_name="distilbert-base-uncased", num_labels=2):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )

    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass of the underlying HuggingFace model."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

    def save(self, path: str):
        """Save the model state_dict."""
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        state = torch.load(path, map_location=torch.device("cpu"))
        # If the keys don't start with "model.", load into the inner HF model
        if not any(k.startswith("model.") for k in state.keys()):
            self.model.load_state_dict(state)
        else:
            self.load_state_dict(state)
        self.eval()

