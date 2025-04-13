import argparse
import torch
from transformers import AutoTokenizer
from ml.scripts.classifier import BiasClassifier  # Ensure BiasClassifier is properly defined

# ------------------------------------------------------------------------------
# Configuration Constants
# ------------------------------------------------------------------------------
MODEL_NAME = "distilbert-base-uncased"  # Must match the pre-trained model used in training
MAX_LEN = 256                         # Maximum sequence length for tokenization

# Initialize the tokenizer using the specified model name.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Create a global instance of the model.
# Note: The model weights will be loaded inside the predict_bias function.
model = BiasClassifier(MODEL_NAME, num_labels=2)

# ------------------------------------------------------------------------------
# Prediction Function
# ------------------------------------------------------------------------------
def predict_bias(original_text: str, swapped_text: str, model_path: str) -> int:
    """
    Load the saved model weights and predict whether the difference is biased or neutral.

    Parameters:
        original_text (str): The original prompt or text.
        swapped_text (str): The demographically swapped text.
        model_path (str): Path to the saved model checkpoint (.pt file).

    Returns:
        int: The prediction (0 for neutral, 1 for biased).
    """
    # Load the model weights from the checkpoint.
    model.load(model_path)

    # Determine the device: use CUDA if available, else CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Prepare the input by concatenating the original and swapped texts with a [SEP] token.
    combo_text = original_text + " [SEP] " + swapped_text

    # Tokenize the concatenated text. This returns a dictionary containing the input_ids and attention_mask.
    inputs = tokenizer(
        combo_text,
        truncation=True,
        padding='max_length',
        max_length=MAX_LEN,
        return_tensors='pt'
    )

    # Move input tensors to the proper device.
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Set model to evaluation mode and perform inference.
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

    # Return the prediction as an integer (0 for neutral, 1 for biased).
    return preds.item()

# ------------------------------------------------------------------------------
# Main Entry Point for Command-Line Inference
# ------------------------------------------------------------------------------
def main():
    """
    Command-line interface for bias prediction.

    Required arguments:
      --original: Original text prompt.
      --swapped: Swapped text prompt.
      
    Optional argument:
      --model_path: File path to the saved model checkpoint.
    """
    parser = argparse.ArgumentParser(description="BiasClassifier Inference")
    parser.add_argument("--original", type=str, required=True, help="Original text")
    parser.add_argument("--swapped", type=str, required=True, help="Swapped text")
    parser.add_argument("--model_path", type=str, default="ml/models/bias_classifier.pt",
                        help="Path to the saved model checkpoint (.pt file)")

    args = parser.parse_args()

    # Perform prediction and get the output label.
    label = predict_bias(args.original, args.swapped, args.model_path)

    # Print the result based on the label.
    if label == 1:
        print("Biased")
    else:
        print("Neutral")

if __name__ == "__main__":
    main()
