import argparse
import torch
from transformers import AutoTokenizer
from ml.scripts.classifier import BiasClassifier

MODEL_NAME = "distilbert-base-uncased"  # Must match the one used in training
MAX_LEN = 256

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Initialize model
model = BiasClassifier(MODEL_NAME, num_labels=2)


def predict_bias(original_text: str, swapped_text: str, model_path: str) -> int:
    """Load the model, predict whether the difference is biased (1) or neutral (0)."""

    # Load weights
    model.load(model_path)

    # Move model to CPU or CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Prep text pair input
    combo_text = original_text + " [SEP] " + swapped_text
    inputs = tokenizer(
        combo_text,
        truncation=True,
        padding='max_length',
        max_length=MAX_LEN,
        return_tensors='pt'
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Inference
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

    return preds.item()  # 0 or 1


def main():
    parser = argparse.ArgumentParser(description="BiasClassifier Inference")
    parser.add_argument("--original", type=str, required=True, help="Original GPT output text")
    parser.add_argument("--swapped", type=str, required=True, help="Swapped GPT output text")
    parser.add_argument("--model_path", type=str, default="ml/models/bias_classifier.pt", help="Path to the saved .pt file")

    args = parser.parse_args()

    label = predict_bias(args.original, args.swapped, args.model_path)

    if label == 1:
        print("Biased")
    else:
        print("Neutral")


if __name__ == "__main__":
    main()
