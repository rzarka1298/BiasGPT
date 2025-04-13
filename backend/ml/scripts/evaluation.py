import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from transformers import AutoTokenizer

# Import your model and dataset classes.
# Ensure that BiasClassifier is defined in your classifier module (ml/scripts/classifier.py).
# If not, you may need to update the import or implement the class.
from backend.ml.scripts.classifier import BiasClassifier, BiasDataset

# Configuration parameters for evaluation.
MODEL_NAME = "distilbert-base-uncased"  # This must match the pre-trained model used during training.
MAX_LEN = 256                         # Maximum sequence length for tokenization.
BATCH_SIZE = 16                       # Batch size for evaluation.

def evaluate_model(model_path: str, data_path: str):
    """
    Evaluate the saved BiasClassifier on a test/validation dataset.
    
    This function performs the following steps:
      1. Loads the CSV data from `data_path` (expects columns: original, swapped, label).
      2. Concatenates the 'original' and 'swapped' texts with a [SEP] token.
      3. Tokenizes the text using AutoTokenizer.
      4. Creates a BiasDataset and a DataLoader for batch processing.
      5. Loads the model from the specified model_path.
      6. Performs inference on the dataset.
      7. Computes and prints a classification report and a confusion matrix.

    Parameters:
      model_path (str): Path to the saved model checkpoint.
      data_path (str): Path to the CSV file containing the evaluation data.
    """
    
    # Load evaluation data from CSV.
    df = pd.read_csv(data_path)
    # Combine the two text columns using a separator.
    texts = (df["original"] + " [SEP] " + df["swapped"]).tolist()
    # Extract labels, ensuring that they are integers.
    labels = df["label"].astype(int).tolist()

    # Initialize the tokenizer using the specified model name.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Create the evaluation dataset.
    dataset = BiasDataset(texts, labels, tokenizer, MAX_LEN)
    # Create a DataLoader to iterate over the dataset in batches.
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Set the device for evaluation (GPU if available, else CPU).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the BiasClassifier model.
    # Ensure that BiasClassifier is implemented to accept the model name and number of labels.
    model = BiasClassifier(MODEL_NAME, num_labels=2)
    # Load the saved model checkpoint.
    model.load(model_path)
    model.to(device)
    model.eval()  # Set the model to evaluation mode.

    all_preds = []  # Will store predictions for the entire evaluation dataset.
    all_labels = [] # Will store true labels.

    # Run inference without computing gradients.
    with torch.no_grad():
        for batch in dataloader:
            # Move input tensors to the evaluation device.
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['labels'].to(device)

            # Forward pass: get the model outputs (logits).
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Compute predictions by choosing the class with the highest logit.
            preds = torch.argmax(logits, dim=1)
            
            # Append predictions and true labels to the lists.
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    # Print out the classification report and confusion matrix.
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, digits=4))
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

def main():
    """
    Main entry point for the evaluation script.
    
    Command-line arguments:
      --model_path: Path to the saved model checkpoint.
      --data_path:  Path to the evaluation CSV file.
    """
    parser = argparse.ArgumentParser(description="Evaluate the BiasClassifier model.")
    parser.add_argument("--model_path", type=str, default="ml/models/bias_classifier.pt",
                        help="Path to the saved model checkpoint.")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the evaluation CSV file (expects columns: original, swapped, label).")
    args = parser.parse_args()

    evaluate_model(args.model_path, args.data_path)

if __name__ == "__main__":
    main()
