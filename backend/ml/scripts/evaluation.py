import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from transformers import AutoTokenizer

from backend.ml.scripts.classifier import BiasClassifier
from train import BiasDataset  # or import from your data module

MODEL_NAME = "distilbert-base-uncased"  # must match training
MAX_LEN = 256
BATCH_SIZE = 16


def evaluate_model(model_path: str, data_path: str):
    """
    Evaluate the saved BiasClassifier on a test or validation dataset.
    """
    # Load data
    df = pd.read_csv(data_path)  # expects columns: original, swapped, label
    texts = (df["original"] + " [SEP] " + df["swapped"]).tolist()
    labels = df["label"].tolist()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = BiasDataset(texts, labels, tokenizer, MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiasClassifier(MODEL_NAME, num_labels=2)
    model.load(model_path)
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    # Metrics
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, digits=4))
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the BiasClassifier.")
    parser.add_argument("--model_path", type=str, default="ml/models/bias_classifier.pt", help="Path to the saved model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the evaluation CSV")

    args = parser.parse_args()

    evaluate_model(args.model_path, args.data_path)


if __name__ == "__main__":
    main()
