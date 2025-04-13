import os
import shutil
import pandas as pd
from dotenv import load_dotenv; load_dotenv()
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import torch

# Import your dataset helper from your classifier file.
from ml.scripts.classifier import BiasDataset

############################################
#        CONFIGURATION & PATHS             #
############################################

DATA = "ml/data/master.csv"                          # Path to the master CSV
MODEL_DIR = Path("ml/models/bias_classifier")         # Directory for the deployed model
NEW_DIR = Path("ml/models/_candidate")                # Directory for the candidate (new) model

############################################
#            DATA LOADING UTILITY          #
############################################

def load_data(test_size=0.3, random_state=42):
    """
    Load the master CSV, concatenate 'original' and 'swapped' texts,
    and split the data into training and validation sets.
    
    Returns:
        Tuple of (train_texts, val_texts, train_labels, val_labels).
    """
    df = pd.read_csv(DATA)
    texts = (df["original"] + " [SEP] " + df["swapped"]).tolist()
    labels = df["label"].astype(int).tolist()
    return train_test_split(texts, labels, test_size=test_size, random_state=random_state)

############################################
#           TRAINING FUNCTION              #
############################################

def train():
    """
    Train a new candidate model using data from the master CSV.
    
    Returns:
        float: The evaluation F1 score from the candidate model.
    """
    # Load data splits.
    train_texts, val_texts, train_labels, val_labels = load_data(test_size=0.2)
    
    # Initialize the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    
    # Create training and evaluation datasets.
    train_ds = BiasDataset(train_texts, train_labels, tokenizer, max_length=256)
    val_ds   = BiasDataset(val_texts,   val_labels,   tokenizer, max_length=256)
    
    # Set up training arguments (using eval_strategy instead of deprecated evaluation_strategy).
    args = TrainingArguments(
        output_dir=str(NEW_DIR),
        per_device_train_batch_size=16,
        num_train_epochs=3,
        eval_strategy="epoch",       # Evaluation every epoch.
        save_strategy="epoch",       # Save checkpoints every epoch.
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,
    )
    
    def compute_metrics(eval_pred):
        from sklearn.metrics import precision_score, recall_score, f1_score
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        # Log distributions for debugging:
        print("Label distribution (true):", pd.Series(labels).value_counts().to_dict())
        print("Prediction distribution:", pd.Series(preds).value_counts().to_dict())
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        return {"precision": precision, "recall": recall, "f1": f1}

    
    # Initialize the model.
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
    
    # Set device (force CPU for this example; adjust as needed).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Initialize the Trainer.
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    
    # Train the candidate model.
    trainer.train()
    
    # Evaluate the candidate model.
    metrics = trainer.evaluate()
    print("Evaluation metrics:", metrics)
    return metrics["eval_f1"]

############################################
#          HOT RELOAD UTILITY              #
############################################

def hot_reload():
    """
    Trigger the backend API to reload the newly deployed model.
    """
    import requests
    try:
        requests.post("http://127.0.0.1:8000/reload_model", timeout=5)
    except Exception as e:
        print("Hot reload failed:", e)

############################################
#         MAIN DEPLOYMENT LOGIC            #
############################################

def main():
    """
    Main routine for training and deploying a new model.
    
    Compares the candidate model's F1 score against the currently deployed model.
    If the candidate model outperforms the current model by a small margin (epsilon),
    it gets deployed and triggers a hot reload; otherwise, it is discarded.
    """
    new_f1 = train()
    old_f1 = 0.0
    if MODEL_DIR.exists() and (MODEL_DIR / "metrics.txt").exists():
        old_f1 = float((MODEL_DIR / "metrics.txt").read_text())
    
    epsilon = 0.00  # Small tolerance to require meaningful improvement.
    if new_f1 >= old_f1:
        print(f"New model better: {new_f1:.3f} > {old_f1:.3f}. Deploying.")
        shutil.rmtree(MODEL_DIR, ignore_errors=True)
        shutil.move(str(NEW_DIR), str(MODEL_DIR))
        (MODEL_DIR / "metrics.txt").write_text(str(new_f1))
        hot_reload()
    else:
        print(f"New model worse: {new_f1:.3f} ≤ {old_f1:.3f}. Discarding.")
        shutil.rmtree(NEW_DIR, ignore_errors=True)

if __name__ == "__main__":
    main()
