import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm

############################################
#           CONFIGURATION PARAMETERS       #
############################################

MODEL_NAME = "distilbert-base-uncased"  # Pretrained model name (must match across training and evaluation)
MAX_LEN = 256                          # Maximum sequence length for tokenization
BATCH_SIZE = 16                        # Batch size used for training and validation
EPOCHS = 4                             # Number of training epochs
LR = 2e-5                              # Learning rate

############################################
#           DATA LOADING & PREPROCESSING   #
############################################

# Load dataset CSV.
# Expected columns: original, swapped, label (labels can be "0" or "1").
df = pd.read_csv("ml/data/dataset.csv")

# Debug output: Print unique values in the label column before processing.
print("Initial unique label values:", df["label"].unique())

# Rename columns if needed (e.g., if dataset has different naming conventions).
df = df.rename(columns={"base_prompt": "original", "swapped_prompt": "swapped"})

# Clean and standardize the label column:
# 1. Convert labels to string.
# 2. Strip any whitespace.
# 3. Replace blank strings or "nan" with a None value.
df["label"] = (
    df["label"]
      .astype(str)
      .str.strip()
      .replace({"": None, "nan": None})
)

# Drop rows with missing or invalid labels (only keep rows labeled "0" or "1").
df = df[df["label"].isin(["0", "1"])]

# Convert labels to integers.
df["label"] = df["label"].astype(int)

# Combine the 'original' and 'swapped' text columns into a single text
# separated by " [SEP] " for consistency with the model input.
texts = (df["original"] + " [SEP] " + df["swapped"]).tolist()
labels = df["label"].tolist()

############################################
#             TOKENIZER SETUP              #
############################################

# Initialize the tokenizer from the specified pretrained model.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

############################################
#         CUSTOM DATASET DEFINITION        #
############################################

class BiasDataset(Dataset):
    """
    PyTorch Dataset for bias classification.
    
    This dataset takes in combined texts and corresponding labels,
    tokenizes the texts, and formats them to be used by a transformer
    model for sequence classification.
    """
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        # Returns the total number of samples.
        return len(self.texts)

    def __getitem__(self, idx):
        # Get the text and corresponding label for the index.
        text = self.texts[idx]
        label = self.labels[idx]
        # Tokenize the text, ensuring it is truncated and padded to max_len.
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        # Squeeze the tensors to remove the extra batch dimension (since return_tensors='pt' adds one).
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

############################################
#              DATA SPLIT                #
############################################

# Split the dataset into training and validation sets.
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Create dataset instances for training and validation.
train_dataset = BiasDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = BiasDataset(val_texts, val_labels, tokenizer, MAX_LEN)

# Create DataLoaders for batching the data.
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

############################################
#         MODEL INITIALIZATION             #
############################################

# Load a pretrained transformer model for sequence classification.
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Set up the device: use GPU if available, else CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the optimizer for training.
optimizer = optim.AdamW(model.parameters(), lr=LR)

# Define the loss criterion.
# Note: When using AutoModelForSequenceClassification with labels,
# the model outputs include a loss value already computed.
criterion = nn.CrossEntropyLoss()  # (Not used directly in this loop since loss comes from the model.)

############################################
#              TRAINING LOOP               #
############################################

for epoch in range(EPOCHS):
    model.train()  # Set model to training mode.
    total_loss = 0
    
    # Iterate over training batches with a progress bar.
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()  # Clear previous gradients.
        
        # Forward pass: model returns a loss (since labels are provided).
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        # Backpropagation.
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    # Calculate and print the average training loss for the epoch.
    avg_train_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch + 1} Training Loss: {avg_train_loss:.4f}")

    ############################################
    #              VALIDATION STEP             #
    ############################################

    model.eval()  # Switch to evaluation mode.
    correct, total = 0, 0  # Counters for accuracy.

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass (no gradient computation during validation).
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            # Accumulate the number of correct predictions.
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    # Compute validation accuracy.
    accuracy = correct / total
    print(f"Epoch {epoch + 1} Validation Accuracy: {accuracy:.4f}\n")

############################################
#          SAVE THE TRAINED MODEL          #
############################################

# Create the models directory if it doesn't exist.
os.makedirs("ml/models", exist_ok=True)
# Save the trained model's state dictionary.
torch.save(model.state_dict(), "ml/models/bias_classifier.pt")
print("Model saved to ml/models/bias_classifier.pt")
