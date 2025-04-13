import torch
from torch.utils.data import Dataset
import torch.nn as nn

# Although AutoModelForSequenceClassification is imported here,
# it might be used in other parts of the project. Keeping it imported
# here in case we extend model definitions in this file.
from transformers import AutoModelForSequenceClassification

class BiasDataset(Dataset):
    """
    A PyTorch Dataset for bias classification tasks.

    This dataset tokenizes input texts using a provided tokenizer and
    prepares them for training a transformer-based classification model.
    It also stores corresponding labels.

    Parameters:
        texts (List[str]): List of raw text strings.
        labels (List[int] or List[float]): The labels corresponding to each text.
        tokenizer (PreTrainedTokenizer): A tokenizer from Hugging Face Transformers,
                                         e.g., from AutoTokenizer.from_pretrained().
        max_length (int, optional): Maximum sequence length for tokenization.
                                    Default is 256.

    Returns:
        A dictionary containing tokenized inputs and the label as tensor for a given index.
    """
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        # Tokenize the input texts.
        # This converts texts to a dictionary with keys such as 'input_ids', 'attention_mask', etc.
        self.data = tokenizer(texts,
                              truncation=True,
                              padding=True,
                              max_length=max_length,
                              return_tensors="pt")
        # Store the labels as provided (they can be integers or floats).
        self.labels = labels

    def __len__(self):
        # Return the total number of samples
        return len(self.labels)

    def __getitem__(self, idx):
        """
        For a given index, return a dictionary with tokenized inputs
        and the corresponding label tensor.
        """
        # Extract and structure tokenized inputs for the current index.
        # Each key's tensor is indexed at position idx.
        item = {key: value[idx] for key, value in self.data.items()}
        # Convert label at index idx to tensor and add it under the key 'labels'.
        item["labels"] = torch.tensor(self.labels[idx])
        return item

# End of BiasDataset definition.


class BiasClassifier(nn.Module):
    """
    A simple wrapper for a transformer-based model for bias classification.
    
    This class loads a pretrained model (e.g. from Hugging Face) for sequence classification,
    and provides convenience methods for forward passes and loading/saving model weights.
    """
    def __init__(self, model_name: str, num_labels: int):
        super(BiasClassifier, self).__init__()
        # Load a pretrained model for sequence classification.
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass through the model.
        
        Parameters:
            input_ids: Tensor of token ids.
            attention_mask: Tensor indicating which tokens are actual words.
            labels (optional): Tensor of target labels.
        
        Returns:
            Model output, which may include loss if labels are provided.
        """
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    def load(self, model_path: str):
        """
        Load model weights from the specified path.
        """
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(state_dict)
    
    def save(self, model_path: str):
        """
        Save model weights to the specified path.
        """
        torch.save(self.model.state_dict(), model_path)
