import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import Dict, Any

class DNADataset(Dataset):
    """
    PyTorch Dataset for DNA sequences.
    Tokenizes sequences on-the-fly using the DNABERT-2 tokenizer.
    """
    def __init__(self, df: pd.DataFrame, tokenizer: Any, max_length: int = 128):
        self.sequences = df["sequence"].values
        self.labels = df["label"].values
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = str(self.sequences[idx])
        label = int(self.labels[idx])

        # Tokenize the DNA sequence
        encoding = self.tokenizer(
            seq,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Squeeze batch dimension (since we only pass one sequence to tokenizer here)
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        
        # Keep track of original sequence text if needed for explanation layers
        # (Note: we cannot put strings in standard PyTorch batch tensors directly,
        # but we can query it using dataset indices if needed).
        return item

def create_data_loader(
    df: pd.DataFrame,
    tokenizer: Any,
    max_length: int = 128,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 0
) -> DataLoader:
    """
    Helper function to create a PyTorch DataLoader for DNA sequence dataframes.
    """
    dataset = DNADataset(df, tokenizer, max_length=max_length)
    
    # We do NOT use pin_memory=True if running on CPU or if CUDA isn't initialized yet,
    # but we can set pin_memory = torch.cuda.is_available() for optimal GPU transfers.
    pin_memory = torch.cuda.is_available()
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
