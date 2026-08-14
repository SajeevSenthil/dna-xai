import torch
import torch.nn as nn
import pytest
from src.models.classifier import DNABERT2Classifier

class MockEncoderConfig:
    def __init__(self, hidden_size: int = 768):
        self.hidden_size = hidden_size

class MockEncoderOutputs:
    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state

class MockEncoder(nn.Module):
    """
    Mock DNABERT-2 encoder for lightweight testing without downloading HF weights.
    """
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.config = MockEncoderConfig(hidden_size)
        # Dummy linear layer so we have parameters to freeze/unfreeze
        self.param = nn.Parameter(torch.randn(hidden_size))

    def forward(self, input_ids, attention_mask=None, **kwargs):
        batch_size, seq_len = input_ids.shape
        # Generate random hidden states (batch_size, seq_len, hidden_size)
        states = torch.randn(batch_size, seq_len, self.config.hidden_size, device=input_ids.device)
        return MockEncoderOutputs(states)

def test_classifier_forward_mean_pooling():
    encoder = MockEncoder(hidden_size=128)
    classifier = DNABERT2Classifier(encoder, num_labels=2, pooling="mean")
    
    # Batch of 4 samples, sequence length 20
    input_ids = torch.randint(0, 100, (4, 20))
    attention_mask = torch.ones((4, 20))
    # Mask out last 5 tokens of first sample
    attention_mask[0, -5:] = 0
    
    logits = classifier(input_ids, attention_mask=attention_mask)
    
    # Assert output dimensions
    assert logits.shape == (4, 2)
    assert not torch.isnan(logits).any()

def test_classifier_forward_cls_pooling():
    encoder = MockEncoder(hidden_size=128)
    classifier = DNABERT2Classifier(encoder, num_labels=3, pooling="cls")
    
    input_ids = torch.randint(0, 100, (2, 10))
    logits = classifier(input_ids)
    
    assert logits.shape == (2, 3)

def test_parameter_freezing():
    encoder = MockEncoder(hidden_size=64)
    classifier = DNABERT2Classifier(encoder, num_labels=2)
    
    # Freeze encoder
    for p in classifier.encoder.parameters():
        p.requires_grad = False
        
    # Unfreeze classifier head (should be True by default, but verify we can set it)
    for p in classifier.classifier.parameters():
        p.requires_grad = True
        
    # Check states
    assert classifier.encoder.param.requires_grad is False
    assert classifier.classifier.weight.requires_grad is True
    assert classifier.classifier.bias.requires_grad is True
