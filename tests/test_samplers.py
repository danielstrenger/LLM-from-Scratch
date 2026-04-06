import torch
import pytest
from models.gptmodel import GPTModel
from samplers.samplers import generate_text_simple, sample_with_temperature

@pytest.fixture
def tiny_model():
    cfg = {
        "vocab_size": 50,
        "context_length": 16,
        "embedding_dim": 32,
        "num_layers": 1,
        "num_heads": 2,
    }
    model = GPTModel(**cfg)
    model.eval()
    return model, cfg

def test_generate_text_simple(tiny_model):
    model, cfg = tiny_model
    batch_size = 1
    input_seq_len = 5
    max_new_tokens = 3
    
    input_ids = torch.randint(0, cfg["vocab_size"], (batch_size, input_seq_len))
    
    output = generate_text_simple(
        model, 
        input_ids, 
        max_new_tokens=max_new_tokens, 
        context_size=cfg["context_length"]
    )
    
    assert output.shape == (batch_size, input_seq_len + max_new_tokens)

def test_sample_with_temperature(tiny_model):
    model, cfg = tiny_model
    batch_size = 2
    input_seq_len = 5
    max_new_tokens = 4
    
    input_ids = torch.randint(0, cfg["vocab_size"], (batch_size, input_seq_len))
    
    output = sample_with_temperature(
        model, 
        input_ids, 
        max_new_tokens=max_new_tokens, 
        context_size=cfg["context_length"],
        temperature=0.8
    )
    
    assert output.shape == (batch_size, input_seq_len + max_new_tokens)

def test_sample_with_temperature_invalid(tiny_model):
    model, cfg = tiny_model
    input_ids = torch.randint(0, cfg["vocab_size"], (1, 5))
    
    with pytest.raises(ValueError, match="temperature must be > 0"):
        sample_with_temperature(
            model, 
            input_ids, 
            max_new_tokens=1, 
            context_size=10, 
            temperature=0.0
        )
