import torch
import pytest
from models.gptmodel import GPTModel
from models.attention import TransformerBlock, CausalSelfAttention
from models.embeddings import SimpleEmbedding

@pytest.fixture
def model_cfg():
    return {
        "vocab_size": 100,
        "context_length": 32,
        "embedding_dim": 64,
        "num_layers": 2,
        "num_heads": 4,
        "dropout": 0.0
    }

def test_simple_embedding(model_cfg):
    batch_size = 4
    seq_len = 16
    emb = SimpleEmbedding(
        model_cfg["vocab_size"], 
        model_cfg["context_length"], 
        model_cfg["embedding_dim"]
    )
    
    input_ids = torch.randint(0, model_cfg["vocab_size"], (batch_size, seq_len))
    output = emb(input_ids)
    
    assert output.shape == (batch_size, seq_len, model_cfg["embedding_dim"])

def test_causal_self_attention(model_cfg):
    batch_size = 2
    seq_len = 8
    attn = CausalSelfAttention(
        model_cfg["embedding_dim"], 
        model_cfg["num_heads"], 
        dropout=0.0
    )
    
    x = torch.randn(batch_size, seq_len, model_cfg["embedding_dim"])
    output = attn(x)
    
    assert output.shape == (batch_size, seq_len, model_cfg["embedding_dim"])

def test_transformer_block(model_cfg):
    batch_size = 2
    seq_len = 8
    block = TransformerBlock(
        model_cfg["embedding_dim"], 
        model_cfg["num_heads"], 
        dropout=0.0
    )
    
    x = torch.randn(batch_size, seq_len, model_cfg["embedding_dim"])
    output = block(x)
    
    assert output.shape == (batch_size, seq_len, model_cfg["embedding_dim"])

def test_gpt_model_forward(model_cfg):
    batch_size = 2
    seq_len = 16
    model = GPTModel(**model_cfg)
    
    input_ids = torch.randint(0, model_cfg["vocab_size"], (batch_size, seq_len))
    logits = model(input_ids)
    
    # Output should be (batch, seq, vocab_size)
    assert logits.shape == (batch_size, seq_len, model_cfg["vocab_size"])

def test_model_causality(model_cfg):
    """Ensure that changing a future token does not affect past logits."""
    model = GPTModel(**model_cfg)
    model.eval()
    
    seq_len = 10
    input_ids = torch.randint(0, model_cfg["vocab_size"], (1, seq_len))
    
    # Get logits for the first 5 tokens
    with torch.no_grad():
        logits1 = model(input_ids)[:, :5, :]
    
    # Change the 8th token
    input_ids_modified = input_ids.clone()
    input_ids_modified[0, 7] = (input_ids[0, 7] + 1) % model_cfg["vocab_size"]
    
    with torch.no_grad():
        logits2 = model(input_ids_modified)[:, :5, :]
    
    # Logits for the first 5 tokens should be identical
    torch.testing.assert_close(logits1, logits2)
