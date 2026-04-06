import pytest
import torch
from data.datasets import GPTDatasetV1, create_dataloader_v1
import tiktoken

@pytest.fixture
def sample_text():
    return "This is a simple sample text used for testing the dataset implementation."

def test_dataset_output_shapes(sample_text):
    tokenizer = tiktoken.get_encoding("gpt2")
    max_length = 4
    stride = 2
    
    dataset = GPTDatasetV1(sample_text, tokenizer, max_length, stride)
    
    # Check that individual samples are the correct shape
    inputs, targets = dataset[0]
    assert inputs.shape == (max_length,)
    assert targets.shape == (max_length,)
    
    # Test that target is input shifted by 1
    # Note: we need to verify tokens here
    token_ids = tokenizer.encode(sample_text)
    assert torch.equal(inputs, torch.tensor(token_ids[0:4]))
    assert torch.equal(targets, torch.tensor(token_ids[1:5]))

def test_dataloader_batch_shape(sample_text):
    batch_size = 2
    max_length = 4
    
    loader = create_dataloader_v1(
        sample_text, 
        batch_size=batch_size, 
        max_length=max_length, 
        stride=1,
        shuffle=False,
        drop_last=True
    )
    
    input_batch, target_batch = next(iter(loader))
    assert input_batch.shape == (batch_size, max_length)
    assert target_batch.shape == (batch_size, max_length)

def test_dataset_too_short():
    tokenizer = tiktoken.get_encoding("gpt2")
    text = "Short text" # few tokens
    
    with pytest.raises(AssertionError):
        GPTDatasetV1(text, tokenizer, max_length=100, stride=1)
