from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_model_simple(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    loss_fn: Union[nn.Module, Callable[..., torch.Tensor]],
    device: Union[str, torch.device],
    num_epochs: int,
    val_loader: Optional[DataLoader] = None,
    scheduler: Optional[Any] = None,
    max_grad_norm: Optional[float] = 1.0,
) -> dict:
    """Train ``model`` for ``num_epochs`` with optional validation and scheduling.

    Args:
        model: PyTorch module mapping ``input_batch`` to logits.
        train_loader: Yields ``(input_batch, target_batch)`` batches.
        optimizer: Optimizer for ``model`` parameters.
        loss_fn: Loss taking flattened logits and flattened targets.
        device: Device string or ``torch.device``.
        num_epochs: Number of full passes over ``train_loader``.
        val_loader: Optional validation loader.
        scheduler: Optional learning rate scheduler.
        max_grad_norm: Maximum norm for gradient clipping.

    Returns:
        A dictionary containing 'train_losses' and 'val_losses' lists.
    """
    train_losses, val_losses = [], []
    
    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        epoch_loss_sum = 0.0
        epoch_steps = 0
        
        for input_batch, target_batch in pbar:
            optimizer.zero_grad()
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            logits = model(input_batch)
            loss = loss_fn(logits.flatten(0, 1), target_batch.flatten())
            
            loss.backward()
            
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
            optimizer.step()
            
            if scheduler is not None:
                # If it's a scheduler that steps per batch (like CosineAnnealingLR with many steps)
                # For simplicity here we assume step() is handled appropriately.
                # Common practice is often per-epoch or per-batch depending on the scheduler.
                pass 

            epoch_loss_sum += loss.item()
            epoch_steps += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.2e}"})

        avg_train_loss = epoch_loss_sum / max(epoch_steps, 1)
        train_losses.append(avg_train_loss)

        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_steps = 0
            with torch.no_grad():
                for input_batch, target_batch in val_loader:
                    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
                    logits = model(input_batch)
                    loss = loss_fn(logits.flatten(0, 1), target_batch.flatten())
                    val_loss_sum += loss.item()
                    val_steps += 1

            avg_val_loss = val_loss_sum / max(val_steps, 1)
            val_losses.append(avg_val_loss)
            print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if scheduler is not None:
            scheduler.step()

    return {
        "train_losses": train_losses,
        "val_losses": val_losses
    }