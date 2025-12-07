"""
Training Functions
==================
Functions for training transformer models.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn

from .constants import TRAIN_CONFIG
from .utils import TensorFloat, set_random_seed


def train_loop(
    make_batch: Callable[[], Tuple[TensorFloat, TensorFloat]],
    input_dim: int,
    qk_dim: int,
    v_dim: int,
    transformer_class: type,
    pos_dim: Optional[int] = None,
    max_seq_len: Optional[int] = None,
    remove_cls: bool = False,
    num_epochs: int = TRAIN_CONFIG.num_epochs,
    lr: float = TRAIN_CONFIG.learning_rate,
    log_interval: int = TRAIN_CONFIG.log_interval,
    seed: Optional[int] = None,
) -> Tuple[object, float]:
    """Train a PytorchTransformer on a given task.
    
    Args:
        make_batch: Callable returning (input_sequence, target) tuples.
        input_dim: Dimension of input token embeddings.
        qk_dim: Dimension of query/key projections.
        v_dim: Dimension of value projections (output dimension).
        transformer_class: The PytorchTransformer class to instantiate.
                          Pass PytorchTransformer from the notebook.
        pos_dim: Dimension of positional encodings. None disables them.
        max_seq_len: Maximum sequence length for positional encodings.
        remove_cls: If True, exclude first token from loss computation.
        num_epochs: Number of training iterations.
        lr: Learning rate for SGD optimizer.
        log_interval: Interval for logging training progress.
        seed: Optional random seed for reproducibility.
    
    Returns:
        Tuple of (trained_model, final_loss_value).
    
    Note: Return type uses 'object' instead of 'PytorchTransformer' to avoid
    circular imports. The actual return type is PytorchTransformer.
    """
    if seed is not None:
        set_random_seed(seed)
    
    # Dimension validation
    assert input_dim > 0, f"input_dim must be positive, got {input_dim}"
    assert qk_dim > 0, f"qk_dim must be positive, got {qk_dim}"
    assert v_dim > 0, f"v_dim must be positive, got {v_dim}"
    
    transformer = transformer_class(input_dim, qk_dim, v_dim, pos_dim, max_seq_len)
    optimizer = torch.optim.SGD(transformer.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    final_loss = 0.0
    for epoch in range(num_epochs):
        seq, target = make_batch()
        
        assert seq.dim() == 2, f"Input must be 2D (seq_len, input_dim), got {seq.shape}"
        assert seq.shape[1] == input_dim, f"Input dim mismatch: expected {input_dim}, got {seq.shape[1]}"
        
        optimizer.zero_grad()
        out = transformer(seq)
        if remove_cls:
            out = out[1:]
        
        assert out.shape == target.shape, f"Output {out.shape} != target {target.shape}"
        
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
        
        if epoch % log_interval == 0:
            print(f'Step {epoch}: loss {final_loss:.6f}')
    
    return transformer, final_loss

