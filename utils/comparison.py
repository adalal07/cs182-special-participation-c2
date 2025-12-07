"""
Comparison Functions
====================
Functions for comparing different transformer implementations.
"""

from __future__ import annotations

from typing import Optional, Tuple

from .utils import NDArrayFloat


def compare_transformers(
    hand_transformer: object,
    learned_transformer: object,
    seq: NDArrayFloat,
    numpy_transformer_class: type,
    plot: bool = True,
    save_plot: bool = False,
    run_name: Optional[str] = None,
) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """Compare hand-designed and learned transformers visually.
    
    Args:
        hand_transformer: Hand-designed NumpyTransformer instance.
        learned_transformer: Trained PytorchTransformer instance.
        seq: Input sequence.
        numpy_transformer_class: The NumpyTransformer class to instantiate.
                                Pass NumpyTransformer from the notebook.
        plot: If True, display plots.
        save_plot: If True, save plots to the plots directory.
        run_name: Optional run identifier for organizing saved plots.
    
    Returns:
        Tuple of (hand_output, learned_output) as numpy arrays.
    
    Note: Type hints use 'object' to avoid circular imports.
    """
    assert seq.ndim == 2, f"seq must be 2D (seq_len, input_dim), got shape {seq.shape}"
    
    separator = '=' * 40
    print(f'{separator} Hand Designed {separator}')
    out_hand = hand_transformer.forward(
        seq, verbose=False, plot=plot, 
        save_plot=save_plot, plot_name="hand_designed", run_name=run_name
    )
    
    assert out_hand.shape[0] == seq.shape[0], \
        f"Output seq_len {out_hand.shape[0]} != input seq_len {seq.shape[0]}"

    # Extract learned weights (transpose due to PyTorch Linear convention)
    py_Km = learned_transformer.Km.weight.T.detach().numpy()
    py_Qm = learned_transformer.Qm.weight.T.detach().numpy()
    py_Vm = learned_transformer.Vm.weight.T.detach().numpy()
    py_pos = None
    if learned_transformer.pos is not None:
        py_pos = learned_transformer.pos.weight.detach().numpy()
    
    print(f'{separator}    Learned    {separator}')
    np_learned = numpy_transformer_class(py_Km, py_Qm, py_Vm, py_pos)
    out_learned = np_learned.forward(
        seq, verbose=False, plot=plot,
        save_plot=save_plot, plot_name="learned", run_name=run_name
    )
    
    assert out_learned.shape == out_hand.shape, \
        f"Shape mismatch: hand={out_hand.shape}, learned={out_learned.shape}"
    
    return out_hand, out_learned

