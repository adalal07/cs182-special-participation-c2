"""
Transformer Tests
=================
Comprehensive tests for NumpyTransformer and PytorchTransformer equivalence.
"""

import inspect
from typing import Optional

import numpy as np
import torch

from utils.constants import (
    RELATIVE_TOLERANCE,
    TEST_ITERATIONS,
    TEST_INPUT_DIM,
    TEST_MIN_SEQ_LEN,
    TEST_MAX_SEQ_LEN,
)
from utils.utils import set_random_seed


def _run_equivalence_test(
    NumpyTransformer,
    PytorchTransformer,
    Km: np.ndarray,
    Qm: np.ndarray,
    Vm: np.ndarray,
    pos: Optional[np.ndarray],
    seq: np.ndarray,
    max_seq_len: int = TEST_MAX_SEQ_LEN,
) -> None:
    """Helper function to test equivalence between NumPy and PyTorch implementations.
    
    Args:
        NumpyTransformer: The NumpyTransformer class.
        PytorchTransformer: The PytorchTransformer class.
        Km, Qm, Vm: Projection matrices.
        pos: Optional positional encoding matrix.
        seq: Input sequence.
        max_seq_len: Maximum sequence length for positional encoding.
    """
    # Determine dimensions
    input_dim = Km.shape[0]
    qk_dim = Km.shape[1]
    v_dim = Vm.shape[1]
    seq_dim = seq.shape[1]
    pos_dim = pos.shape[1] if pos is not None else None
    
    # Run NumPy implementation
    out_np = NumpyTransformer(Km, Qm, Vm, pos).forward(seq, verbose=False)
    
    # Run PyTorch implementation
    transformer = PytorchTransformer(seq_dim, qk_dim, v_dim, pos_dim, max_seq_len)
    state_dict = transformer.state_dict()
    state_dict['Km.weight'] = torch.FloatTensor(Km.T)
    state_dict['Qm.weight'] = torch.FloatTensor(Qm.T)
    state_dict['Vm.weight'] = torch.FloatTensor(Vm.T)
    if pos is not None:
        state_dict['pos.weight'] = torch.FloatTensor(pos)
    transformer.load_state_dict(state_dict)
    out_py = transformer(torch.FloatTensor(seq)).detach().numpy()
    
    # Compare outputs
    if not np.allclose(out_np, out_py, rtol=RELATIVE_TOLERANCE):
        print('ERROR: Implementation mismatch!')
        print(f'NumPy output: {out_np}')
        print(f'PyTorch output: {out_py}')
        raise ValueError('NumPy and PyTorch outputs do not match')


def test_transformer_equivalence(
    NumpyTransformer,
    PytorchTransformer,
    seed: int = 42,
    num_iterations: int = TEST_ITERATIONS,
) -> None:
    """Test that NumpyTransformer and PytorchTransformer produce identical outputs.
    
    Args:
        NumpyTransformer: The NumpyTransformer class from the notebook.
        PytorchTransformer: The PytorchTransformer class from the notebook.
        seed: Random seed for reproducibility.
        num_iterations: Number of test iterations to run.
    """
    set_random_seed(seed)
    qk_dim = np.random.randint(1, 5)
    v_dim = np.random.randint(1, 5)
    
    for i in range(num_iterations):
        Km = np.random.randn(TEST_INPUT_DIM, qk_dim)
        Qm = np.random.randn(TEST_INPUT_DIM, qk_dim)
        Vm = np.random.randn(TEST_INPUT_DIM, v_dim)
        
        if i < num_iterations // 2:
            pos_dim = np.random.randint(2, 4)
            pos = np.random.randn(TEST_MAX_SEQ_LEN, pos_dim)
            seq_dim = TEST_INPUT_DIM - pos_dim
        else:
            pos_dim = None
            pos = None
            seq_dim = TEST_INPUT_DIM

        seq = np.random.randn(
            np.random.randint(TEST_MIN_SEQ_LEN, TEST_MAX_SEQ_LEN + 1), 
            seq_dim
        )
        
        _run_equivalence_test(
            NumpyTransformer, PytorchTransformer,
            Km, Qm, Vm, pos, seq
        )
    
    print('✓ All equivalence tests passed!')


def test_with_positional_encoding(
    NumpyTransformer,
    PytorchTransformer,
    seed: int = 42,
    num_tests: int = 5,
) -> None:
    """Test transformers with positional encodings.
    
    Args:
        NumpyTransformer: The NumpyTransformer class from the notebook.
        PytorchTransformer: The PytorchTransformer class from the notebook.
        seed: Random seed for reproducibility.
        num_tests: Number of test cases to run.
    """
    set_random_seed(seed)
    
    for i in range(num_tests):
        qk_dim = np.random.randint(1, 5)
        v_dim = np.random.randint(1, 5)
        pos_dim = np.random.randint(2, 4)
        seq_dim = TEST_INPUT_DIM - pos_dim
        
        Km = np.random.randn(TEST_INPUT_DIM, qk_dim)
        Qm = np.random.randn(TEST_INPUT_DIM, qk_dim)
        Vm = np.random.randn(TEST_INPUT_DIM, v_dim)
        pos = np.random.randn(TEST_MAX_SEQ_LEN, pos_dim)
        
        seq = np.random.randn(
            np.random.randint(TEST_MIN_SEQ_LEN, TEST_MAX_SEQ_LEN + 1),
            seq_dim
        )
        
        try:
            _run_equivalence_test(
                NumpyTransformer, PytorchTransformer,
                Km, Qm, Vm, pos, seq
            )
        except ValueError as e:
            raise AssertionError(f"Test {i} failed: outputs don't match with positional encoding") from e
    
    print(f'✓ All {num_tests} tests with positional encoding passed!')


def test_without_positional_encoding(
    NumpyTransformer,
    PytorchTransformer,
    seed: int = 42,
    num_tests: int = 5,
) -> None:
    """Test transformers without positional encodings.
    
    Args:
        NumpyTransformer: The NumpyTransformer class from the notebook.
        PytorchTransformer: The PytorchTransformer class from the notebook.
        seed: Random seed for reproducibility.
        num_tests: Number of test cases to run.
    """
    set_random_seed(seed)
    
    for i in range(num_tests):
        qk_dim = np.random.randint(1, 5)
        v_dim = np.random.randint(1, 5)
        seq_dim = TEST_INPUT_DIM
        
        Km = np.random.randn(TEST_INPUT_DIM, qk_dim)
        Qm = np.random.randn(TEST_INPUT_DIM, qk_dim)
        Vm = np.random.randn(TEST_INPUT_DIM, v_dim)
        pos = None
        
        seq = np.random.randn(
            np.random.randint(TEST_MIN_SEQ_LEN, TEST_MAX_SEQ_LEN + 1),
            seq_dim
        )
        
        try:
            _run_equivalence_test(
                NumpyTransformer, PytorchTransformer,
                Km, Qm, Vm, pos, seq
            )
        except ValueError as e:
            raise AssertionError(f"Test {i} failed: outputs don't match without positional encoding") from e
    
    print(f'✓ All {num_tests} tests without positional encoding passed!')


def test_dimension_variations(
    NumpyTransformer,
    PytorchTransformer,
    seed: int = 42,
) -> None:
    """Test transformers with various dimension combinations.
    
    Args:
        NumpyTransformer: The NumpyTransformer class from the notebook.
        PytorchTransformer: The PytorchTransformer class from the notebook.
        seed: Random seed for reproducibility.
    """
    set_random_seed(seed)
    
    # Test different dimension combinations
    test_configs = [
        {"input_dim": 5, "qk_dim": 1, "v_dim": 1, "pos_dim": None},
        {"input_dim": 5, "qk_dim": 3, "v_dim": 3, "pos_dim": None},
        {"input_dim": 7, "qk_dim": 2, "v_dim": 4, "pos_dim": 2},
        {"input_dim": 10, "qk_dim": 5, "v_dim": 3, "pos_dim": 3},
    ]
    
    for i, config in enumerate(test_configs):
        input_dim = config["input_dim"]
        qk_dim = config["qk_dim"]
        v_dim = config["v_dim"]
        pos_dim = config["pos_dim"]
        
        if pos_dim is not None:
            seq_dim = input_dim - pos_dim
            pos = np.random.randn(10, pos_dim)
        else:
            seq_dim = input_dim
            pos = None
        
        Km = np.random.randn(input_dim, qk_dim)
        Qm = np.random.randn(input_dim, qk_dim)
        Vm = np.random.randn(input_dim, v_dim)
        
        seq = np.random.randn(4, seq_dim)
        
        try:
            _run_equivalence_test(
                NumpyTransformer, PytorchTransformer,
                Km, Qm, Vm, pos, seq, max_seq_len=10
            )
        except ValueError as e:
            raise AssertionError(f"Config {i} failed: {config}") from e
    
    print(f'✓ All {len(test_configs)} dimension variation tests passed!')


def generate_grading_outputs(
    PytorchTransformer,
    seed: int = 1998,
) -> dict:
    """Generate outputs for grading.
    
    Args:
        PytorchTransformer: The PytorchTransformer class from the notebook.
        seed: Random seed for reproducibility.
    
    Returns:
        Dictionary containing grading outputs.
    """
    set_random_seed(seed)
    
    test_transformer = PytorchTransformer(7, 4, 3, 2, 9)
    o = test_transformer(torch.randn(8, 7))
    
    outputs = {
        "torch_transformer_shape": list(o.shape),
        "torch_transformer_value": o.view(-1).tolist()[2:7],
        "torch_transformer_init": inspect.getsource(PytorchTransformer.__init__),
        "torch_transformer_forward": inspect.getsource(PytorchTransformer.forward),
    }
    
    return outputs

