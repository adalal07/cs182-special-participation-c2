# Migration Guide: Code Extraction to Separate Modules

This guide explains the new structure after extracting helper functions and tests into separate modules.

## New Structure

```
hand_transformer/
├── utils/
│   ├── __init__.py          # Package exports
│   ├── constants.py         # Configuration classes and constants
│   └── helper.py            # Helper functions (training, visualization, comparison)
├── tests/
│   ├── __init__.py          # Test package exports
│   ├── test_equivalence.py  # Main equivalence tests
│   ├── test_with_positional_encoding.py
│   ├── test_without_positional_encoding.py
│   ├── test_dimension_variations.py
│   └── test_grading.py      # Grading output generation
├── q_hand_transformer.ipynb # Main notebook (simplified)
└── README.md
```

## Changes to Notebook

### Cell 1: Imports

The notebook now imports everything from the `utils` and `tests` packages:

```python
from utils import (
    PlotConfig, TrainingConfig, PLOT_CONFIG, TRAIN_CONFIG,
    RELATIVE_TOLERANCE, TEST_ITERATIONS, etc.,
    set_random_seed, rescale_and_plot, save_figure,
    train_loop, compare_transformers, etc.
)

from tests import (
    test_transformer_equivalence,
    test_with_positional_encoding,
    test_without_positional_encoding,
    test_dimension_variations,
    generate_grading_outputs,
)
```

### Updated Function Signatures

#### `train_loop()`

**Old:**
```python
transformer_py, loss = train_loop(make_batch, input_dim, qk_dim, v_dim, ...)
```

**New:**
```python
transformer_py, loss = train_loop(
    make_batch, input_dim, qk_dim, v_dim,
    PytorchTransformer,  # Add transformer_class parameter
    pos_dim=..., max_seq_len=..., ...
)
```

The `transformer_class` parameter is now **required** and must be passed as the 4th positional argument (after `v_dim`).

#### `compare_transformers()`

**Old:**
```python
out_hand, out_learned = compare_transformers(np_transformer, py_transformer, seq)
```

**New:**
```python
out_hand, out_learned = compare_transformers(
    np_transformer, py_transformer, seq,
    NumpyTransformer,  # Add numpy_transformer_class parameter
    plot=True, save_plot=False, run_name=None
)
```

The `numpy_transformer_class` parameter is now **required** and must be passed as the 4th positional argument.

### Test Function

The `test()` function in the notebook is now a wrapper that calls the test functions from the `tests` package. It requires `NumpyTransformer` and `PytorchTransformer` to be defined in the notebook before calling.

## Manual Cleanup Required

After the extraction, Cell 1 may still contain some duplicate code that needs to be removed:

1. **Remove old configuration classes**: `PlotConfig`, `TrainingConfig` (now in `utils/constants.py`)
2. **Remove old constants**: `RELATIVE_TOLERANCE`, `TEST_ITERATIONS`, etc. (now in `utils/constants.py`)
3. **Remove old helper functions**: `set_random_seed`, `rescale_and_plot`, `save_figure`, `train_loop`, `compare_transformers` (now in `utils/helper.py`)
4. **Remove old test function implementation**: The full `test()` implementation (now uses imported test functions)

Cell 1 should only contain:
- Imports from standard library
- Imports from `utils` package
- Imports from `tests` package
- `TO_SAVE` dictionary initialization
- The `test()` wrapper function

## Updating Existing Code

### Example 1: Identity Task

**Before:**
```python
transformer_py, loss = train_loop(
    make_batch_identity, 
    input_dim=len(A), 
    qk_dim=Km.shape[1], 
    v_dim=Vm.shape[1]
)
compare_transformers(np_transformer, transformer_py, seq)
```

**After:**
```python
transformer_py, loss = train_loop(
    make_batch_identity,
    input_dim=len(A),
    qk_dim=Km.shape[1],
    v_dim=Vm.shape[1],
    PytorchTransformer  # Add this
)
compare_transformers(
    np_transformer, 
    transformer_py, 
    seq,
    NumpyTransformer  # Add this
)
```

### Example 2: Position-Based Task

**Before:**
```python
transformer_py, loss = train_loop(
    make_batch_first,
    input_dim=len(A),
    qk_dim=Km.shape[1],
    v_dim=Vm.shape[1],
    pos_dim=pos_dim,
    max_seq_len=pos.shape[0]
)
```

**After:**
```python
transformer_py, loss = train_loop(
    make_batch_first,
    input_dim=len(A),
    qk_dim=Km.shape[1],
    v_dim=Vm.shape[1],
    PytorchTransformer,  # Add this
    pos_dim=pos_dim,
    max_seq_len=pos.shape[0]
)
```

## Benefits

1. **Cleaner Notebook**: Cell 1 is now much shorter and focused on imports
2. **Reusable Code**: Helper functions can be used in other notebooks or scripts
3. **Better Testing**: Tests are organized into focused unit test files
4. **Easier Maintenance**: Changes to helpers don't require editing the notebook
5. **Professional Structure**: Follows standard Python project organization

## Running Tests

You can now run individual test files:

```python
# In the notebook, after defining NumpyTransformer and PytorchTransformer:
from tests import test_transformer_equivalence

test_transformer_equivalence(NumpyTransformer, PytorchTransformer)
```

Or run all tests:

```python
from tests import (
    test_transformer_equivalence,
    test_with_positional_encoding,
    test_without_positional_encoding,
    test_dimension_variations,
)

test_transformer_equivalence(NumpyTransformer, PytorchTransformer)
test_with_positional_encoding(NumpyTransformer, PytorchTransformer)
test_without_positional_encoding(NumpyTransformer, PytorchTransformer)
test_dimension_variations(NumpyTransformer, PytorchTransformer)
```

## Troubleshooting

### Import Errors

If you get import errors, make sure:
1. The `utils` and `tests` directories exist
2. The `__init__.py` files are present in both directories
3. You're running the notebook from the `hand_transformer` directory

### Function Signature Errors

If you get errors about missing arguments:
- Check that you're passing `PytorchTransformer` to `train_loop()`
- Check that you're passing `NumpyTransformer` to `compare_transformers()`

### Test Function Errors

If `test()` fails with "NumpyTransformer not found":
- Make sure you've defined `NumpyTransformer` and `PytorchTransformer` in the notebook before calling `test()`
- The classes must be in the notebook's namespace, not just imported

