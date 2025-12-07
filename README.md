# Hand Transformer Implementation

This notebook implements a simplified Transformer model from scratch to help you understand how Transformers work. You'll implement both NumPy (reference) and PyTorch (learnable) versions of a single-head, single-layer Transformer.

## Setup Options

You have two options for running this notebook:

### Option 1: Google Colab (Recommended for Quick Start)

**No setup required!** The notebook works out-of-the-box in Google Colab.

1. Open the notebook in Google Colab:
   - Upload `q_hand_transformer.ipynb` to Google Colab, or
   - Use the "File" → "Upload notebook" option in Colab

2. Run all cells sequentially (Cell → Run All)

3. **Note:** A GPU is not necessary for this task. If you're using Colab, you can select "Runtime" → "Change runtime type" and choose "None" as the hardware accelerator.

### Option 2: Local Setup with Conda

If you prefer to run the notebook locally, follow these steps to set up a conda environment:

#### Prerequisites

- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your system

#### Step 1: Create a Conda Environment

Open a terminal/command prompt and navigate to the directory containing this notebook, then run:

```bash
conda create -n transformer_env python=3.9 -y
```

This creates a new conda environment named `transformer_env` with Python 3.9.

#### Step 2: Activate the Environment

```bash
conda activate transformer_env
```

#### Step 3: Install Required Packages

Install the required libraries:

```bash
# Install relevant packages
conda install numpy pytorch matplotlib -c pytorch -y
```

Or use pip if you prefer:

```bash
pip install numpy torch matplotlib
```

#### Step 4: Install Jupyter (if not already installed)

```bash
conda install jupyter notebook -y
```

Or with pip:

```bash
pip install jupyter notebook
```

#### Step 5: Launch Jupyter Notebook

```bash
jupyter notebook
```

This will open Jupyter in your browser. Navigate to `q_hand_transformer.ipynb` and open it.

#### Step 6: Verify Installation

In the first cell of the notebook, run the imports to verify everything is installed correctly:

```python
import numpy as np
import torch
import matplotlib.pyplot as plt
```

If all imports succeed without errors, you're ready to go!

## Running the Notebook

1. **Execute cells sequentially**: Run cells from top to bottom using "Run All" or execute them one by one.

2. **Important**: The same variables will be defined in different ways in various subparts of the homework. If you encounter errors stating that a variable has the wrong shape or a function is missing an argument, ensure that you have re-run the cells in that particular problem subpart.

3. **Plot Saving**: The notebook includes functionality to save plots to a local `plots/` directory. You can enable this by setting `save_plot=True` in the relevant function calls, or use the `save_figure()` function directly.

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you've activated the conda environment (`conda activate transformer_env`) before launching Jupyter.

2. **Kernel Issues**: If Jupyter doesn't see your conda environment:
   ```bash
   conda install ipykernel -y
   python -m ipykernel install --user --name transformer_env --display-name "Python (transformer_env)"
   ```
   Then restart Jupyter and select the "Python (transformer_env)" kernel.

3. **PyTorch Installation**: If you need GPU support (not required for this notebook), visit [PyTorch's installation page](https://pytorch.org/get-started/locally/) for GPU-specific instructions.

## Project Structure

```
hand_transformer/
├── README.md                    # This file
├── q_hand_transformer.ipynb    # Main notebook
└── plots/                       # Generated plots directory (created automatically)
```

## Additional Notes

- The notebook implements a simplified Transformer (single layer, single head, no residual connections, etc.) for educational purposes.
- All helper functions are consolidated in the first cell for easy reference.
- The notebook includes comprehensive assertions for debugging and validation.
- Plot configurations can be customized via the `PlotConfig` class in the first cell.

## Questions or Issues?

If you encounter any problems, please check:
1. That all dependencies are correctly installed
2. That you're using the correct Python version (3.9 recommended)
3. That you've activated the conda environment before running Jupyter

