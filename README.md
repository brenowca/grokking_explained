# Grokking Explained

This project contains a collection of small experiments analysing **grokking** behaviour in synthetic datasets and on MNIST.  The repository mixes stand-alone Python scripts and several Jupyter notebooks that contain additional helper functions and analysis.  Below is a short overview of how things are organised and how to run the experiments.

## Repository Layout

```
.├── trainUtils.py          # Model implementations and training helpers
.├── grokking_mnist.py      # Experiments on EMNIST/MNIST data
.├── grokking_unbalanced.py # Experiments on custom CSV datasets
.├── datasets/              # Synthetic datasets used by the scripts
.├── exp_*/                 # Folders containing experiment outputs
└── *.ipynb                 # Notebooks with extra utilities and analysis
```

## How to Reproduce the Experiments

### Set up the Environment
- Create a virtual environment (conda, `venv` or pyenv)
- Install the requirements manually.  The repository does not ship a `requirements.txt` so you will need typical packages such as `torch`, `pandas`, `numpy`, `scikit-learn` and `jupyter`.

> If you are using a Mac M1, M2 or M3 you might need to install PyTorch Geometric from source using the instructions in [their webpage](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#installation-from-source).

### Run the Notebooks
The notebooks contain helper functions, data-generation code and further analysis (`grokking_data.ipynb`, `Clustering.ipynb`, etc.).  Start Jupyter with:

```bash
python -m jupyter notebook --notebook-dir=.
```

### Run the Experiment Scripts
Longer running experiments can be launched directly from the command line.  For example, to train on MNIST:

```bash
python grokking_mnist.py --requireGPU --total_steps 100000
```

Both `grokking_mnist.py` and `grokking_unbalanced.py` expose a number of command line arguments (learning rate, batch size, dataset path, etc.) so you can sweep over different configurations.

### Opening the Notebooks as Slides

```bash
jupyter nbconvert grokking_data.ipynb --to slides --post serve
```

## What to Explore Next

- **Inspect the notebooks** for helper functions and how datasets are built.
- **Look at the datasets** under `datasets/` to understand the CSV format and the `class`/`super_class` labels.
- **Review `trainUtils.py`** to see how models and training loops are implemented.
- **Check the experiment outputs** inside `exp_*` directories for metrics and saved checkpoints.
- Scripts import some utilities from a missing `dataUtils` module; those functions are defined inside the notebooks.

## Results Summary

[TODO] extract the results from the notebook section and append them here

---
## Citing the Paper
```bibtex
bibtex
```

