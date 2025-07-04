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
The notebooks contain helper functions, data-generation code and further analysis (`grokking_data.ipynb`, `Clustering.ipynb`, etc.).
Each notebook or script covers part of the analysis or experiments showed in the paper.

1. Description of the synthetic datasets
    > Plots to better grasp the properties of the datasets
    > * [Synthetic equidistant dataset creation](1_creating_and_sampling_supergroups_equidistants.ipynb)
    > * [Synthetic equivariant dataset creation](1_creating_and_sampling_supergroups_equivariant.ipynb)
2. Experiments on the synthetic equidistant dataset
    > The notebook presents only one run for each kind of experiment. The experiments in the paper use multiple runs using the [unbalanced sampling script](grokking_unbalanced.py) and  the [balanced sampling script](grokking_balanced.py).
3. Experiments on the synthetic equivariant dataset
    > The notebook presents only one run for each kind of experiment. The experiments in the paper use multiple runs using the [unbalanced sampling script](grokking_unbalanced.py) and  the [balanced sampling script](grokking_balanced.py).
4. Initial experiments on MNIST
    > Exploratory analysis of the MNIST dataset
5. Clustering Extended MNIST classes
    > Process to create different clusters for each class, visualizing the result and defining the most separated cluster [Clustering notebook](5_Clustering_Extended_MNIST_classes.ipynb).
6. Experiments on clustered MNIST classes
    > Similar set of experiments than the previous ones, but on clustered MNIST.


#### How to run the notebooks

To initialize your jupyter notebook server and run the notebooks you can do the following:
```bash
python -m jupyter notebook --notebook-dir=.
```

#### Opening the notebooks as slide presentations
```bash
jupyter nbconvert <notebook name> --to slides --post serve
```

### Run the experiments scripts
Some experiments are not performed in notebooks, because they take hours in a cluster structure, so the chances that my connection was interrupted in the middle of the process is high, also I wanted to start multiple experiments in parallel.
You can get a description of every script parameters by running `python <scriptname.py> --help` where `<scriptname.py>`stands for the path to the specific script being used. E.g.:
```bash
python grokking_mnist.py --requireGPU --total_steps 100000
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

