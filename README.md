# Grokking Explained

## How to reproduce the experiments

### Set up the environment
- Install a virtual environment (use conda or pyenv)
- Install the requirements

```shell
pip install -r requirements.txt
```
> If you are using a mac M1, M2, or M3 you might need to install pytorch geometric from source using the instructions in [their webpage](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#installation-from-source).


### Run the notebooks
- [Jupyter notebook - Minibert supervised inspection](minibert_supervised_inspection.ipynb)
> To initialize your jupyter notebook server and run the notebooks you can do the following:
    ```
    python -m jupyter notebook --notebook-dir=.
    ```

### Run the experiments scripts
Some experiments are not performed in notebooks, because they take hours in a cluster machine, so the chances that my connection was interrupted in the middle of the process is high, also I wanted to start multiple experiments in parallel.
```bash
python grokking_mnist.py --requireGPU --total_steps 100000
```

### Opening the notebooks as slide presentations
```bash
jupyter nbconvert minibert_supervised_inspection.ipynb --to slides --post serve
```

## Results summary

[TODO] extract the results from the notebook section and append them here

---
## Citing the Paper
```
bibtex
```

