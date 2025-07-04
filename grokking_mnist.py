# Utils
from os.path import join as path_join
from os import makedirs
from typing import Dict
from trainUtils import models



clusterings: Dict[str, Dict[str, str]] = {
    "mnist": {
        "train": "clustering_mnist/train_mnist_dataset.pkl",
        "test": "clustering_mnist/test_mnist_dataset.pkl"
    },
    "mnist_10": {
        "train": "clustering_mnist/train_mnist_10_dataset.pkl",
        "test": "clustering_mnist/test_mnist_10_dataset.pkl"
    },
    "mnist_5": {
        "train": "clustering_mnist/train_mnist_5_dataset.pkl",
        "test": "clustering_mnist/test_mnist_5_dataset.pkl"
    },
    "mnist_10_all": {
        "train": "clustering_mnist/train_mnist_10_dataset.pkl",
        "test": "clustering_mnist/test_mnist_all_dataset.pkl"
    },
    "mnist_all": {
        "train": "clustering_mnist/train_mnist_all_dataset.pkl",
        "test": "clustering_mnist/test_mnist_all_dataset.pkl"
    },
    "letters": {
        "train": "clustering_mnist/train_mnist_dataset.pkl",
        "test": "clustering_mnist/test_mnist_dataset.pkl"
    }
}

# EMNIST_SPLITS = ('byclass', 'bymerge', 'balanced', 'letters', 'digits', 'mnist', 'mnist_10')
EMNIST_SPLITS = clusterings.keys()
EMNIST_ROOT = 'datasets/emnist'


def main(args, exps_folder, fraction):
    import torch
    from torch.utils.data import Subset
    from trainUtils import run_on_dataset, device
    from dataUtils import sample_subclasses, ClusteredEMNIST
    import numpy as np
    
    # Load clustered dataset
    train_dataset = ClusteredEMNIST.load(clusterings[args.emnist_split]['train'])
    test_dataset = ClusteredEMNIST.load(clusterings[args.emnist_split]['test'])
    
    print(f"{'='*10} EMNIST - {args.emnist_split} {'='*10}")
    
    # We will use a constant sample from the test dataset and garanty that it is the same accross different runs using a fixed arbitrary seed.
    rndGen = torch.Generator(device=device).manual_seed(42)
    test_subset_indices = torch.randperm(len(test_dataset), generator=rndGen, device=device)[:10**4]
    sampled_test_dataset = Subset(test_dataset, test_subset_indices)
    
    # Get the unique classes
    unique_classes = np.unique(train_dataset.targets)
    n_classes = len(unique_classes)
    
    if args.num_samples is None:
        sample_method = None
    else:
        unique_subclasses = np.unique(train_dataset.sub_class)
        # unique_sub_classes = np.unique(train_dataset.sub_class)
        def sample_method(dataset, num_samples):
            return sample_subclasses(dataset, fraction, num_samples, unique_classes, unique_subclasses)
    
    def model_gen():
        return models[args.model](input=3*32*32, output=n_classes)
    
    mnist_results = run_on_dataset(model_init=model_gen, # Image shape
                         list_num_samples=args.num_samples,
                         sample_method= sample_method,
                         lr_values=args.learning_rate,
                         weight_decay_values=args.weight_decay,
                         scaling_factor=args.scaling_factor,
                         eval_frequency=args.eval_frequency,
                         batch_size=args.batch_size,
                         total_steps=args.total_steps,
                         train_dataset=train_dataset,
                         test_dataset=sampled_test_dataset,
                         results_folder=exps_folder)
    return mnist_results

if __name__ == "__main__":
    import argparse
    import yaml
    import time
    timestr = time.strftime("%Y-%m-%d-%H-%M-%S")

    # Define hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", nargs="+", type=float, default=[0.001])
    parser.add_argument("--weight_decay", nargs="+", type=float, default=[0.0001])
    parser.add_argument("--scaling_factor", type=int, default=8)
    parser.add_argument("--total_steps", type=int, default=10**5)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--fraction", nargs="+", type=float, default=[0.])
    parser.add_argument("--num_samples", nargs="+", type=int, default=None)
    parser.add_argument("--eval_frequency", type=int, default=100)
    parser.add_argument("--train_seed", type=int, default=42)
    parser.add_argument("--emnist_split", type=str, choices=EMNIST_SPLITS, default='mnist', help="EMNIST data split used in the experiment")
    parser.add_argument("--exps_folder", type=str, default=f"experiments/mnist{timestr}")
    parser.add_argument("--model", type=str, choices = models.keys(), default="mlp", help="DL model to be used in this experiment instance")
    parser.add_argument("--requireGPU", action='store_true', help="Only runs if there is a GPU available")

    args = parser.parse_args()

    if args.requireGPU:
        import torch
        assert torch.cuda.is_available(), "No CUDA available!"
    
    # Write parameters to YAML file
    makedirs(args.exps_folder, exist_ok=True)
    hyperparams_path = path_join(args.exps_folder, "hyperparams.yaml")
    with open(hyperparams_path, "w") as file:
        print(vars(args))
        yaml.dump(vars(args), file)
    print(f"Hyperparameters saved to {hyperparams_path}")
    
    # This package imports torch wich takes a lot of time if loaded in the begining of the script
    from dataUtils import flatten_training_results, plot_flat_training_results
    
    for fraction in args.fraction:
        # Write parameters to YAML file
        params = vars(args)
        params["fraction"] = [fraction]
        exps_folder = path_join(args.exps_folder, f"{args.emnist_split}_{[fraction]}frac")
        print(f"{'-'*10} Fraction - {fraction} {'-'*10}")
        makedirs(exps_folder, exist_ok=True)

        mnist_results = main(args, exps_folder, fraction)
    
        df = flatten_training_results(mnist_results)
        df_path = path_join(exps_folder, 'results.csv')
        df.to_csv(df_path)
            
        plot_flat_training_results(df, path_join(
            exps_folder, 'acc_by_steps.pdf'))

