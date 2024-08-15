# Utils
from os.path import join as path_join
from os import makedirs
from typing import Dict, Tuple
from trainUtils import models

EMNIST_SPLITS = ('byclass', 'bymerge', 'balanced', 'letters', 'digits', 'mnist')
EMNIST_ROOT = 'datasets/emnist'

clusterings: Dict[str, Dict[str, Tuple[Tuple[int]]]] = {
    "mnist": {
        "binary_ordered": ((0, 1, 2, 3, 4), (5, 6, 7, 8)),
        "even_odd": ((0, 2, 4, 6, 8), (1, 3, 5, 7, 9)),
        "even_odd_pairs": ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9)),
        "similar_pairs": ((0, 8), (1, 7), (6, 9), (3, 5), (2, 4))
    },
    "digits": {
        "binary_ordered": ((0, 1, 2, 3, 4), (5, 6, 7, 8)),
        "even_odd": ((0, 2, 4, 6, 8), (1, 3, 5, 7, 9)),
        "even_odd_pairs": ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9)),
        "similar_pairs": ((0, 8), (1, 7), (6, 9), (3, 5), (2, 4))
    }
}

def clustering_to_map(key:str):
    list_of_list = clusterings[key]
    return {val: k for k, val_list in enumerate(list_of_list) for val in val_list}

def apply_clustering(df):
    pass


def main(args, exps_folder, fraction):
    from torchvision import datasets, transforms
    import torch
    from torch.utils.data import Subset
    from trainUtils import run_on_dataset, device
    from dataUtils import dataset_sampler
    import numpy as np
    
    # At the time of writing, the main url for downloading the data is broken. 
    # So we need this one to be able to download the EMNSIT files in order to build the datasets.
    # If they fix this problem when you can comment the line bellow.
    datasets.EMNIST.url = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        ])
    
    # Load the EMNIST dataset
    train_dataset = datasets.EMNIST(
        root=EMNIST_ROOT,
        split=args.emnist_split,
        train=True,
        transform=transform,
        download=True
    )

    test_dataset = datasets.EMNIST(
        root=EMNIST_ROOT,
        split=args.emnist_split,  # Use the same split as for the train dataset
        train=False,
        transform=transform,
        download=True
    )
    
    print(f"{'='*10} EMNIST - {args.emnist_split} {'='*10}")
    
    # Get the unique classes
    unique_classes = np.unique(train_dataset.targets)
    n_classes = len(unique_classes)

    # Randomly shuffle the classes and split into two groups
    np.random.shuffle(unique_classes)
    half_point = n_classes // 2
    underrepresented_classes = unique_classes[:half_point]
    regular_classes = unique_classes[half_point:]

    print(f"> Unique classes: {n_classes}")
    print(f"> Classes that will be underrepresented: {underrepresented_classes}")
    print(f"> Classes that will be downsampled: {regular_classes}")

    # We will use a constant sample from the test dataset and garanty that it is the same accross different runs using a fixed arbitrary seed.
    rndGen = torch.Generator(device=device).manual_seed(42)
    test_subset_indices = torch.randperm(len(test_dataset), generator=rndGen, device=device)[:10**4]
    sampled_test_dataset = Subset(test_dataset, test_subset_indices)

    def sample_method(dataset, num_samples):
        return dataset_sampler(dataset, num_samples, fraction, underrepresented_classes, regular_classes)
    
    def model_gen():
        return models[args.model](input=28*28, output=n_classes)
    
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


def validate_args(args, clusterings):
    split = args.emnist_split
    val = args.clustering
    if not val in clusterings[split]:
        raise ValueError(f"--clustering should be one of {', '.join(clusterings[split])}")


if __name__ == "__main__":
    import argparse
    import yaml
    import time
    timestr = time.strftime("%Y-%m-%d-%H-%M-%S")
    from argparse import RawTextHelpFormatter

    sep = ', \n\t'
    # Define hyperparameters
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("--learning_rate", nargs="+", type=float, default=[0.001])
    parser.add_argument("--weight_decay", nargs="+", type=float, default=[0.0001])
    parser.add_argument("--scaling_factor", type=int, default=8)
    parser.add_argument("--total_steps", type=int, default=10**5)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--fraction", nargs="+", type=float, default=[0.])
    parser.add_argument("--num_samples", nargs="+", type=int, default=[100, 200, 300, 400, 500, 600])
    parser.add_argument("--clustering", type=str, required=True,
                        help= "Class aggregation to create super-class. \n" +
                        "For each --emnist_split option there is a specific set of available clusterings. \n" +
                        "\n".join([ f"If --emnist_split is '{s}', --clustering should be one of: \n\t{sep.join(vals)}. " for s, vals in clusterings.items()]) )
    parser.add_argument("--eval_frequency", type=int, default=100)
    parser.add_argument("--train_seed", type=int, default=42)
    parser.add_argument("--emnist_split", type=str, choices=EMNIST_SPLITS, default='mnist', help="EMNIST data split used in the experiment")
    parser.add_argument("--exps_folder", type=str, default=f"experiments/mnist{timestr}")
    parser.add_argument("--model", type=str, choices = models.keys(), default="mlp", help="DL model to be used in this experiment instance")
    parser.add_argument("--requireGPU", action='store_true', help="Only runs if there is a GPU available")

    args = parser.parse_args()
    
    try:
        validate_args(args, clusterings=clusterings)
    except ValueError as e:
        parser.error(e)

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

