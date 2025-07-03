# Utils
from os.path import join as path_join
from os import makedirs
from trainUtils import DecoderTransformer, run_on_dataset, MLP


datasets = {"similar":{"train":"datasets/V0_binaryTreeMultiDim_distSuperClasses_ns_10000_ndg_9_ndpg_13_train.csv",
                        "test":"datasets/V0_binaryTreeMultiDim_distSuperClasses_ns_10000_ndg_9_ndpg_13_test.csv",
                        "all":"datasets/V0_binaryTreeMultiDim_distSuperClasses_ns_10000_ndg_9_ndpg_13.csv"},
            
            "equidistant":{"train":"datasets/V1eq_binaryTreeMultiDim_distSuperClasses_ns_10000_ndg_9_ndpg_13_train.csv",
                           "test":"datasets/V1eq_binaryTreeMultiDim_distSuperClasses_ns_10000_ndg_9_ndpg_13_test.csv",
                           "all":"datasets/V1eq_binaryTreeMultiDim_distSuperClasses_ns_10000_ndg_9_ndpg_13.csv"}}


models = {"mlp":lambda: MLP(117, 256, 4, 3),
          "2-layer_transformer": lambda: DecoderTransformer(input_dim=117, hidden_dim=128, num_layers=2, num_heads=4)}


def main(args, exps_folder, fraction):
    from dataUtils import SyntheticDataset, balanced_df_sample, sample_classes, unbalanced_double_it_class_sampler_factory
    import pandas as pd
    
    df = pd.read_csv(datasets[args.dataset]["all"])
    df_train = pd.read_csv(datasets[args.dataset]["train"])
    df_test = pd.read_csv(datasets[args.dataset]["test"])
    # df_train, df_test = doubleit_train_test(df, test_size = 0.4, random_state=42)
    df_test_small = balanced_df_sample(df_test, 10000, 'class')
    
    class_list = sorted(df['class'].unique().tolist())
    superclass_list = sorted(df['super_class'].unique().tolist())
        
    superclass_to_class = {sc: df.loc[df["super_class"] == sc, "class"].unique().tolist() for sc in superclass_list}
    train_seed = args.train_seed
    
    dataset_train = SyntheticDataset(df_train, target_name='super_class', class_name='class', target_values = superclass_list, class_values = class_list )
    dataset_test = SyntheticDataset(df_test_small, target_name='super_class', class_name='class', target_values = superclass_list, class_values = class_list )

    underrepresented_classes= sample_classes(superclass_to_class, .5, random_seed=42)
    print(f"There are {len(underrepresented_classes)} classes to be downsampled. They are: {underrepresented_classes}")
    assert len(underrepresented_classes), "There are not enough underrepresented classes"

    sample_method = unbalanced_double_it_class_sampler_factory(underrepresented_classes=underrepresented_classes, fraction=fraction, target='class', random_state=train_seed)
    
    results_mlp_unbalanced = run_on_dataset(model_init=models[args.model],
                         list_num_samples=args.num_samples,
                         sample_method= sample_method,
                         lr_values= args.learning_rate,
                         weight_decay_values=args.weight_decay,
                         scaling_factor=args.scaling_factor,
                         eval_frequency=args.eval_frequency,
                         batch_size=args.batch_size,
                         total_steps=args.total_steps,
                         train_dataset=dataset_train,
                         test_dataset=dataset_test,
                         results_folder=exps_folder)
    
    return results_mlp_unbalanced


if __name__ == "__main__":
    import argparse
    import yaml
    import time
    import hashlib
    
    # Get unique experiment name
    timestr = time.strftime("%Y-%m-%d-%H-%M-%S")
    sha = hashlib.sha1()
    sha.update(str(time.time()).encode("utf-8"))
    hash = sha.hexdigest()[:8]
    print(f"Running exp {timestr}_{hash}")

    # Define hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", nargs="+", type=float, default=[0.001])
    parser.add_argument("--weight_decay", nargs="+", type=float, default=[0.0001])
    parser.add_argument("--scaling_factor", type=int, default=8)
    parser.add_argument("--total_steps", type=int, default=10**5)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--fraction", nargs="+", type=float, default=[0.])
    parser.add_argument("--train_seed", type=int, default=42)
    parser.add_argument("--num_samples", nargs="+", type=int, default=[100, 200, 300, 400, 500, 600])
    parser.add_argument("--eval_frequency", type=int, default=100)
    
    parser.add_argument("--dataset", type=str, choices = datasets.keys(), required=True, help="Dataset to be used in this experiment instance")
    parser.add_argument("--model", type=str, choices = models.keys(), default="mlp", help="DL model to be used in this experiment instance")

    parser.add_argument("--exps_folder", type=str,
                        default=f"experiments/exp_{timestr}_{hash}")
    parser.add_argument("--requireGPU", action='store_true', help="Only runs if there is a GPU available")

    args = parser.parse_args()

    if args.requireGPU:
        import torch
        assert torch.cuda.is_available(), "No CUDA available!"

    for fraction in args.fraction:
        # Write parameters to YAML file
        params = vars(args)
        params["fraction"] = [fraction]
        exps_folder = args.exps_folder + f"_frac{[fraction]}"
        makedirs(exps_folder, exist_ok=True)
        hyperparams_path = path_join(exps_folder, "params.yaml")
        with open(hyperparams_path, "w") as file:
            print(params)
            yaml.dump(params, file)
        print(f"Hyperparameters saved to {hyperparams_path}")

        results = main(args, exps_folder, fraction)
        
        from dataUtils import flatten_training_results, plot_flat_training_results
        df = flatten_training_results(results)
        df_path = path_join(exps_folder, 'results.csv')
        df.to_csv(df_path)

        plot_flat_training_results(df, path_join(
            exps_folder, 'acc_by_steps.pdf'))
