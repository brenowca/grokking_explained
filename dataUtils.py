# Basic data manipulation
from collections import defaultdict
import pandas as pd
import numpy as np

# Data and visualization
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# Deep Learning
import torch
from torch.utils.data import Dataset, Subset, DataLoader

# Utils
from tqdm import tqdm
import random
from os.path import isfile
from io import BufferedReader
import pickle
from typing import Union


class ClusteredEMNIST(torch.utils.data.Dataset):
    def __init__(self, original_dataset, encoder, clustering, num_unique_labels, clusters_per_class=None, device=None, batch_size=64):
        self.original_dataset = original_dataset
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "mps" 
                      if torch.backends.mps.is_available() else "cpu")
        self.sub_class, self.idx_col = self._generate_new_column(
            original_dataset, encoder, clustering, num_unique_labels, clusters_per_class, batch_size)
        assert len(original_dataset), len(self.sub_class)

    @classmethod
    def load(cls, file: Union[str, BufferedReader]) -> "ClusteredEMNIST":
        if not isinstance(file, BufferedReader):
            with open(file, 'rb') as f:
                return pickle.load(f)
        else:
            return pickle.load(file)

    @property
    def targets(self):
        return self.original_dataset.targets

    @property
    def classes(self):
        return self.original_dataset.classes

    def __len__(self):
        return len(self.idx_col)

    def __getitem__(self, index):
        # Get the original data and target
        img, target = self.original_dataset[self.idx_col[index]]
        # You can modify this logic as needed
        sub_class = self.sub_class[index]

        # Step 5: Return a tuple with the original data and the new column
        return img, target, sub_class

    def _generate_new_column(self, original_dataset, encoder, clustering, num_unique_labels, clusters_per_class=None, batch_size=64):
        # Prepare DataLoader
        dataloader = DataLoader(
            original_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

        new_column = []

        # Step 1: Process data in batches
        digit_tensors = {digit: [] for digit in range(num_unique_labels)}
        digit_indices = {digit: [] for digit in range(num_unique_labels)}

        for batch_idx, (imgs, targets) in enumerate(dataloader):
            imgs = imgs.to(self.device)
            # Assume encoder is a model that extracts features
            features = encoder(imgs)

            for i, target in enumerate(targets):
                target = target.item()
                digit_tensors[int(target)].append(features[i].cpu())
                digit_indices[int(target)].append(batch_idx * batch_size + i)

        # Step 2: Predict new_column for each digit using the clustering model
        for digit in range(num_unique_labels):
            if digit_tensors[digit]:
                # Stack the images to pass through the model
                tensor_stack = torch.stack(digit_tensors[digit])

                # Predict new_column values using the clustering model
                new_column_values = clustering[digit].predict(
                    tensor_stack.cpu().detach().numpy().astype('float'))

                # Assign the predicted values to the corresponding indices
                for i, idx in enumerate(digit_indices[digit]):
                    value = new_column_values[i].item()
                    keep_item = True
                    if clusters_per_class is not None:
                        keep_item = ((value == clusters_per_class[digit]) or (value in clusters_per_class[digit]))
                    if keep_item:
                        new_column.append((idx, value))

        # Step 3: Sort the new_column list by the original index order
        new_column.sort(key=lambda x: x[0])

        # Extract just the new column values in order
        idx_col, out_column = list(zip(*new_column))
        return out_column, idx_col
    
def sample_subclasses(dataset, f, total_samples, unique_classes, unique_subclasses):
    """
    Samples a specific number of indices from a dataset, ensuring a balanced distribution 
    across different subclasses within each unique class.

    Parameters:
    -----------
    dataset : list or dataset-like object
        The dataset to sample from, where each element is expected to be a tuple containing 
        at least a target class and a subclass identifier.

    f : float
        A factor (0 <= f <= 1) determining the proportion of samples to draw from each subclass 
        within a class. If the dataset has two subclasses, `f` controls the proportion of samples 
        taken from the first subclass. If `f` is less than 0.5 and the first subclass is larger, 
        or if `f` is greater than 0.5 and the second subclass is larger, `f` is adjusted 
        accordingly to maintain balance. If the dataset has more than two subclasses, `f`is ignored
        and the largest subclass is used.

    total_samples : int
        The total number of samples to draw from the dataset across all classes.

    unique_classes : list or iterable
        A list or iterable of unique class identifiers present in the dataset.

    unique_subclasses : list or iterable
        A list or iterable of unique subclass identifiers. The function handles the case 
        where there are either 2 subclasses per class or more than 2 subclasses.

    Returns:
    --------
    sampled_indices : list
        A list of indices representing the sampled entries from the dataset. The returned 
        indices ensure that the desired balance across subclasses is maintained based on 
        the specified factor `f`.

    Notes:
    ------
    - When there are exactly two subclasses, the function adjusts the sampling factor `f` 
      to ensure that a balanced number of samples are drawn from the larger subclass if `f` 
      is not set appropriately.
    - If there are more than two subclasses, the function samples only from the largest subclass 
      within each unique class.
    """
    # Group indices by target and subclass
    grouped_indices = {class_id: {subclass_id:[] for subclass_id in unique_subclasses} for class_id in unique_classes}

    for i, (_, target, sub_class) in enumerate(dataset):
        grouped_indices[int(target)][int(sub_class)].append(i)

    # Sample indices according to the factor f
    sampled_indices = []

    for class_id in range(10):
        indices_raw = [ (i,seq) for i,seq in grouped_indices[class_id].items() ]
        if len(unique_subclasses) == 2:
            indices_0 = grouped_indices[class_id][0]
            indices_1 = grouped_indices[class_id][1]
            
            if len(indices_0) > len(indices_1) and f < 0.5:
                f = 1-f
            elif len(indices_0) < len(indices_1) and f > 0.5:
                f = 1-f

            # Calculate the number of samples needed
            num_samples_0 = int(min(total_samples * f /len(unique_classes), len(indices_0))) #int(len(indices_0) * f)
            num_samples_1 =  int(min(total_samples * (1 - f) / len(unique_classes), len(indices_1))) #int(len(indices_1) * (1 - f))

            # Sample the required number of indices
            sampled_0 = random.sample(indices_0, num_samples_0)
            sampled_1 = random.sample(indices_1, num_samples_1)

            # Combine the samples
            sampled_indices.extend(sampled_0 + sampled_1)
        else:
            indices = [ seq for _, seq in sorted(indices_raw, key=lambda x: x[0])]
            largest_cluster = sorted(indices_raw, key=lambda x: x[1])[0][0]
            
            num_samples = int(min(total_samples/len(unique_classes), len(indices[largest_cluster]))) #int(len(indices_0) * f)
            sampled_indices.extend(random.sample(indices[largest_cluster], num_samples))
            

    # Optionally shuffle the final sampled indices
    random.shuffle(sampled_indices)

    # Create a Subset dataset with the sampled indices
    sampled_dataset = Subset(dataset, sampled_indices)

    return sampled_dataset


def sample_classes(doubleit_class_map, fraction, random_seed=None):
    if random_seed is not None:
        myRandom = random.Random(random_seed)
    else:
        myRandom = random

    classes = []
    for super_class, class_values in doubleit_class_map.items():
        num_classes_to_select = int(max(0, len(class_values) * fraction))
        selected_classes = myRandom.sample(class_values, num_classes_to_select)
        classes.extend(selected_classes)

    return classes


class SyntheticDataset(Dataset):
    def __init__(self, df, target_name='super_class', class_name='class', target_values=None, class_values=None):
        self.dataframe = df
        # Exclude 'class' and 'super_class' columns
        self.feature_columns = list(f'd{i}' for i in range(0, 117))

        self.target_name = target_name
        self.class_name = class_name

        self.target_values = target_values if target_values is not None else df[self.target_name].unique(
        ).tolist()
        self.class_values = class_values if class_values is not None else df[self.class_name].unique(
        ).tolist()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        features = torch.Tensor(
            self.dataframe.loc[idx, self.feature_columns].values.astype('float32'))
        target_label = torch.Tensor(
            [self.dataframe.loc[idx, self.target_name]]).type(torch.LongTensor)
        # target = torch.zeros(len(self.target_values))
        # target[target_index]=1
        return features, target_label

# Sample size per class (assuming you want the same number of samples for each class)


def balanced_df_sample(df, num_samples, target, random_state=42):
    num_classes = len(df[target].unique())
    sample_size_per_class = num_samples//num_classes
    print(f"Sample size per class size: {sample_size_per_class}")
    # Create an empty DataFrame to store the balanced samples
    balanced_df = pd.DataFrame(columns=df.columns)

    # Iterate through unique 'class' values
    for class_label in df[target].unique():
        # Sample 'sample_size_per_class' random samples for each 'class'
        class_samples = df[df[target] == class_label].sample(
            sample_size_per_class, random_state=random_state)
        # print(f"df[df[{target}] == {class_label}].sample({sample_size_per_class}, random_state = {random_state})")
        # print(class_samples)
        if len(class_samples):
            balanced_df = pd.concat([balanced_df, class_samples], axis=0)
        else:
            print(f"Empty class {class_label}")

    # Reset the index of the balanced DataFrame
    balanced_df.reset_index(drop=True, inplace=True)
    print(f"Balanced dataset size {len(balanced_df)}")
    return balanced_df


def unbalanced_df_sample(df, num_samples, target, underrepresented_classes, fraction, random_state=42):
    df_classes = df[target].unique().tolist()
    regular_classes = [
        c for c in df_classes if c not in underrepresented_classes]
    intersect_classes = [
        c for c in underrepresented_classes if c in df_classes]
    assert len(
        regular_classes) > 0, "There are as many or more underrepresented classes than classes"

    weighted_num_classes = len(regular_classes) + \
        len(intersect_classes)*(fraction)
    sample_size_per_class = num_samples/weighted_num_classes
    print(
        f"Sample size per class: {sample_size_per_class}/{len(regular_classes)} [{weighted_num_classes}]")
    # Create an empty DataFrame to store the balanced samples
    unbalanced_df = pd.DataFrame(columns=df.columns)

    # Iterate through unique 'class' values
    for class_label in df[target].unique():
        # Sample 'sample_size_per_class' random samples for each 'class'
        if class_label in intersect_classes:
            sample_size = int(np.ceil(sample_size_per_class*fraction))
        else:
            sample_size = int(sample_size_per_class)
        class_samples = df[df[target] == class_label].sample(
            sample_size, random_state=random_state)
        if len(class_samples):
            unbalanced_df = pd.concat([unbalanced_df, class_samples], axis=0)
        else:
            print(f"Empty class {class_label}")

    # Reset the index of the balanced DataFrame
    unbalanced_df.reset_index(drop=True, inplace=True)
    # print(unbalanced_df)
    print(f"Unbalanced dataset size {len(unbalanced_df)}")
    return unbalanced_df


def balanced_double_it_class_sampler(dataset, num_samples, target='class'):
    sampled_df = balanced_df_sample(dataset.dataframe, num_samples, target)
    return SyntheticDataset(sampled_df)


def balanced_double_it_super_class_sampler(dataset, num_samples):
    sampled_df = balanced_df_sample(
        dataset.dataframe, num_samples, 'super_class')
    return SyntheticDataset(sampled_df)


def unbalanced_double_it_class_sampler_factory(underrepresented_classes, fraction, target='class', random_state=42):
    def un(dataset: SyntheticDataset, num_samples):
        sampled_df = unbalanced_df_sample(df=dataset.dataframe,
                                          num_samples=num_samples,
                                          target=target,
                                          underrepresented_classes=underrepresented_classes,
                                          fraction=fraction,
                                          random_state=random_state)
        return SyntheticDataset(sampled_df,
                                target_name=dataset.target_name,
                                class_name=dataset.class_name,
                                target_values=list(dataset.target_values),
                                class_values=list(dataset.class_values))
    return un


def dataset_sampler(dataset, num_samples, fraction, classes_a, classes_b):
    weighted_num_classes = len(classes_a)*(fraction) + len(classes_b)
    sample_size_per_class = num_samples/weighted_num_classes

    sample_size_a = int(np.ceil(sample_size_per_class*fraction))
    sample_size_b = int(sample_size_per_class)

    # Initialize dictionaries to store indices for each class
    class_indices = defaultdict(list)

    # Collect indices for each class
    for idx, (_, label) in enumerate(torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)):
        class_indices[label.item()].append(idx)

    # Shuffle indices within each class
    for class_id in class_indices:
        np.random.shuffle(class_indices[class_id])

    # Downsample classes
    downsampled_indices = []
    for class_id in classes_a:
        # Fraction of samples per class
        downsampled_indices.extend(class_indices[class_id][:sample_size_a])
    for class_id in classes_b:
        # Original number of samples per class
        downsampled_indices.extend(class_indices[class_id][:sample_size_b])

    # Shuffle the combined indices to ensure randomness
    np.random.shuffle(downsampled_indices)

    # Create the new dataset
    new_dataset = Subset(dataset, downsampled_indices)
    return new_dataset


df_doubleit = pd.read_csv(
    "datasets/V0_binaryTreeMultiDim_distSuperClasses_ns_10000_ndg_9_ndpg_13.csv")


def doubleit_train_test(df, test_size=0.4, random_state=42):
    test_size = 0.4
    X = df.drop(["class", "super_class"], axis=1)  # Features (numeric columns)
    y = df[["class", "super_class"]]  # Target variables
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)
    # Create DataFrames for training and test data
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    return (train_df, test_df)


# If there the training split is already saved into a csv file, then load it
if isfile('datasets/V0_binaryTreeMultiDim_distSuperClasses_ns_10000_ndg_9_ndpg_13_train.csv'):
    train_df_doubleit = pd.read_csv(
        'datasets/V0_binaryTreeMultiDim_distSuperClasses_ns_10000_ndg_9_ndpg_13_train.csv')
    test_df_doubleit = pd.read_csv(
        'datasets/V0_binaryTreeMultiDim_distSuperClasses_ns_10000_ndg_9_ndpg_13_test.csv')
# If there is no training and test split saved: split the dataset into training and test
else:
    test_size = 0.4
    random_state = 42
    train_df_doubleit, test_df_doubleit = doubleit_train_test(
        df_doubleit, test_size=test_size, random_state=random_state)


def plot_dist_grid_list(df, col_list, cols_per_row, class_col, fraction=0.1, x_min=None, x_max=None, figsize=(16, 12), **kwargs):
    # Set the style of the plot
    sns.set(style="whitegrid")
    num_cols = len(col_list)

    # Create subplots
    fig, axes = plt.subplots(num_cols // cols_per_row + 1,
                             cols_per_row, figsize=figsize, **kwargs)

    # Flatten the axes array
    axes = axes.flatten()

    # Loop through numerical columns and create distribution plots
    sampled_df = df.sample(frac=fraction)
    for i, coln in tqdm(list(enumerate(col_list)), "Create distribution plots"):
        ax = axes[i]
        col = f'd{coln}'
        sns.histplot(data=sampled_df, x=col, hue=class_col, kde=True, ax=ax,
                     common_norm=False, legend=False)  # legend= i==cols_per_row-1)
        ax.set_title(f'Distribution of {col}')
        ax.set_ylabel('')

        # Set the common x-axis limit for all subplots
        if (x_min is not None and x_max is not None):
            ax.set_xlim(x_min, x_max)

    # Remove empty
    for i in tqdm(range(num_cols, len(axes)), "Remove empty subplots"):
        fig.delaxes(axes[i])

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


def plot_dist_grid(df, num_dims, num_cols, cols_per_row, class_col, fraction=0.1, x_min=None, x_max=None, figsize=(16, 12), **kwargs):
    col_list = list(np.random.choice(num_dims+1, num_cols, replace=False))
    plot_dist_grid_list(df, col_list, cols_per_row, class_col,
                        fraction, x_min, x_max, figsize, **kwargs)


def pairplot(dim_cols, class_col, fraction=0.01):
    dim_cols = [f"d{i}" for i in dim_cols]
    sampled_df = df_doubleit.sample(frac=fraction)

    # Set the style of the plot
    sns.set(style="ticks")

    # Create a pairplot for pairwise correlations
    sns.pairplot(data=sampled_df, vars=dim_cols,
                 hue=class_col, diag_kind="hist")

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()


def flatten_training_results(results):
    # Create a flat list to hold the flattened data
    flat_data = []

    # Flatten the nested lists within the 'test_accuracy' and 'train_accuracy' keys
    for item in results:
        lr = item["lr"]
        weight_decay = item["weight_decay"]
        scaling_factor = item["scaling_factor"]
        num_samples = item["num_samples"]
        optimization_steps = item["optimization_steps"]

        for (step, test_accuracy), (_, train_accuracy), (_, train_loss), (_, weight_norm) in zip(item["test_accuracy"], item["train_accuracy"], item["train_loss"], item["weight_norms"]):
            # for (step, test_accuracy), (_,train_accuracy) in zip(item["test_accuracy"], item["train_accuracy"]):
            flat_data.append({
                "lr": lr,
                "weight_decay": weight_decay,
                "scaling_factor": scaling_factor,
                "num_samples": num_samples,
                "optimization_steps": optimization_steps,
                "step": step,
                "test_accuracy": test_accuracy,
                "train_accuracy": train_accuracy,
                "train_loss": train_loss,
                "weight_norm": weight_norm
            })
    df = pd.DataFrame(flat_data)
    return df


def plot_flat_training_results(df, file_path=None):
    # Create a grid of subplots
    g = sns.FacetGrid(df, col="num_samples",
                      row="weight_decay", margin_titles=True)

    # Plot test_accuracy and train_accuracy data in each subplot
    g.map(sns.lineplot, "step", "test_accuracy", color="blue",
          label="Test Accuracy", markers=True, errorbar=None)
    g.map(sns.lineplot, "step", "train_accuracy", color="orange",
          label="Train Accuracy", markers=True, errorbar=None)

    g.set_titles(
        col_template="Num Samples = {col_name}", row_template="Weight Decay = {row_name}")

    # Add a legend
    g.add_legend()

    # Show the plot
    plt.show()
    if file_path != None:
        g.savefig(file_path)
