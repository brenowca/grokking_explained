# Basic data manipulation
import numpy as np

# ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset

# Utils
from tqdm import tqdm
import os
from os.path import join as path_join
import json

# Use CUDA if available, MPS (Metal Performance Shaders) backend if available on Apple devices, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "mps" 
                      if torch.backends.mps.is_available() else "cpu")

models = {"mlp":lambda input=117, output=3: MLP(input, 256, output, 3),
          "2-layer_transformer": lambda input=117, output=10: TransformerClassifier(input_dim=input, d_model=128, num_layers=2, num_classes=output, nhead=4)}

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers):
        super(MLP, self).__init__()
        self.name= f"MLP_{hidden_size}_{output_size}_{num_hidden_layers}"
        
        # Input layer
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_size)])
        
        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            
        # Output layer
        self.out_layer = nn.Linear(hidden_size, output_size)
        
        # ReLU activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = self.relu(layer(x))
        x = self.out_layer(x)
        return x
    
    
def defaultMLPinit():
    return MLP(28 * 28, 256, 10, 3)


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, num_classes, max_seq_len=200, dim_feedforward=2048, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        params = f"{input_dim}_{d_model}_{nhead}_{num_layers}_{max_seq_len}_{dim_feedforward}"
        self.name= f"Transformer_{params}"
        
        # 1. Input Embedding/Projection Layer
        self.embedding = nn.Linear(input_dim, d_model)
        
        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification Head
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        # 3. Apply the embedding layer first
        # Input x shape: (batch_size, seq_len, input_dim)
        x = self.embedding(x.unsqueeze(1))
        # Now x shape: (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x += self.positional_encoding[:, :x.size(1), :]
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)
        
        # Classification head
        x = x.mean(dim=1)  # Pooling: mean across the sequence dimension
        x = self.fc(x)
        
        return x


def default2LTransformer():
    return TransformerClassifier(input_dim=28*28, d_model=128, num_layers=2, num_classes=10, nhead=4)


def eval_model(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device).flatten()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def save_model(model, results_folder, checkpoint_name):
    os.makedirs(results_folder, exist_ok=True)
    torch.save(model.state_dict(), path_join(results_folder, checkpoint_name))


def load_model(empty_model, folder, checkpoint_name, device):
    empty_model.load_state_dict(torch.load(path_join(folder, checkpoint_name)))
    empty_model.to(device)
    empty_model.eval()
    
    
def train_loop(model_init, device, total_steps, lr, weight_decay, scaling_factor, eval_frequency, train_loader, test_loader, train_eval_loader, results_folder=None):
    model = model_init().to(device)
                
    # Scale the model's parameters
    with torch.no_grad():
        for param in model.parameters():
            param.data *= scaling_factor
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    train_accs = []
    test_accs = []
    norms = []
    losses = []

    step = 0
    with tqdm(total=total_steps) as pbar:
        while step < total_steps:
            for data in train_loader:
                images, labels = data[0].to(device), data[1].to(device).flatten()
                model.train()
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # Evaluate the model at the specified frequency
                if (step + 1) % eval_frequency == 0:
                    test_accuracy = eval_model(model, test_loader, device)
                    train_accuracy = eval_model(model, train_eval_loader, device)
                    
                    train_accs.append([step+1, train_accuracy])
                    test_accs.append([step+1, test_accuracy])
                    
                    total_norm = sum(torch.pow(p, 2).sum() for p in model.parameters())
                    norms.append((step+1, float(np.sqrt(total_norm.item()))))
                    
                    losses.append((step+1, loss.item()))
                    
                    pbar.set_description("Acc: {0:2.1f}% | {1:2.1f}%".format(
                    train_accuracy * 100, 
                    test_accuracy * 100))
                    
                    if results_folder is not None:
                        save_model(model, results_folder, f"{model.name}_step_{step+1}.pth")
                    
                    # print(f'Step [{step + 1}/{total_step}], Loss: {loss.item():.4f}')
                step += 1
                pbar.update(1)


    # Append the results to the list
    results = {
        'lr': lr,
        'weight_decay': weight_decay,
        'scaling_factor': scaling_factor,
        'num_samples': len(train_loader.dataset),
        'optimization_steps': step,
        'test_accuracy': test_accs,
        'train_accuracy': train_accs,
        'train_loss': losses,
        'weight_norms': norms
        }
    if results_folder is not None:
        with open(path_join(results_folder, 'results.json'), 'w') as f:
            json.dump(results, f)
    return results

def run_on_mnist(model_init, list_num_samples, lr_values, weight_decay_values, scaling_factor, eval_frequency, batch_size, total_steps, results_folder=None):
    # Initialize lists to store results
    results = []

    # Define the training loop for hyperparameter tuning
    for num_samples in list_num_samples:
        # Create a random subset using RandomSampler
        subset_indices = torch.randperm(len(full_mnist_train_dataset))[:num_samples].tolist()
        
        train_dataset = Subset(full_mnist_train_dataset, subset_indices)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        train_eval_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_mnist_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        
        for lr in lr_values:
            for weight_decay in weight_decay_values:
                print(f"Num Samples: {num_samples}")
                print(f"LR: {lr}")
                print(f"Weight Decay: {weight_decay}")
                sub_folder = path_join(results_folder, f"size_{num_samples}_lr_{lr}_wd_{weight_decay}")
                train_result = train_loop(model_init, device, total_steps, lr, weight_decay, scaling_factor, eval_frequency, train_loader, test_loader, train_eval_loader, sub_folder)
                results.append(train_result)
        #         break
        #     break
        # break
    return results


def calc_acc_rf_train_double_it(data_fracs, num_runs, train_df_doubleit, test_df_doubleit):
    print(f"Double-it data fractions: {str(list(data_fracs))[1:-1]}")

    # Initialize lists to store accuracies
    accuracies = []

    print(f"Original train dataset size {train_df_doubleit.shape[0]}")

    # Loop over each fraction
    for fraction in data_fracs:
        fraction_accuracies = []
        sampled_train_df = train_df_doubleit.sample(frac=fraction, random_state=42)  # You can adjust the random_state for reproducibility

        # Separate the features and target columns for the sampled data
        X_train = sampled_train_df.drop(["class", "super_class"], axis=1).to_numpy()
        y_train = sampled_train_df[["class"]].to_numpy().ravel()
        
        X_test = test_df_doubleit.drop(["class", "super_class"], axis=1).to_numpy()
        y_test = test_df_doubleit[["class"]].astype(int).to_numpy().ravel()

        header = f"-------Fraction {fraction:6.4f}----------"
        print(header)
        print(f"Shape of X_train: {X_train.shape}")
        print(f"Shape of y_train: {y_train.shape}")
        print(f"Shape of X_test: {X_test.shape}")
        print(f"Shape of y_test: {y_test.shape}")
        
        for _ in tqdm(range(num_runs)):
            #Define the Random Forrest classifier (benchmark)
            random_state = np.random.randint(1, 1000)
            clf = RandomForestClassifier(n_estimators=100, random_state=random_state)

            # Train the classifier on the training data
            clf.fit(X_train, y_train)

            # Make predictions on the test data
            y_pred = clf.predict(X_test)

            # Calculate accuracy and store it
            accuracy = accuracy_score(y_test, y_pred)
            fraction_accuracies.append(accuracy)
        print(f"Average acc: {np.average(fraction_accuracies)}")
        print('-'*len(header)+'\n')
        accuracies.append(fraction_accuracies)
    return accuracies

def value_to_list(value_or_list):
    # if value_or_list is None:
    #     return []
    if value_or_list is None:
        return []
    if hasattr(value_or_list, '__iter__'):
        return list(value_or_list)
    return [value_or_list]

def random_sample(dataset, num_samples):
    # Create a random subset using RandomSampler
    subset_indices = torch.randperm(len(dataset))[:num_samples].tolist()
    # Instantiate data loaders
    return Subset(dataset, subset_indices)

def run_on_dataset(model_init, list_num_samples=None, sample_method=random_sample, lr_values=0.01, weight_decay_values=0.001, scaling_factor=1., eval_frequency=100, batch_size=200, total_steps=10**5, train_dataset=None, test_dataset=None,cumulative_train_samples=True, results_folder=None):
    # Initialize lists to store results
    results = []
    
    if list_num_samples is None:
        list_num_samples = [len(train_dataset)]
    else: 
        list_num_samples = value_to_list(list_num_samples)

    if cumulative_train_samples:
        list_num_samples = sorted(list_num_samples, reverse=True)
    
    lr_values = value_to_list(lr_values)
    weight_decay_values = value_to_list(weight_decay_values)
    list_num_samples = value_to_list(list_num_samples)

    # Define the training loop for hyperparameter tuning
    for num_samples in list_num_samples:
        if num_samples is None or num_samples >= len(train_dataset):
            num_samples = len(train_dataset)
            sampled_train_dataset = train_dataset
        else:
            sampled_train_dataset = sample_method(train_dataset, num_samples)
        if cumulative_train_samples:
            train_dataset = sampled_train_dataset
            
        train_loader = torch.utils.data.DataLoader(sampled_train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        train_eval_loader = torch.utils.data.DataLoader(sampled_train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        
        # Iterate through parameters
        for lr in lr_values:
            for weight_decay in weight_decay_values:
                print(f"Num Samples: {num_samples}")
                print(f"LR: {lr}")
                print(f"Weight Decay: {weight_decay}")
                
                # Define results folder path
                sub_folder = path_join(results_folder, f"size_{num_samples}_lr_{lr}_wd_{weight_decay}")
                results_path = path_join(sub_folder, 'results.json')
                
                # Run exp if folder does not contain a results file, load it in case the file exists
                if not os.path.exists(results_path):
                    train_result = train_loop(model_init, device, total_steps, lr, weight_decay, scaling_factor, eval_frequency, train_loader, test_loader, train_eval_loader, sub_folder)
                else:
                    print(f"{'-'*5} Reading results from file {'-'*5}")
                    with open(results_path, 'r') as f:
                        train_result = json.load(f)
                results.append(train_result)
    return results