import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import umap
import matplotlib.pyplot as plt
from trainUtils import MLP

def load_dataset(path):
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c.startswith('d')]
    x = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(df['class'].values, dtype=torch.long)
    return TensorDataset(x, y)

def load_model(checkpoint, device):
    model = MLP(117, 256, 4, 3).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model

def extract_features(model, loader, device):
    feats = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            h = x.view(x.size(0), -1)
            for layer in model.layers:
                h = model.relu(layer(h))
            feats.append(h.cpu())
            labels.append(y)
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = load_dataset(args.data)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    model_before = load_model(args.early_checkpoint, device)
    model_after = load_model(args.late_checkpoint, device)
    feats_b, labels = extract_features(model_before, loader, device)
    feats_a, _ = extract_features(model_after, loader, device)
    reducer = umap.UMAP(n_components=2, random_state=0)
    emb_b = reducer.fit_transform(feats_b)
    emb_a = reducer.fit_transform(feats_a)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    scatter0 = axes[0].scatter(emb_b[:,0], emb_b[:,1], c=labels, cmap='tab10', s=5)
    axes[0].set_title('Before Grokking')
    axes[1].scatter(emb_a[:,0], emb_a[:,1], c=labels, cmap='tab10', s=5)
    axes[1].set_title('After Grokking')
    axes[0].legend(*scatter0.legend_elements(), title='Subclass')
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Saved figure to {args.output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize latent space before and after grokking.')
    parser.add_argument('--early-checkpoint', required=True, help='Path to early checkpoint file')
    parser.add_argument('--late-checkpoint', required=True, help='Path to late checkpoint file')
    parser.add_argument('--data', default='datasets/V1eq_binaryTreeMultiDim_distSuperClasses_ns_10000_ndg_9_ndpg_13_test.csv', help='CSV file with test data')
    parser.add_argument('--output', default='latent_space_evolution.png', help='Output image file')
    args = parser.parse_args()
    main(args)
