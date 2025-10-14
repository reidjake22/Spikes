"""
Efficient and Modular Neural Data Preprocessing

This module efficiently loads neural response data from parquet files and prepares them
in wide format suitable for machine learning models.

Main Features:
- Modular functions with clear responsibility separation.
- Efficient parquet data loading and global indexing.
- Transformation from long to wide format (latency and count features).
- Robust handling of missing data.
"""

import os
import glob
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch


# Global configuration
RESULTS_DIR = "/home/jake/Document/Spikes/projects/mnist_class/mnist_class_wip/results/trials/"


def get_parquet_files(weight_version, noise_version, split):
    """Construct and return a sorted list of parquet file paths."""
    pattern = os.path.join(RESULTS_DIR, weight_version, noise_version, split,
                           "neuron_input_batch_*", "metrics.parquet")
    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} parquet files.")
    return files


def load_and_combine_parquet(files):
    """Load parquet files efficiently, assigning globally unique stimulus IDs."""
    print("Loading parquet files...")
    dfs, global_offset = [], 0

    for idx, file in enumerate(files):
        if idx % 10 == 0:
            print(f"Loading file {idx}/{len(files)}")

        df = pd.read_parquet(file)
        df['stimulus_id'] += global_offset

        global_offset = df['stimulus_id'].max() + 1
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    print(f"Loaded total rows: {len(combined_df)}")
    print(f"Stimulus IDs: {combined_df['stimulus_id'].min()} to {combined_df['stimulus_id'].max()}")
    return combined_df


def prepare_features(df, max_stimuli=None):
    """Convert long-format data to wide-format features."""
    print("Preparing neural features...")

    if max_stimuli:
        selected_stimuli = sorted(df['stimulus_id'].unique())[:max_stimuli]
        df = df[df['stimulus_id'].isin(selected_stimuli)]
        print(f"Filtered to {max_stimuli} stimuli. Data shape: {df.shape}")

    latency = df.pivot(index='stimulus_id', columns='neuron_id', values='latency')
    count = df.pivot(index='stimulus_id', columns='neuron_id', values='count')

    latency.columns = [f'latency_n{col}' for col in latency.columns]
    count.columns = [f'count_n{col}' for col in count.columns]

    features = pd.concat([latency, count], axis=1)
    labels = df.groupby('stimulus_id')['label'].first()

    X, y = features.to_numpy(), labels.to_numpy()

    if np.isnan(X).any():
        print("NaNs found, filling with zeros.")
        X = np.nan_to_num(X, nan=0.0)

    print(f"Features shape: {X.shape}, Labels shape: {y.shape}")
    return X, y


def scale_features(X):
    """Standardize features using StandardScaler."""
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"Feature mean: {X_scaled.mean():.4f}, std: {X_scaled.std():.4f}")
    return X_scaled


def convert_to_tensors(X, y):
    """Convert numpy arrays to PyTorch tensors."""
    print("Converting features and labels to tensors...")
    return torch.FloatTensor(X), torch.LongTensor(y)


# Example usage:
if __name__ == "__main__":
    weight_version = "v1"
    noise_version = "noise_0"
    split = "train"

    files = get_parquet_files(weight_version, noise_version, split)
    df = load_and_combine_parquet(files)

    X, y = prepare_features(df, max_stimuli=400)
    X_scaled = scale_features(X)
    X_tensor, y_tensor = convert_to_tensors(X_scaled, y)

    unique, counts = np.unique(y, return_counts=True)
    print("Class distribution:")
    for cls, cnt in zip(unique, counts):
        print(f"Class {cls}: {cnt} samples ({cnt / len(y) * 100:.1f}%)")