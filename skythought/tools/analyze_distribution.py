#!/usr/bin/env python3

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from vllm import LLM, SamplingParams
from vllm.config import PoolerConfig
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

###############################################################################
# PSEUDO-CODE FOR vLLM + Qwen-32B
###############################################################################

from tqdm import tqdm
global model

def get_logits_from_qwen_vllm(texts, batch_size=256):
    """
    Function to obtain logits from the Qwen VLLM model for a list of text samples using batching.
    
    Args:
        texts (list): A list of text samples.
        batch_size (int): The number of text samples to process in one batch.
    
    Returns:
        numpy.ndarray: A 2D numpy array of logits [N_samples x D_features].
    """
    # Initialize the LLM model with an overridden pooler config
    all_logits = []

    global model
    # Process texts in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing Batches"):
        batch_texts = texts[i:i + batch_size]  # Create a batch
        batch_outputs = model.encode(batch_texts)  # Encode the entire batch

        # Extract numerical logits from PoolingRequestOutput
        batch_logits = [output.outputs.data[-1].numpy() for output in batch_outputs]  # Adjust if attribute name differs
        all_logits.extend(batch_logits)  # Add batch logits to the result list
    
    return np.array(all_logits)  # Convert to numpy array

def run_pca_and_kde(logits_array, labels, pca_components=2, kde_bandwidth=0.5):
    """
    Perform PCA and KDE on the logits array, separated by labels.
    
    Args:
        logits_array (numpy.ndarray): The logits array [N_samples x D_features].
        labels (list): List of labels corresponding to each sample.
        pca_components (int): Number of principal components for PCA.
        kde_bandwidth (float): Bandwidth parameter for KDE.
    
    Returns:
        dict: A dictionary containing PCA-transformed coordinates and KDE grids for each label.
    """
    # Standardize the data
    scaler = StandardScaler()
    logits_scaled = scaler.fit_transform(logits_array)

    # PCA
    pca = PCA(n_components=pca_components, random_state=42)
    coords_2d = pca.fit_transform(logits_scaled)  # [N x 2]
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance by the first {pca_components} components: {explained_variance}")

    # Prepare data by labels
    label_set = sorted(list(set(labels)))
    label_to_coords = {label: coords_2d[np.array(labels) == label] for label in label_set}

    # Create a grid based on all data
    x_min, x_max = coords_2d[:, 0].min() - 1, coords_2d[:, 0].max() + 1
    y_min, y_max = coords_2d[:, 1].min() - 1, coords_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    grid_points = np.vstack([xx.ravel(), yy.ravel()]).T

    # Perform KDE for each label
    kde_results = {}
    for label in label_set:
        kde = KernelDensity(kernel='gaussian', bandwidth=kde_bandwidth)
        kde.fit(label_to_coords[label])
        log_density = kde.score_samples(grid_points)
        zz = np.exp(log_density).reshape(xx.shape)
        kde_results[label] = zz

    return coords_2d, xx, yy, kde_results, label_set

def main():
    parser = argparse.ArgumentParser(
        description="Run Qwen-32B logits analysis with PCA + KDE for multiple JSON inputs."
    )
    parser.add_argument(
        "--reasoning_json",
        type=str,
        required=True,
        help="Path to the reasoning JSON file containing text samples."
    )
    parser.add_argument(
        "--pretrain_json",
        type=str,
        required=True,
        help="Path to the pretrain JSON file containing text samples."
    )
    parser.add_argument(
        "--instruction_json",
        type=str,
        required=True,
        help="Path to the instruction JSON file containing text samples."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="kde_plot.png",
        help="Path to save the KDE/PCA plot (default: kde_plot.png)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for processing texts."
    )
    parser.add_argument(
        "--kde_bandwidth",
        type=float,
        default=0.5,
        help="Bandwidth for Kernel Density Estimation."
    )
    parser.add_argument(
        "--max_num_samples",
        type=int,
        help="Max number of samples to draw from each JSON file."
    )

    args = parser.parse_args()

    global model 
    model = LLM(model="Qwen/Qwen2.5-32B", task="embed", override_pooler_config=PoolerConfig(pooling_type="ALL"), tensor_parallel_size=4)
    # 1) Load JSON data
    def load_json(path, max_samples=None):
        with open(path, 'r') as f:
            data = json.load(f)
        if max_samples and len(data) > max_samples:
            data = random.sample(data, max_samples)
        texts = [item["text"] for item in data]
        return texts

    reasoning_texts = load_json(args.reasoning_json, args.max_num_samples)
    pretrain_texts = load_json(args.pretrain_json, args.max_num_samples)
    instruction_texts = load_json(args.instruction_json, args.max_num_samples)

    print(f"Loaded {len(reasoning_texts)} reasoning samples.")
    print(f"Loaded {len(pretrain_texts)} pretrain samples.")
    print(f"Loaded {len(instruction_texts)} instruction samples.")

    # Combine all texts and create labels
    all_texts = reasoning_texts + pretrain_texts + instruction_texts
    labels = (
        ["Reasoning"] * len(reasoning_texts) +
        ["Pretrain"] * len(pretrain_texts) +
        ["Instruction"] * len(instruction_texts)
    )

    # 2) Get final-layer logits using Qwen-32B via vLLM
    logits_array = get_logits_from_qwen_vllm(all_texts, batch_size=args.batch_size)

    # Debugging: Check the logits array
    print("Logits shape:", logits_array.shape)
    print("Sample logits (first 5):\n", logits_array[:5])
    # Debugging: Check for NaNs or Infs
    if np.isnan(logits_array).any():
        print("[ERROR] Logits array contains NaNs.")
    if np.isinf(logits_array).any():
        print("[ERROR] Logits array contains infinite values.")

    # 3) PCA -> 2D & 4) KDE
    coords_2d, xx, yy, kde_results, label_set = run_pca_and_kde(
        logits_array, labels, kde_bandwidth=args.kde_bandwidth
    )

    # 5) Plot KDE with PCA for each JSON file separately
    plt.figure(figsize=(10, 8))

    # Define color palette for differentiation
    palette = sns.color_palette("hsv", 3)

    # Process each JSON file logits and labels separately
    json_labels = ["Instruction", "Pretrain", "Reasoning"]
    logits_by_json = [instruction_texts, pretrain_texts, reasoning_texts]

    for idx, (texts, label) in enumerate(zip(logits_by_json, json_labels)):
        # Step 2: Get logits for the current set of texts
        logits_array = get_logits_from_qwen_vllm(texts, batch_size=args.batch_size)
        
        # Step 3 & 4: Run PCA and KDE for the current logits
        coords_2d, xx, yy, zz = run_pca_and_kde(logits_array, kde_bandwidth=args.kde_bandwidth)
        
        # Plot KDE contours for the current category
        plt.contourf(xx, yy, zz, levels=20, alpha=0.4, colors=[palette[idx]], label=f"{label} KDE")
        
        # Overlay scatter points for the current category
        plt.scatter(coords_2d[:, 0], coords_2d[:, 1], s=20, alpha=0.6, color=palette[idx], label=f"{label} Samples")

    # Add plot details
    plt.title("PCA + KDE of Qwen-32B Logits for Each JSON File")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plt.savefig(args.output, dpi=200)
    print(f"[INFO] Plot saved to {args.output}")

    # Optionally, display the plot
    plt.show()


if __name__ == "__main__":
    main()
