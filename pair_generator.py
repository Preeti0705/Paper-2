"""
Pair Construction Module for Contrastive Siamese Defect Learning (CSDL)
Reference: Section III-D, "Pair Construction Strategy for CSDL"

Strategy:
1. Equal 1:1 ratio of positive pairs (same class) and negative pairs (different classes).
2. For each sample, k-nearest neighbors of the same class are selected to form meaningful positive pairs.
3. Negative pairs are sampled randomly across opposite classes to ensure diversity.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors


def construct_contrastive_pairs(X, y, k_pos_neighbors=5, random_state=42):
    """
    Constructs balanced positive and negative pairs from feature matrix X and labels y.
    
    Pairs:
    - Positive pair: (x_i, x_j) where y_i == y_j. Pair label = 0 (similar/same class).
    - Negative pair: (x_i, x_k) where y_i != y_k. Pair label = 1 (dissimilar/different class).
    """
    rng = np.random.default_rng(random_state)
    n_samples = len(X)

    unique_classes = np.unique(y)
    class_indices = {c: np.where(y == c)[0] for c in unique_classes}

    positive_pairs = []
    
    # 1. Meaningful positive pairs using K-Nearest Neighbors within the same class
    for c in unique_classes:
        indices = class_indices[c]
        if len(indices) <= 1:
            continue
        k_val = min(k_pos_neighbors + 1, len(indices))
        nbrs = NearestNeighbors(n_neighbors=k_val, metric="euclidean").fit(X[indices])
        distances, nn_idx = nbrs.kneighbors(X[indices])

        for i_pos, neighbors in enumerate(nn_idx):
            src_idx = indices[i_pos]
            for n_pos in neighbors[1:]:  # skip self
                target_idx = indices[n_pos]
                positive_pairs.append((src_idx, target_idx, 0))  # 0 = same class

    n_positive = len(positive_pairs)

    # 2. Balanced negative pairs (1:1 ratio) sampled randomly across opposite classes
    negative_pairs = []
    for _ in range(n_positive):
        c1, c2 = rng.choice(unique_classes, size=2, replace=False)
        idx1 = rng.choice(class_indices[c1])
        idx2 = rng.choice(class_indices[c2])
        negative_pairs.append((idx1, idx2, 1))  # 1 = different class

    all_pairs = positive_pairs + negative_pairs
    rng.shuffle(all_pairs)

    x1_list = np.array([X[p[0]] for p in all_pairs], dtype=np.float32)
    x2_list = np.array([X[p[1]] for p in all_pairs], dtype=np.float32)
    pair_labels = np.array([p[2] for p in all_pairs], dtype=np.float32)

    return x1_list, x2_list, pair_labels


class SiamesePairDataset(Dataset):
    """
    PyTorch Dataset yielding (x1, x2, pair_label).
    """
    def __init__(self, x1, x2, labels):
        self.x1 = torch.tensor(x1, dtype=torch.float32)
        self.x2 = torch.tensor(x2, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.x1[idx], self.x2[idx], self.labels[idx]
