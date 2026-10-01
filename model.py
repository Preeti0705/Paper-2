"""
Contrastive Siamese Defect Learning (CSDL) & Siamese Neural Network (SNN) Model
Reference: Sections III-E, III-F, Equations (5), (9), Table I, Table II

Components:
1. SiameseTwinNetwork: Twin subnetwork with shared weights producing discriminative embeddings.
2. ContrastiveLoss: Margin-based contrastive loss minimizing distance for same-class pairs and maximizing for distinct-class pairs.
3. DistanceClassifier: Maps learned embedding distances relative to class prototypes to binary defect predictions (Eq. 9).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SiameseTwinNetwork(nn.Module):
    def __init__(self, in_features, embedding_dim=64, activation="relu", dropout_rate=0.15):
        super(SiameseTwinNetwork, self).__init__()
        
        hidden_dim = max(128, embedding_dim * 2)

        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)
        self.bn2 = nn.BatchNorm1d(embedding_dim)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        if str(activation).lower() == "tanh":
            self.act = nn.Tanh()
        else:
            self.act = nn.ReLU()

    def forward_one(self, x):
        h = self.fc1(x)
        if h.size(0) > 1:
            h = self.bn1(h)
        h = self.act(h)
        h = self.dropout(h)
        
        h = self.fc2(h)
        if h.size(0) > 1:
            h = self.bn2(h)
        h = F.normalize(h, p=2, dim=1)
        return h

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        return out1, out2


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        euclidean_distance = F.pairwise_distance(out1, out2, p=2)
        loss_similar = (1.0 - label) * 0.5 * torch.pow(euclidean_distance, 2)
        loss_dissimilar = label * 0.5 * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        return torch.mean(loss_similar + loss_dissimilar)


class CSDLClassifier:
    def __init__(self, feature_mask=None, embedding_dim=64, lr=0.001, margin=1.0, activation="relu", optimizer_type="adam", epochs=25, batch_size=32, device=None):
        self.feature_mask = feature_mask
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.margin = margin
        self.activation = activation
        self.optimizer_type = optimizer_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = None
        self.clean_prototype = None
        self.defect_prototype = None
        self.threshold = 0.5

    def _apply_mask(self, X):
        if self.feature_mask is not None:
            mask = np.array(self.feature_mask, dtype=bool)
            if mask.ndim == 1 and len(mask) == X.shape[1]:
                return X[:, mask]
        return X

    def fit(self, X_train, y_train, pair_generator_fn, val_data=None):
        X_masked = self._apply_mask(X_train)
        in_features = X_masked.shape[1]
        
        self.model = SiameseTwinNetwork(
            in_features=in_features,
            embedding_dim=self.embedding_dim,
            activation=self.activation
        ).to(self.device)
        
        criterion = ContrastiveLoss(margin=self.margin)
        
        if str(self.optimizer_type).lower() == "rmsprop":
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
            
        x1_arr, x2_arr, labels_arr = pair_generator_fn(X_masked, y_train)
        n_pairs = len(labels_arr)
        
        # Pre-convert to tensors
        t_x1 = torch.tensor(x1_arr, dtype=torch.float32).to(self.device)
        t_x2 = torch.tensor(x2_arr, dtype=torch.float32).to(self.device)
        t_y = torch.tensor(labels_arr, dtype=torch.float32).to(self.device)
        
        self.model.train()
        for epoch in range(self.epochs):
            indices = torch.randperm(n_pairs)
            for start_idx in range(0, n_pairs, self.batch_size):
                batch_idx = indices[start_idx : start_idx + self.batch_size]
                b_x1 = t_x1[batch_idx]
                b_x2 = t_x2[batch_idx]
                b_y = t_y[batch_idx]
                
                optimizer.zero_grad()
                out1, out2 = self.model(b_x1, b_x2)
                loss = criterion(out1, out2, b_y)
                loss.backward()
                optimizer.step()

        # Compute class prototype centroids in embedding space
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_masked, dtype=torch.float32).to(self.device)
            embeddings = self.model.forward_one(X_tensor).cpu().numpy()
            
            clean_embeds = embeddings[y_train == 0]
            defect_embeds = embeddings[y_train == 1]
            
            self.clean_prototype = np.mean(clean_embeds, axis=0, keepdims=True)
            self.defect_prototype = np.mean(defect_embeds, axis=0, keepdims=True)
            
            self.clean_prototype /= np.linalg.norm(self.clean_prototype)
            self.defect_prototype /= np.linalg.norm(self.defect_prototype)

        return self

    def get_embeddings(self, X):
        X_masked = self._apply_mask(X)
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_masked, dtype=torch.float32).to(self.device)
            embeds = self.model.forward_one(X_tensor).cpu().numpy()
        return embeds

    def predict_proba(self, X):
        embeds = self.get_embeddings(X)
        d_clean = np.linalg.norm(embeds - self.clean_prototype, axis=1)
        d_defect = np.linalg.norm(embeds - self.defect_prototype, axis=1)
        
        temperature = 1.0
        scores = np.stack([-d_clean / temperature, -d_defect / temperature], axis=1)
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:, 1] >= self.threshold).astype(int)