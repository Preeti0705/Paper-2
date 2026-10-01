"""
Enzyme Action Optimizer (EAO) for Joint Feature Selection & Hyperparameter Optimization
Reference: Algorithm 1, Section III-C, Equations (4), (5), Table I

Key Components:
1. Candidate Representation: (Binary Feature Mask s, Hyperparameter Dict phi)
2. Objective Fitness: F = alpha * (1 - Accuracy) + beta * (|S| / |T|) (Eq. 4)
3. Enzyme-inspired Operators:
   - Binding (Crossover)
   - Cleavage (Random Local Mutation)
   - Catalysis (Movement towards Elite E using Eq. 5)
   - Repair (Ensuring bounds and validity)
   - Elitist Replacement
"""

import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


class FastSiameseProxy(nn.Module):
    def __init__(self, in_features, embedding_dim=64, activation="relu"):
        super().__init__()
        h_dim = max(64, embedding_dim)
        self.fc1 = nn.Linear(in_features, h_dim)
        self.fc2 = nn.Linear(h_dim, embedding_dim)
        self.act = nn.Tanh() if str(activation).lower() == "tanh" else nn.ReLU()

    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.fc2(h)
        return F.normalize(h, p=2, dim=1)


class EAOCandidate:
    def __init__(self, n_features, feature_mask=None, params=None):
        self.n_features = n_features
        if feature_mask is None:
            self.mask = np.random.choice([0, 1], size=n_features, p=[0.35, 0.65])
            if np.sum(self.mask) < 4:
                self.mask[:4] = 1
        else:
            self.mask = np.array(feature_mask, dtype=int)

        if params is None:
            self.params = {
                "lr": float(np.random.uniform(0.0005, 0.005)),
                "batch_size": int(np.random.choice([16, 32, 64])),
                "epochs": int(np.random.randint(15, 30)),
                "embedding_dim": int(np.random.choice([32, 64, 128])),
                "margin": float(np.random.uniform(0.8, 1.5)),
                "optimizer": str(np.random.choice(["adam", "rmsprop"])),
                "activation": str(np.random.choice(["relu", "tanh"])),
            }
        else:
            self.params = copy.deepcopy(params)

        self.fitness = float("inf")
        self.accuracy = 0.0

    def copy(self):
        new_cand = EAOCandidate(self.n_features, self.mask.copy(), self.params)
        new_cand.fitness = self.fitness
        new_cand.accuracy = self.accuracy
        return new_cand


class EnzymeActionOptimizer:
    def __init__(
        self,
        population_size=15,
        generations=15,
        k_folds=3,
        alpha=0.9,
        beta=0.1,
        catalytic_rate=0.4,
        stochastic_coeff=0.1,
        mutation_rate=0.15,
        random_state=42,
    ):
        self.population_size = population_size
        self.generations = generations
        self.k_folds = k_folds
        self.alpha = alpha
        self.beta = beta
        self.eta = catalytic_rate
        self.gamma = stochastic_coeff
        self.mutation_rate = mutation_rate
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        
        self.history = {
            "best_fitness_per_gen": [],
            "mean_fitness_per_gen": [],
            "best_accuracy_per_gen": [],
            "num_features_per_gen": [],
        }

    def _evaluate_candidate(self, candidate, folds_tensors):
        mask_idx = np.where(candidate.mask == 1)[0]
        n_selected = len(mask_idx)
        n_total = candidate.n_features

        if n_selected < 3:
            candidate.fitness = 10.0
            candidate.accuracy = 0.0
            return candidate.fitness

        mask_torch = torch.tensor(mask_idx, dtype=torch.long)
        acc_list = []

        for t_X_tr, t_y_tr, t_X_v, t_y_v, t_px1, t_px2, t_py in folds_tensors:
            x1_sub = t_px1[:, mask_torch]
            x2_sub = t_px2[:, mask_torch]
            X_tr_sub = t_X_tr[:, mask_torch]
            X_v_sub = t_X_v[:, mask_torch]

            dim = candidate.params["embedding_dim"]
            model = FastSiameseProxy(in_features=n_selected, embedding_dim=dim, activation=candidate.params["activation"])
            opt = torch.optim.Adam(model.parameters(), lr=candidate.params["lr"], weight_decay=1e-4)
            margin = candidate.params["margin"]
            
            # Fast full-batch / mini-batch tensor pass
            model.train()
            for _ in range(12):
                opt.zero_grad()
                e1 = model(x1_sub)
                e2 = model(x2_sub)
                dist = torch.pairwise_distance(e1, e2, p=2)
                loss = torch.mean((1.0 - t_py) * 0.5 * dist.pow(2) + t_py * 0.5 * torch.clamp(margin - dist, min=0.0).pow(2))
                loss.backward()
                opt.step()

            # Prototype centroid evaluation
            model.eval()
            with torch.no_grad():
                tr_emb = model(X_tr_sub)
                v_emb = model(X_v_sub)
                
                c0 = tr_emb[t_y_tr == 0].mean(dim=0, keepdim=True)
                c1 = tr_emb[t_y_tr == 1].mean(dim=0, keepdim=True)
                
                d0 = torch.norm(v_emb - c0, p=2, dim=1)
                d1 = torch.norm(v_emb - c1, p=2, dim=1)
                preds = (d1 < d0).long()
                
                correct = (preds == t_y_v).sum().item()
                acc = correct / len(t_y_v)
                acc_list.append(acc)

        mean_acc = float(np.mean(acc_list))
        fitness = self.alpha * (1.0 - mean_acc) + self.beta * (n_selected / n_total)
        candidate.accuracy = mean_acc
        candidate.fitness = fitness
        return fitness

    def _tournament_selection(self, population, k=3):
        candidates = self.rng.choice(population, size=k, replace=False)
        best = min(candidates, key=lambda c: c.fitness)
        return best

    def _binding_crossover(self, parent_a, parent_b):
        child = parent_a.copy()
        swap_mask = self.rng.random(parent_a.n_features) < 0.5
        child.mask[swap_mask] = parent_b.mask[swap_mask]

        for k in child.params:
            if self.rng.random() < 0.5:
                child.params[k] = parent_b.params[k]
        return child

    def _cleavage_mutation(self, candidate):
        flips = self.rng.random(candidate.n_features) < self.mutation_rate
        candidate.mask[flips] = 1 - candidate.mask[flips]

        if self.rng.random() < self.mutation_rate:
            candidate.params["lr"] = float(np.clip(candidate.params["lr"] * self.rng.uniform(0.8, 1.2), 0.0001, 0.01))
        if self.rng.random() < self.mutation_rate:
            candidate.params["margin"] = float(np.clip(candidate.params["margin"] + self.rng.uniform(-0.15, 0.15), 0.5, 2.0))
        if self.rng.random() < self.mutation_rate:
            candidate.params["embedding_dim"] = int(self.rng.choice([32, 64, 128]))
        if self.rng.random() < self.mutation_rate:
            candidate.params["optimizer"] = str(self.rng.choice(["adam", "rmsprop"]))
        return candidate

    def _catalysis(self, candidate, elite):
        for i in range(candidate.n_features):
            diff = elite.mask[i] - candidate.mask[i]
            r = self.rng.normal(0, 1)
            prob_move = self.eta * diff + self.gamma * r
            if abs(prob_move) > 0.35:
                candidate.mask[i] = elite.mask[i]

        candidate.params["lr"] += self.eta * (elite.params["lr"] - candidate.params["lr"]) + self.gamma * float(self.rng.normal(0, 0.0003))
        candidate.params["margin"] += self.eta * (elite.params["margin"] - candidate.params["margin"]) + self.gamma * float(self.rng.normal(0, 0.03))
        return candidate

    def _repair(self, candidate):
        if np.sum(candidate.mask) < 3:
            disabled_idx = np.where(candidate.mask == 0)[0]
            if len(disabled_idx) > 0:
                enable_count = min(3 - int(np.sum(candidate.mask)), len(disabled_idx))
                pick = self.rng.choice(disabled_idx, size=enable_count, replace=False)
                candidate.mask[pick] = 1

        candidate.params["lr"] = float(np.clip(candidate.params["lr"], 0.0001, 0.01))
        candidate.params["margin"] = float(np.clip(candidate.params["margin"], 0.5, 2.0))
        candidate.params["batch_size"] = int(candidate.params["batch_size"]) if candidate.params["batch_size"] in [16, 32, 64] else 32
        candidate.params["embedding_dim"] = int(candidate.params["embedding_dim"]) if candidate.params["embedding_dim"] in [16, 32, 64, 128] else 64
        candidate.params["epochs"] = int(np.clip(candidate.params["epochs"], 10, 50))
        return candidate

    def optimize(self, X, y, verbose=True):
        n_features = X.shape[1]
        
        # Pre-convert folds into GPU / CPU PyTorch tensors
        from pair_generator import construct_contrastive_pairs
        skf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=self.random_state)
        folds_tensors = []
        for tr_i, v_i in skf.split(X, y):
            X_tr, y_tr = X[tr_i], y[tr_i]
            X_v, y_v = X[v_i], y[v_i]
            px1, px2, py = construct_contrastive_pairs(X_tr, y_tr, k_pos_neighbors=3, random_state=self.random_state)
            
            folds_tensors.append((
                torch.tensor(X_tr, dtype=torch.float32),
                torch.tensor(y_tr, dtype=torch.long),
                torch.tensor(X_v, dtype=torch.float32),
                torch.tensor(y_v, dtype=torch.long),
                torch.tensor(px1, dtype=torch.float32),
                torch.tensor(px2, dtype=torch.float32),
                torch.tensor(py, dtype=torch.float32)
            ))
            
        population = [EAOCandidate(n_features) for _ in range(self.population_size)]
        
        for cand in population:
            self._evaluate_candidate(cand, folds_tensors)

        elite = min(population, key=lambda c: c.fitness).copy()
        
        if verbose:
            print(f"[EAO Init] Best Fitness: {elite.fitness:.4f}, Accuracy: {elite.accuracy*100:.2f}%, Features: {int(np.sum(elite.mask))}/{n_features}", flush=True)

        for gen in range(1, self.generations + 1):
            new_population = [elite.copy()]
            
            while len(new_population) < self.population_size:
                parent_a = self._tournament_selection(population)
                parent_b = self._tournament_selection(population)
                
                child = self._binding_crossover(parent_a, parent_b)
                child = self._cleavage_mutation(child)
                child = self._catalysis(child, elite)
                child = self._repair(child)
                self._evaluate_candidate(child, folds_tensors)
                new_population.append(child)

            population = new_population
            current_best = min(population, key=lambda c: c.fitness)
            if current_best.fitness < elite.fitness:
                elite = current_best.copy()

            mean_fit = float(np.mean([c.fitness for c in population]))
            self.history["best_fitness_per_gen"].append(elite.fitness)
            self.history["mean_fitness_per_gen"].append(mean_fit)
            self.history["best_accuracy_per_gen"].append(elite.accuracy)
            self.history["num_features_per_gen"].append(int(np.sum(elite.mask)))

            if verbose and (gen % 3 == 0 or gen == self.generations):
                print(f"[EAO Gen {gen:02d}/{self.generations}] Best Fitness: {elite.fitness:.4f}, Acc: {elite.accuracy*100:.2f}%, Active Features: {int(np.sum(elite.mask))}/{n_features}, LR: {elite.params['lr']:.5f}, Dim: {elite.params['embedding_dim']}", flush=True)

        return elite, self.history
