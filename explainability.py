"""
Explainability Layer Using SHAP and Integrated Gradients (IG)
Reference: Section III-G, Equations (10), (11), Figures 2, 4

Features:
1. SHAP Global Feature Attribution (Eq. 10):
   Computes marginal contribution phi_i of each software metric to the defect prediction score.
2. Integrated Gradients Instance-Level Reasoning (Eq. 11):
   Computes path integral of gradients along straight-line path from baseline x' (e.g. clean module) to x.
3. Actionable Developer Explanation Generator:
   Summarizes structural, complexity, and coupling risks for refactoring and testing prioritization.
"""

import numpy as np
import torch
import shap
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import io
import base64


class PyTorchModelWrapper(torch.nn.Module):
    """
    Differentiable wrapper around Siamese subnetwork and defect prototype distance
    for Integrated Gradients computation.
    """
    def __init__(self, twin_model, clean_proto, defect_proto):
        super(PyTorchModelWrapper, self).__init__()
        self.twin = twin_model
        self.clean_proto = torch.tensor(clean_proto, dtype=torch.float32)
        self.defect_proto = torch.tensor(defect_proto, dtype=torch.float32)

    def forward(self, x):
        # Forward through subnetwork
        h = self.twin.forward_one(x)
        clean_p = self.clean_proto.to(x.device)
        defect_p = self.defect_proto.to(x.device)
        
        # Negative Euclidean distance score towards defect prototype
        d_clean = torch.norm(h - clean_p, p=2, dim=1, keepdim=True)
        d_defect = torch.norm(h - defect_p, p=2, dim=1, keepdim=True)
        
        # Relative defect proneness score (higher = more defective)
        score = d_clean - d_defect
        return score


class ExplainabilityEngine:
    def __init__(self, csdl_classifier, feature_names):
        self.clf = csdl_classifier
        self.full_feature_names = np.array(feature_names)
        
        if self.clf.feature_mask is not None:
            mask = np.array(self.clf.feature_mask, dtype=bool)
            self.active_feature_names = self.full_feature_names[mask]
        else:
            self.active_feature_names = self.full_feature_names
            
        self.device = self.clf.device
        self.torch_wrapper = PyTorchModelWrapper(
            self.clf.model,
            self.clf.clean_prototype,
            self.clf.defect_prototype
        ).to(self.device)
        self.ig = IntegratedGradients(self.torch_wrapper)

    def compute_shap_global(self, X_background, X_explain=None, n_samples=50):
        """
        Computes SHAP global feature importances across the dataset (Eq. 10).
        """
        X_bg_masked = self.clf._apply_mask(X_background)
        if len(X_bg_masked) > n_samples:
            idx = np.random.choice(len(X_bg_masked), n_samples, replace=False)
            X_bg_sub = X_bg_masked[idx]
        else:
            X_bg_sub = X_bg_masked

        if X_explain is not None:
            X_exp_masked = self.clf._apply_mask(X_explain)
            if len(X_exp_masked) > n_samples:
                idx2 = np.random.choice(len(X_exp_masked), n_samples, replace=False)
                X_exp_sub = X_exp_masked[idx2]
            else:
                X_exp_sub = X_exp_masked
        else:
            X_exp_sub = X_bg_sub

        def model_predict_fn(x_in):
            # Returns P(defective)
            embeds = []
            self.clf.model.eval()
            with torch.no_grad():
                t = torch.tensor(x_in, dtype=torch.float32).to(self.device)
                h = self.clf.model.forward_one(t).cpu().numpy()
            d_clean = np.linalg.norm(h - self.clf.clean_prototype, axis=1)
            d_defect = np.linalg.norm(h - self.clf.defect_prototype, axis=1)
            scores = np.stack([-d_clean, -d_defect], axis=1)
            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            return probs[:, 1]

        explainer = shap.KernelExplainer(model_predict_fn, X_bg_sub)
        shap_values = explainer.shap_values(X_exp_sub, nsamples=60)
        
        # Global mean absolute SHAP values per active feature
        mean_shap = np.mean(np.abs(shap_values), axis=0)
        
        feature_importance = sorted(
            [{"feature": name, "shap_importance": float(val)} for name, val in zip(self.active_feature_names, mean_shap)],
            key=lambda item: item["shap_importance"],
            reverse=True,
        )
        return feature_importance, shap_values

    def compute_integrated_gradients(self, x_instance, baseline=None, steps=50):
        """
        Computes instance-level Integrated Gradients attribution for a single module x (Eq. 11).
        """
        x_masked = self.clf._apply_mask(np.array(x_instance).reshape(1, -1))
        x_tensor = torch.tensor(x_masked, dtype=torch.float32, requires_grad=True).to(self.device)
        
        if baseline is None:
            # Baseline: Zero vector (clean origin in min-max scaled space)
            baseline_tensor = torch.zeros_like(x_tensor)
        else:
            b_masked = self.clf._apply_mask(np.array(baseline).reshape(1, -1))
            baseline_tensor = torch.tensor(b_masked, dtype=torch.float32).to(self.device)

        self.torch_wrapper.eval()
        attributions, delta = self.ig.attribute(
            x_tensor,
            baselines=baseline_tensor,
            n_steps=steps,
            return_convergence_delta=True
        )

        attr_vals = attributions.detach().cpu().numpy().flatten()
        
        local_attributions = []
        for name, val, raw_v in zip(self.active_feature_names, attr_vals, x_masked.flatten()):
            local_attributions.append({
                "feature": name,
                "attribution": float(val),
                "value": float(raw_v),
                "direction": "Increases Defect Risk" if val > 0 else "Decreases Defect Risk",
            })

        local_attributions = sorted(local_attributions, key=lambda item: abs(item["attribution"]), reverse=True)
        return local_attributions

    def generate_developer_explanation(self, prediction_result, local_attributions, top_k=5):
        """
        Creates actionable developer recommendations based on top contributing metrics.
        """
        is_defective = prediction_result["predicted_class"] == 1
        prob = prediction_result["defect_probability"]

        lines = []
        if is_defective:
            lines.append(f"[WARNING: DEFECTIVE] Module Flagged as DEFECTIVE (Confidence: {prob*100:.1f}%)")
            lines.append("Top metrics driving defect proneness:")
            for item in local_attributions[:top_k]:
                if item["attribution"] > 0:
                    lines.append(f"- {item['feature']} (Normalized Val: {item['value']:.2f}): Strongly increases defect risk (+{item['attribution']:.4f}).")
            lines.append("\nActionable Recommendations:")
            lines.append("- Prioritize unit and boundary test cases for modules with high cyclomatic and branch complexity.")
            lines.append("- Consider refactoring large functions to reduce coupling and total operand counts.")
        else:
            lines.append(f"[CLEAN] Module Flagged as CLEAN (Reliability: {(1-prob)*100:.1f}%)")
            lines.append("Metrics are within healthy defect-free boundaries.")
            
        return "\n".join(lines)
