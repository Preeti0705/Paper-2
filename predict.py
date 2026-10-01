"""
Prediction and Inference Pipeline for CSDL-SEN-XAI
Reference: Section III-F, III-G, Equations (9), (10), (11)

Functionality:
1. Loads trained CSDL-SNN weights, prototypes, and preprocessor.
2. Accepts software metric inputs (dictionary, JSON, or CSV row).
3. Computes embedding distance defect probability (Eq. 9).
4. Generates instance-level Integrated Gradients attributions (Eq. 11).
5. Provides developer-friendly actionable recommendations.
"""

import os
import joblib
import torch
import numpy as np
import pandas as pd

from preprocess import load_preprocessor
from model import SiameseTwinNetwork, CSDLClassifier
from explainability import ExplainabilityEngine


class DefectPredictor:
    def __init__(
        self,
        model_weights_path="models/csdl_snn_weights.pt",
        prototypes_path="models/csdl_prototypes.joblib",
        preprocessor_path="models/preprocessor.joblib",
        device=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        if not os.path.exists(prototypes_path):
            raise FileNotFoundError("Trained prototypes not found. Please run 'python train.py' first.")

        self.scaler, self.feature_names = load_preprocessor(preprocessor_path)
        proto_data = joblib.load(prototypes_path)
        
        self.clean_prototype = proto_data["clean_prototype"]
        self.defect_prototype = proto_data["defect_prototype"]
        self.feature_mask = proto_data["feature_mask"]
        self.best_params = proto_data["best_params"]
        self.active_features = proto_data["active_features"]
        
        # Reconstruct CSDL model
        mask_bool = np.array(self.feature_mask, dtype=bool)
        n_in = int(np.sum(mask_bool))
        
        self.twin_net = SiameseTwinNetwork(
            in_features=n_in,
            embedding_dim=self.best_params.get("embedding_dim", 64),
            activation=self.best_params.get("activation", "relu"),
        ).to(self.device)
        
        self.twin_net.load_state_dict(torch.load(model_weights_path, map_location=self.device))
        self.twin_net.eval()

        self.csdl_clf = CSDLClassifier(
            feature_mask=self.feature_mask,
            embedding_dim=self.best_params.get("embedding_dim", 64),
            activation=self.best_params.get("activation", "relu"),
            device=self.device,
        )
        self.csdl_clf.model = self.twin_net
        self.csdl_clf.clean_prototype = self.clean_prototype
        self.csdl_clf.defect_prototype = self.defect_prototype

        self.xai_engine = ExplainabilityEngine(self.csdl_clf, self.feature_names)

    def predict_instance(self, metrics_dict_or_array, run_xai=True):
        """
        Runs prediction and XAI for a single module instance.
        """
        if isinstance(metrics_dict_or_array, dict):
            # Fill aligned array
            x_raw = []
            for col in self.feature_names:
                val = metrics_dict_or_array.get(col, 0.0)
                x_raw.append(float(val))
            x_arr = np.array(x_raw).reshape(1, -1)
        elif isinstance(metrics_dict_or_array, (list, np.ndarray)):
            x_arr = np.array(metrics_dict_or_array, dtype=float).reshape(1, -1)
        elif isinstance(metrics_dict_or_array, pd.DataFrame):
            x_arr = metrics_dict_or_array[self.feature_names].values.astype(float)
        else:
            raise ValueError("Input must be a dict, list, numpy array, or DataFrame.")

        # Normalize with fitted Min-Max scaler (Eq. 1)
        x_norm = self.scaler.transform(x_arr)
        
        # Predict probability and class (Eq. 9)
        probs = self.csdl_clf.predict_proba(x_norm)[0]
        prob_defective = float(probs[1])
        predicted_class = int(prob_defective >= 0.5)
        label_str = "Defective" if predicted_class == 1 else "Non-Defective / Clean"

        result = {
            "predicted_class": predicted_class,
            "label": label_str,
            "defect_probability": prob_defective,
            "clean_probability": float(probs[0]),
            "raw_metrics": {name: float(val) for name, val in zip(self.feature_names, x_arr.flatten())},
            "normalized_metrics": {name: float(val) for name, val in zip(self.feature_names, x_norm.flatten())},
        }

        if run_xai:
            local_attributions = self.xai_engine.compute_integrated_gradients(x_norm[0])
            recommendation = self.xai_engine.generate_developer_explanation(result, local_attributions)
            result["integrated_gradients"] = local_attributions
            result["developer_recommendation"] = recommendation

        return result


if __name__ == "__main__":
    predictor = DefectPredictor()
    
    sample_defective = {
        "loc": 46.0,
        "v(g)": 15.0,
        "ev(g)": 3.0,
        "iv(g)": 1.0,
        "n": 239.0,
        "v": 1362.41,
        "l": 0.04,
        "d": 22.3,
        "i": 61.1,
        "e": 30377.95,
        "b": 0.45,
        "t": 1687.66,
        "lOCode": 8.0,
        "lOComment": 35.0,
        "lOBlank": 22.0,
        "locCodeAndComment": 0.0,
        "uniq_Op": 15.0,
        "uniq_Opnd": 37.0,
        "total_Op": 129.0,
        "total_Opnd": 110.0,
        "branchCount": 29.0
    }
    
    res = predictor.predict_instance(sample_defective)
    print("=== CSDL-SEN-XAI Inference Result ===")
    print(f"Prediction: {res['label']} (Defect Risk: {res['defect_probability']*100:.2f}%)")
    print("\nDeveloper Explanation:")
    print(res["developer_recommendation"])