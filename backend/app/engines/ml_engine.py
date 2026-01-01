import random
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import numpy as np

# Simple GCN Model for Node Classification (Risk Prediction)
# class GCN(nn.Module):
#     def __init__(self, in_features, hidden_features, out_features):
#         super(GCN, self).__init__()
#         self.linear1 = nn.Linear(in_features, hidden_features)
#         self.linear2 = nn.Linear(hidden_features, out_features)
#
#     def forward(self, x, adj):
#         # Layer 1: H1 = ReLU(A * X * W1)
#         # Using simple matrix multiplication for demonstration
#         # In production, sparse matrix operations (torch.sparse) are needed
#         x = torch.matmul(adj, x)
import os
import numpy as np

class MLEngine:
    """MLEngine with optional scikit-learn fallback.

    Behavior:
    - Try to load a persisted sklearn regressor/classifier from `ml_model.joblib`.
    - If missing, train a small default regressor on synthetic labels derived from
      a deterministic heuristic, persist it, and use for inference.
    - Keeps previous deterministic fallback when model unavailable.
    """

    def __init__(self):
        self.model = None
        self.is_initialized = False
        self.model_path = os.path.join(os.path.dirname(__file__), "ml_model.joblib")

    def initialize_model(self):
        # Try to load sklearn model
        try:
            from joblib import load
            if os.path.exists(self.model_path):
                self.model = load(self.model_path)
                self.is_initialized = True
                print(f"MLEngine: loaded model from {self.model_path}")
                return
        except Exception:
            pass

        # Not loaded; will train on demand in predict_risk
        self.is_initialized = True

    def _train_default_model(self, X, y):
        try:
            from sklearn.ensemble import RandomForestRegressor
            from joblib import dump
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X, y)
            dump(rf, self.model_path)
            self.model = rf
            print(f"MLEngine: trained and saved default model to {self.model_path}")
            return True
        except Exception as e:
            print(f"MLEngine: failed to train sklearn model: {e}")
            return False

    def predict_risk(self, features, adjacency_matrix):
        """
        Predict risk probability for nodes.
        - `features`: list or array shape (N, F)
        - `adjacency_matrix`: array-like shape (N, N)
        Returns list of risk scores in [0,1].
        """
        if not self.is_initialized:
            self.initialize_model()

        features = np.array(features)
        adjacency_matrix = np.array(adjacency_matrix)

        N = features.shape[0]
        # Derive simple graph features
        degrees = np.sum(adjacency_matrix, axis=1)
        max_deg = np.max(degrees) + 1e-5
        norm_deg = degrees / max_deg

        # Build input matrix for sklearn: original features + norm_deg column
        X = np.hstack([features, norm_deg.reshape(-1, 1)])

        # If sklearn model available, use it
        if self.model is not None:
            try:
                # Model may predict continuous risk; clip to [0,1]
                preds = self.model.predict(X)
                preds = np.clip(preds, 0.0, 1.0)
                return preds.tolist()
            except Exception as e:
                print(f"MLEngine: model prediction failed: {e}")

        # If no model, train a default one using deterministic heuristic as labels
        try:
            # Heuristic label generation: reuse previous deterministic function
            avg_features = np.mean(features, axis=1)
            heuristic_risk = 1 - (avg_features * norm_deg)
            # Prepare training data by adding slight noise variations
            X_train = np.vstack([X, X + 0.01, X - 0.01])
            y_train = np.concatenate([heuristic_risk, np.clip(heuristic_risk + 0.02, 0, 1), np.clip(heuristic_risk - 0.02, 0, 1)])

            trained = self._train_default_model(X_train, y_train)
            if trained and self.model is not None:
                preds = self.model.predict(X)
                return np.clip(preds, 0.0, 1.0).tolist()
        except Exception as e:
            print(f"MLEngine: default training/prediction failed: {e}")

        # Final deterministic fallback
        avg_features = np.mean(features, axis=1)
        risk = 1 - (avg_features * norm_deg)
        return risk.clip(0, 1).tolist()
