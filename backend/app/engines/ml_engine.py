import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Simple GCN Model for Node Classification (Risk Prediction)
class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(GCN, self).__init__()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, adj):
        # Layer 1: H1 = ReLU(A * X * W1)
        # Using simple matrix multiplication for demonstration
        # In production, sparse matrix operations (torch.sparse) are needed
        x = torch.matmul(adj, x)
        x = self.linear1(x)
        x = F.relu(x)
        
        # Layer 2: H2 = Softmax(A * H1 * W2)
        x = torch.matmul(adj, x)
        x = self.linear2(x)
        return F.softmax(x, dim=1)

class MLEngine:
    def __init__(self):
        self.model = None
        self.is_initialized = False

    def initialize_model(self):
        # 6 features (Stars, Forks, Issues, Activity, OpenRank, Constraint)
        # 2 output classes (Safe, Risky)
        self.model = GCN(in_features=6, hidden_features=16, out_features=2)
        self.model.eval() # Inference mode
        self.is_initialized = True

    def predict_risk(self, features, adjacency_matrix):
        """
        Predict risk probability for nodes using GCN.
        features: Tensor of shape (N, 6)
        adjacency_matrix: Tensor of shape (N, N)
        """
        if not self.is_initialized:
            self.initialize_model()

        with torch.no_grad():
            # Add self-loops to adjacency matrix for GCN
            I = torch.eye(adjacency_matrix.shape[0])
            adj_hat = adjacency_matrix + I
            
            # Normalize adjacency matrix (D^-0.5 * A * D^-0.5)
            # Simplified normalization for demo: Row-normalize
            row_sum = adj_hat.sum(1)
            d_inv = torch.pow(row_sum, -1).flatten()
            d_inv[torch.isinf(d_inv)] = 0.
            d_mat = torch.diag(d_inv)
            norm_adj = torch.matmul(d_mat, adj_hat)

            output = self.model(features, norm_adj)
            
            # Return probability of class 1 (Risky)
            return output[:, 1].tolist()
