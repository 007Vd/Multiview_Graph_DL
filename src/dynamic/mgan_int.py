import torch
import torch.nn as nn
from torch.utils.data import Dataset
from dynamic_graph import DynamicGraphBuilder
# ---------------------------------------------------------
# 1. Dataset Modifications
# ---------------------------------------------------------
class DynamicGraphDataset(Dataset):
    def __init__(self, features, labels, momentum_window=10, top_k=3):
        """
        features: (Samples, Window_Size, N) - The rolling windows of returns
        labels: (Samples, N) - The future returns
        """
        self.features = features
        self.labels = labels
        self.graph_builder = DynamicGraphBuilder(momentum_window=momentum_window, top_k=top_k)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, index):
        # Shape: (Window_Size, N)
        X_window = self.features[index]
        
        # Build dynamic adjacency strictly from the input window
        # No future information is leaked
        dynamic_adj = self.graph_builder.build_graph(X_window)
        
        y_future = self.labels[index]
        
        return X_window, dynamic_adj, y_future
# ---------------------------------------------------------
# 2. DynamicMGAN Modifications
# ---------------------------------------------------------
class DynamicMGAN(nn.Module):
    def __init__(self, in_features, out_features):
        super(DynamicMGAN, self).__init__()
        # Graph convolution layer customized for the dynamic adjacency matrix.
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.activation = nn.LeakyReLU(0.2)
    def forward(self, E_stock, dynamic_adj):
        """
        E_stock: (B, N, H) - Independent stock embeddings from LSTM
        dynamic_adj: (B, N, N) - Dynamic adjacency matrix for each sample
        """
        # Linear transformation
        h = self.W(E_stock)  # (B, N, out_features)
        
        # Graph convolution using the dynamic adjacency matrix
        # dynamic_adj is (B, N, N), h is (B, N, out_features)
        # bmm (batch matrix multiplication) computes dynamic_adj @ h
        E_dynamic = torch.bmm(dynamic_adj, h)  # (B, N, out_features)
        
        return self.activation(E_dynamic)
# ---------------------------------------------------------
# 3. Updated Forward Pass Example
# ---------------------------------------------------------
class FullDynamicPipeline(nn.Module):
    def __init__(self, num_assets, hidden_dim, mgan_out_features):
        super(FullDynamicPipeline, self).__init__()
        
        # Assume StockEncoder outputs E_stock of shape (B, N, hidden_dim)
        # using the LSTM + Historical Attention
        # We define a dummy encoder here for structural demonstration
        self.encoder = nn.Linear(30, hidden_dim) 
        
        # The new dynamic MGAN module
        self.dynamic_mgan = DynamicMGAN(hidden_dim, mgan_out_features)
        
        # Static MGANs would also be evaluated here and concatenated,
        # but omitted for brevity to highlight the dynamic path.
        
        # Portfolio Head
        self.portfolio_head = nn.Sequential(
            nn.Linear(mgan_out_features, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, X_window, dynamic_adj):
        """
        X_window: (B, Window_Size, N)
        dynamic_adj: (B, N, N)
        """
        # LSTM + Historical Attention Pass
        X_perm = X_window.permute(0, 2, 1)
        E_stock = self.encoder(X_perm)  # (B, N, hidden_dim)
        
        # Dynamic MGAN Pass
        E_dynamic = self.dynamic_mgan(E_stock, dynamic_adj)  # (B, N, mgan_out_features)
        
        # Here you would typically concatenate E_dynamic with your static MGAN embeddings
        # E_final = torch.cat([E_static_sector, E_static_industry, E_dynamic], dim=-1)
        
        # Portfolio Head Pass
        logits = self.portfolio_head(E_dynamic).squeeze(-1)  # (B, N)
        weights = torch.softmax(logits, dim=1)  # (B, N)
        
        return weights
