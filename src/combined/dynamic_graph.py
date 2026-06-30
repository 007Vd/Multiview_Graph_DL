import torch
class DynamicGraphBuilder:
    def __init__(self, momentum_window, top_k=3):
        """
        Initializes the dynamic graph builder.
        
        Args:
            momentum_window (int): Number of previous days to compute momentum.
            top_k (int): Number of top similarities to keep per node. Default is 3.
        """
        self.momentum_window = momentum_window
        self.top_k = top_k
    def build_graph(self, returns_window):
        """
        Builds a dynamic adjacency matrix based on stock momentum.
        
        Args:
            returns_window (torch.Tensor or np.ndarray): Historical returns of shape (window_size, N)
            
        Returns:
            torch.Tensor: Dynamic adjacency matrix of shape (N, N)
        """
        # Convert to tensor if passed as numpy array
        if not isinstance(returns_window, torch.Tensor):
            returns_window = torch.tensor(returns_window, dtype=torch.float32)
            
        window_size, N = returns_window.shape
        if window_size < self.momentum_window:
            raise ValueError(f"window_size ({window_size}) must be >= momentum_window ({self.momentum_window})")
            
        # Step 1: Compute momentum over the momentum_window
        # m_i(t) = Product(1 + r) - 1
        recent_returns = returns_window[-self.momentum_window:]
        momentum = torch.prod(1 + recent_returns, dim=0) - 1.0  # Shape: (N,)
        
        # Step 3: Normalize momentum
        # m_norm = (m - mean) / (std + 1e-8) computed over stocks for that day
        m_mean = momentum.mean()
        m_std = momentum.std(unbiased=False) 
        m_norm = (momentum - m_mean) / (m_std + 1e-8)
        
        # Step 4: Similarity matrix
        # A_ij = exp( -(m_i - m_j)^2 / (2 * sigma^2) )
        # where sigma is std(momentum). 
        # (m_i - m_j)^2 / sigma^2 is equivalent to (m_norm_i - m_norm_j)^2.
        dist_sq = (m_norm.unsqueeze(1) - m_norm.unsqueeze(0)) ** 2  # Shape: (N, N)
        A = torch.exp(-dist_sq / 2.0)
        
        # Step 5: Top-K sparsification
        # Keep only the Top-K largest similarities for each node
        k = min(self.top_k, N)
        topk_vals, topk_indices = torch.topk(A, k=k, dim=1)
        
        A_dynamic = torch.zeros_like(A)
        # Scatter the top-k values back into a sparse dense-tensor
        A_dynamic.scatter_(1, topk_indices, topk_vals)
        
        # Step 6: Self loops
        # Always add self loops. Diagonal entries should be 1.
        A_dynamic.fill_diagonal_(1.0)
        
        # Step 7: Output
        return A_dynamic