import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv

class DocumentGraphEncoder(nn.Module):
    """
    Message-Passing Graph Neural Network that contextually models the document layout.
    Uses TransformerConv to allow attention over neighbors conditional on edge types 
    (e.g. 'above', 'left_of', etc).
    """
    def __init__(self, d_model: int = 256, num_layers: int = 4, num_edge_types: int = 10):
        super().__init__()
        self.d_model = d_model
        
        # Embed categorical edge types
        self.edge_emb = nn.Embedding(num_edge_types, d_model // 2)
        
        # Graph Attention Layers
        self.layers = nn.ModuleList([
            TransformerConv(
                in_channels=d_model, 
                out_channels=d_model // 4, 
                heads=4, 
                concat=True, 
                beta=True, 
                edge_dim=d_model // 2
            )
            for _ in range(num_layers)
        ])
        
        # Normalization
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, d_model) node embeddings
            edge_index: (2, E) graph connectivity
            edge_type: (E,) long tensor of categorical edge types
            
        Returns:
            (N, d_model) context-aware node embeddings
        """
        # (E, d_model // 2)
        edge_attr = self.edge_emb(edge_type)
        
        for conv, norm in zip(self.layers, self.norms):
            res = x
            x = conv(x, edge_index, edge_attr)
            x = norm(x + res)
            x = torch.nn.functional.gelu(x)
            
        return x
