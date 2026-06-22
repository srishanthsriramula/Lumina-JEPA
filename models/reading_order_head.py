import torch
import torch.nn as nn

class ReadingOrderPredictor(nn.Module):
    """
    Neural head that sits on top of the GNN.
    Given contextualized representations of two nodes, it predicts a score
    indicating how likely Node A is read immediately before Node B.
    """
    def __init__(self, d_model: int = 256):
        super().__init__()
        
        # Takes the concatenation of (Node A, Node B)
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, node_a: torch.Tensor, node_b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_a: (B, d_model) Contextualized representations of source nodes
            node_b: (B, d_model) Contextualized representations of target nodes
            
        Returns:
            (B, 1) Logits indicating the strength of the A -> B reading order connection
        """
        x = torch.cat([node_a, node_b], dim=-1)
        logits = self.mlp(x)
        return logits
