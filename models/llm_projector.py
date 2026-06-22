import torch
import torch.nn as nn

class GraphToLLMProjector(nn.Module):
    """
    Translates the highly-compressed document graph embeddings (from the JEPA)
    into the exact dimension expected by a target Large Language Model (e.g., Llama 3).
    This allows the LLM to 'read' the graph tokens natively.
    """
    def __init__(self, d_model: int = 256, d_llm: int = 4096):
        super().__init__()
        
        # A simple 2-layer MLP is standard practice for adapter projections 
        # (similar to LLaVA's vision-language projector)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_llm)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, d_model) Contextualized node embeddings from JEPA
            
        Returns:
            (N, d_llm) LLM-ready embeddings
        """
        return self.proj(x)
