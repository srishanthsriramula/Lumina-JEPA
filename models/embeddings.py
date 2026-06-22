import torch
import torch.nn as nn
from config import NODE_TYPES

class NodeEmbedder(nn.Module):
    """
    Converts raw graph node features (bounding box geometry, categorical type,
    and optional visual features) into a dense latent vector for the GNN.
    """
    def __init__(self, d_model: int = 256, d_visual: int = 768):
        super().__init__()
        self.d_model = d_model
        
        # We have N node types + 1 for 'unknown'/masked
        self.num_classes = len(NODE_TYPES) + 1
        
        # Categorical embedding for layout types
        self.type_emb = nn.Embedding(self.num_classes, d_model)
        
        # Continuous embedding for bounding boxes: (x, y, w, h, area)
        self.bbox_emb = nn.Linear(5, d_model)
        
        # Visual feature projection (from CLIP dim to GNN dim)
        self.vis_proj = nn.Linear(d_visual, d_model)
        
        # Learned token for nodes that DO NOT have visual features (e.g., standard text)
        self.no_vision_token = nn.Parameter(torch.randn(1, d_model))
        
        # Final projection to mix geometric, semantic, and visual features
        self.proj = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )
        
    def forward(
        self, 
        bbox_features: torch.Tensor, 
        type_indices: torch.Tensor,
        visual_features: torch.Tensor = None,
        has_vision_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            bbox_features: (N, 5) float tensor [x, y, w, h, area]
            type_indices: (N,) long tensor
            visual_features: (N, d_visual) optional float tensor for visual crops
            has_vision_mask: (N,) bool tensor indicating which nodes actually have vision features
            
        Returns:
            (N, d_model) embedded node vectors
        """
        N = bbox_features.size(0)
        
        # Embed semantic types
        t_emb = self.type_emb(type_indices) # (N, d_model)
        
        # Embed geometric boxes
        b_emb = self.bbox_emb(bbox_features) # (N, d_model)
        
        # Prepare visual embeddings
        v_emb = self.no_vision_token.expand(N, -1).clone() # default to NO_VISION
        
        if visual_features is not None and has_vision_mask is not None:
            # Ensure has_vision_mask has at least one True to avoid empty tensor projection
            if has_vision_mask.any():
                projected_vis = self.vis_proj(visual_features[has_vision_mask])
                v_emb[has_vision_mask] = projected_vis
        
        # Mix them
        x = torch.cat([t_emb, b_emb, v_emb], dim=-1) # (N, d_model * 3)
        x = self.proj(x) # (N, d_model)
        
        return x
