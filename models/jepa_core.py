import copy
import torch
import torch.nn as nn
from .embeddings import NodeEmbedder
from .gnn_encoder import DocumentGraphEncoder

class DocumentJEPA(nn.Module):
    """
    Joint-Embedding Predictive Architecture for Documents.
    Self-supervised pretraining: predicts the latent representations of masked nodes 
    using the unmasked graph context.
    """
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model
        
        # Base embedder
        self.embedder = NodeEmbedder(d_model=d_model)
        
        # Context Encoder (learns via backprop)
        self.context_encoder = DocumentGraphEncoder(d_model=d_model, num_layers=4)
        
        # Target Encoder (EMA copy, no gradients)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
        # Predictor (MLP mapping Context representations to Target representations)
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # Learnable mask token to replace masked node features
        self.mask_token = nn.Parameter(torch.randn(1, d_model))
        
    @torch.no_grad()
    def update_target_encoder(self, momentum: float = 0.996):
        """EMA momentum update for target encoder parameters."""
        for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(momentum).add_(param_q.data, alpha=1.0 - momentum)
            
    def forward(
        self, 
        bbox_features: torch.Tensor, 
        type_indices: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_type: torch.Tensor,
        masked_node_indices: torch.Tensor,
        visual_features: torch.Tensor = None,
        has_vision_mask: torch.Tensor = None
    ):
        """
        JEPA Training Step:
        1. Base embedding of all nodes (Multimodal).
        2. Generate targets via Target Encoder (unmasked).
        3. Mask the selected nodes in the context.
        4. Pass through Context Encoder.
        5. Predict targets.
        """
        # 1. Base Embeddings
        x_full = self.embedder(bbox_features, type_indices, visual_features, has_vision_mask)
        
        # 2. Target Generation (No Gradients)
        with torch.no_grad():
            x_target = self.target_encoder(x_full, edge_index, edge_type)
            # We only care about predicting the masked nodes
            y_true = x_target[masked_node_indices]
            
        # 3. Context Masking
        x_context = x_full.clone()
        x_context[masked_node_indices] = self.mask_token
        
        # 4. Context Encoding
        x_context = self.context_encoder(x_context, edge_index, edge_type)
        
        # 5. Prediction
        context_reps = x_context[masked_node_indices]
        y_pred = self.predictor(context_reps)
        
        # The loss is computed outside (usually Smooth L1 or Cosine Similarity)
        return y_pred, y_true
