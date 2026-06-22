import os
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import NODE_TYPES
from data.loader import load_pages
from graph.builder import build_graphs
from models.jepa_core import DocumentJEPA

def main():
    print("Initializing Document JEPA...")
    # Use MPS if on Mac, otherwise CPU/CUDA
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 128 is small enough for fast CPU testing if MPS fails
    model = DocumentJEPA(d_model=128).to(device)
    optimizer = optim.AdamW(model.context_encoder.parameters(), lr=1e-3)
    
    print("Loading test data...")
    # Load 16 real pages
    pages = list(load_pages("doclaynet", limit=16))
    graphs = build_graphs(pages)
    
    # Mapping definitions
    edge_type_map = {'above': 0, 'below': 1, 'left_of': 2, 'right_of': 3, 'contains': 4, 'reading_order': 5}
    cat_to_idx = {c: i for i, c in enumerate(NODE_TYPES)}
    
    print(f"Starting verification training loop on {len(graphs)} graphs...")
    model.train()
    
    for epoch in range(5):
        epoch_loss = 0.0
        valid_graphs = 0
        
        for dg in graphs:
            G = dg.graph
            if G.number_of_nodes() < 5:
                continue
                
            # Prepare tensors
            bbox_features = []
            type_indices = []
            node_ids = list(G.nodes())
            id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
            
            for nid in node_ids:
                attrs = G.nodes[nid]
                bbox = attrs.get('bbox', (0,0,0,0))
                area = attrs.get('area', 0)
                # Normalize bbox roughly to [0, 1] for stable embedding
                bbox_features.append([bbox[0]/1024, bbox[1]/1024, bbox[2]/1024, bbox[3]/1024, area/1000000])
                t = attrs.get('type', 'unknown')
                type_indices.append(cat_to_idx.get(t, len(NODE_TYPES)))
                
            bbox_features = torch.tensor(bbox_features, dtype=torch.float).to(device)
            type_indices = torch.tensor(type_indices, dtype=torch.long).to(device)
            
            src_edges, dst_edges, edge_types = [], [], []
            for u, v, data in G.edges(data=True):
                src_edges.append(id_to_idx[u])
                dst_edges.append(id_to_idx[v])
                etype = data.get('edge_type', 'above')
                edge_types.append(edge_type_map.get(etype, 0))
                
            if not src_edges:
                continue
                
            edge_index = torch.tensor([src_edges, dst_edges], dtype=torch.long).to(device)
            edge_type = torch.tensor(edge_types, dtype=torch.long).to(device)
            
            # Mask 15% of nodes
            num_nodes = len(node_ids)
            num_masked = max(1, int(num_nodes * 0.15))
            masked_indices = torch.randperm(num_nodes)[:num_masked].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            y_pred, y_true = model(bbox_features, type_indices, edge_index, edge_type, masked_indices)
            
            # Smooth L1 Loss between predicted latent context and target target context
            loss = F.smooth_l1_loss(y_pred, y_true)
            
            loss.backward()
            optimizer.step()
            
            # EMA Update for Target Encoder
            model.update_target_encoder()
            
            epoch_loss += loss.item()
            valid_graphs += 1
            
        print(f"Epoch {epoch + 1}/5 - JEPA Loss: {epoch_loss/valid_graphs:.4f}")
        
    print("Verification complete! The architecture compiles and trains successfully.")

if __name__ == "__main__":
    main()
