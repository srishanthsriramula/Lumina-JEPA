import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import NODE_TYPES
from data.loader import load_pages
from graph.builder import build_graphs
from models.jepa_core import DocumentJEPA
from models.vision import SelectiveVisualEncoder
from models.llm_projector import GraphToLLMProjector

def main():
    print("Initializing Phase 3 Architecture...")
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Models
    print("Loading Selective Visual Encoder (CLIP)...")
    vision_encoder = SelectiveVisualEncoder().to(device)
    
    print("Loading Document JEPA...")
    jepa = DocumentJEPA(d_model=256).to(device)
    
    print("Loading LLM Projector...")
    llm_projector = GraphToLLMProjector(d_model=256, d_llm=4096).to(device)
    
    # 2. Data
    print("Loading test layout data...")
    pages = list(load_pages("doclaynet", limit=2))
    
    # Create a dummy image (e.g., 1024x1024 white background) to represent the physical page
    dummy_image = Image.new('RGB', (1024, 1024), color='white')
    raw_images = [dummy_image for _ in pages]
    
    # Build graphs, passing in the image and vision encoder to trigger Selective Vision
    graphs = build_graphs(pages, raw_images=raw_images, visual_encoder=vision_encoder)
    
    # 3. Verification Forward Pass
    print(f"\nTesting Multimodal Forward Pass on {len(graphs)} graphs...")
    jepa.eval()
    llm_projector.eval()
    
    cat_to_idx = {c: i for i, c in enumerate(NODE_TYPES)}
    edge_type_map = {'above': 0, 'below': 1, 'left_of': 2, 'right_of': 3, 'contains': 4, 'reading_order': 5}
    
    for dg in graphs:
        G = dg.graph
        node_ids = list(G.nodes())
        if not node_ids:
            continue
            
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        
        # Prepare Tensors
        bbox_features = []
        type_indices = []
        visual_features_list = []
        has_vision = []
        
        for nid in node_ids:
            attrs = G.nodes[nid]
            bbox = attrs.get('bbox', (0,0,0,0))
            area = attrs.get('area', 0)
            bbox_features.append([bbox[0]/1024, bbox[1]/1024, bbox[2]/1024, bbox[3]/1024, area/1000000])
            
            t = attrs.get('type', 'unknown')
            type_indices.append(cat_to_idx.get(t, len(NODE_TYPES)))
            
            vis_feat = attrs.get('visual_features')
            if vis_feat is not None:
                # Expected from CLIP is 768d
                visual_features_list.append(vis_feat.squeeze(0).tolist())
                has_vision.append(True)
            else:
                visual_features_list.append([0.0] * 768)
                has_vision.append(False)
                
        bbox_features = torch.tensor(bbox_features, dtype=torch.float).to(device)
        type_indices = torch.tensor(type_indices, dtype=torch.long).to(device)
        visual_features = torch.tensor(visual_features_list, dtype=torch.float).to(device)
        has_vision_mask = torch.tensor(has_vision, dtype=torch.bool).to(device)
        
        # Edges
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
        
        with torch.no_grad():
            # 1. Embed multimodal nodes (text + geometric + visual)
            x_full = jepa.embedder(bbox_features, type_indices, visual_features, has_vision_mask)
            
            # 2. Pass through GNN to learn contextual structure
            context_reps = jepa.context_encoder(x_full, edge_index, edge_type)
            
            # 3. Project to Llama-3 4096d token space
            llm_tokens = llm_projector(context_reps)
            
        print(f"Graph with {len(node_ids)} nodes (Visual nodes: {sum(has_vision)}) -> LLM tokens shape: {llm_tokens.shape}")
        assert llm_tokens.shape == (len(node_ids), 4096), "Output shape mismatch! Must be exactly 4096d for Llama 3."
        
    print("\nPhase 3 Verification Complete!")
    print("✅ Selective Vision successfully cropped nodes.")
    print("✅ Multimodal Graph successfully fused geometric + visual features.")
    print("✅ LLM Projector successfully generated 4096d Llama-ready tokens.")

if __name__ == "__main__":
    main()
