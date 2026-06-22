import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

# Ensure the root directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import NODE_TYPES
from data.loader import load_pages
from graph.builder import build_graphs
from models.jepa_core import DocumentJEPA
from models.vision import SelectiveVisualEncoder

def get_kaggle_dataset_path():
    """Helper to find the dataset path in Kaggle or fallback to local config."""
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "COCO" in dirs and "PNG" in dirs:
                print(f"Auto-detected Kaggle dataset at: {root}")
                return root
    return "doclaynet"  # Fallback to local config lookup in data/loader.py

def main():
    print("Initializing Phase 4 Kaggle Training Loop...")
    
    # 1. Device Setup (Kaggle T4x2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count()
    print(f"Using device: {device} | Number of GPUs: {num_gpus}")
    
    # 2. Model Setup
    print("Loading Models...")
    # Vision encoder acts as feature extractor, frozen
    vision_encoder = SelectiveVisualEncoder().to(device)
    vision_encoder.eval()
    
    jepa = DocumentJEPA(d_model=256).to(device)
    
    # Wrap JEPA in DataParallel to utilize both T4 GPUs if available
    if num_gpus > 1:
        print("Wrapping JEPA in DataParallel for multi-GPU training...")
        jepa = nn.DataParallel(jepa)
    
    # 3. Optimizer & Schedulers
    # Note: Only optimize context_encoder parameters. Target encoder is EMA updated.
    if num_gpus > 1:
        optimizer = optim.AdamW(jepa.module.context_encoder.parameters(), lr=1e-4, weight_decay=1e-4)
    else:
        optimizer = optim.AdamW(jepa.context_encoder.parameters(), lr=1e-4, weight_decay=1e-4)
        
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    
    # 4. Data Loading
    dataset_name = get_kaggle_dataset_path()
    print(f"Streaming data from: {dataset_name}")
    # Using a larger limit for Kaggle (e.g. 10000 pages per epoch)
    num_pages = 5000 
    
    cat_to_idx = {c: i for i, c in enumerate(NODE_TYPES)}
    edge_type_map = {'above': 0, 'below': 1, 'left_of': 2, 'right_of': 3, 'contains': 4, 'reading_order': 5}
    
    epochs = 5
    accumulation_steps = 16
    
    os.makedirs("checkpoints", exist_ok=True)
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        jepa.train()
        
        # We re-instantiate the generator each epoch
        pages_generator = load_pages(dataset_name, split="train", limit=num_pages)
        
        epoch_loss = 0.0
        valid_graphs = 0
        optimizer.zero_grad()
        
        for idx, page in enumerate(tqdm(pages_generator, total=num_pages, desc="Training")):
            # Load the actual PNG image for true Vision-Fusion training.
            try:
                # Based on the standard Kaggle dataset structure
                img_path = f"{dataset_name}/PNG/{page.page_id}.png"
                raw_image = Image.open(img_path).convert("RGB")
            except Exception as e:
                # Fallback to white image if something goes wrong so the loop doesn't instantly crash
                raw_image = Image.new('RGB', (1024, 1024), color='white')
            
            # Build graph and run Vision Cropper
            dg = build_graphs([page], raw_images=[raw_image], visual_encoder=vision_encoder)[0]
            G = dg.graph
            
            node_ids = list(G.nodes())
            if len(node_ids) < 5:
                continue
                
            id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
            
            # Prepare Tensors
            bbox_features, type_indices, visual_features_list, has_vision = [], [], [], []
            for nid in node_ids:
                attrs = G.nodes[nid]
                bbox = attrs.get('bbox', (0,0,0,0))
                area = attrs.get('area', 0)
                bbox_features.append([bbox[0]/1024, bbox[1]/1024, bbox[2]/1024, bbox[3]/1024, area/1000000])
                t = attrs.get('type', 'unknown')
                type_indices.append(cat_to_idx.get(t, len(NODE_TYPES)))
                vis_feat = attrs.get('visual_features')
                if vis_feat is not None:
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
            
            # Mask 15% of nodes
            num_nodes = len(node_ids)
            num_masked = max(1, int(num_nodes * 0.15))
            masked_indices = torch.randperm(num_nodes)[:num_masked].to(device)
            
            # Forward
            y_pred, y_true = jepa(
                bbox_features=bbox_features,
                type_indices=type_indices,
                edge_index=edge_index,
                edge_type=edge_type,
                masked_node_indices=masked_indices,
                visual_features=visual_features,
                has_vision_mask=has_vision_mask
            )
            
            # Loss computation
            loss = F.smooth_l1_loss(y_pred, y_true) / accumulation_steps
            loss.backward()
            
            epoch_loss += (loss.item() * accumulation_steps)
            valid_graphs += 1
            
            # Gradient Accumulation Step
            if (valid_graphs % accumulation_steps) == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                # EMA Update Target Encoder
                if num_gpus > 1:
                    jepa.module.update_target_encoder()
                else:
                    jepa.update_target_encoder()
                    
        # Step LR schedule
        scheduler.step()
        
        avg_loss = epoch_loss / max(1, valid_graphs)
        print(f"Epoch {epoch+1} Completed. Avg Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        model_to_save = jepa.module if num_gpus > 1 else jepa
        checkpoint_path = f"checkpoints/jepa_epoch_{epoch+1}.pt"
        torch.save(model_to_save.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    main()
