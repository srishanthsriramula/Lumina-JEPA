import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure the root directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import NODE_TYPES
from data.loader import load_pages
from graph.builder import build_graphs
from models.jepa_core import DocumentJEPA
from models.vision import SelectiveVisualEncoder
from models.llm_projector import GraphToLLMProjector

def get_kaggle_dataset_path():
    """Helper to find the dataset path in Kaggle or fallback to local config."""
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "COCO" in dirs and "PNG" in dirs:
                print(f"Auto-detected Kaggle dataset at: {root}")
                return root
    return "doclaynet"

def main():
    print("Initializing Phase 6: Instruction Tuning Loop...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Frozen Models
    print("Loading Frozen Vision and Graph Models...")
    vision_encoder = SelectiveVisualEncoder().to(device)
    vision_encoder.eval()
    
    jepa = DocumentJEPA(d_model=256).to(device)
    if os.path.exists("checkpoints/jepa_epoch_5.pt"):
        jepa.load_state_dict(torch.load("checkpoints/jepa_epoch_5.pt", map_location=device))
        print("Loaded JEPA epoch 5 weights.")
    else:
        print("WARNING: jepa_epoch_5.pt not found. Using untrained JEPA.")
    jepa.eval()
    
    # Freeze JEPA
    for param in jepa.parameters():
        param.requires_grad = False
        
    print("Loading Frozen TinyLlama...")
    llm_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    # Llama doesn't pad naturally, set pad token to eos
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    llm = AutoModelForCausalLM.from_pretrained(llm_name).to(device)
    llm.eval()
    # Freeze LLM
    for param in llm.parameters():
        param.requires_grad = False
        
    d_llm = llm.config.hidden_size
    
    # 2. Load Trainable Projector
    print("Initializing Trainable Projector...")
    projector = GraphToLLMProjector(d_model=256, d_llm=d_llm).to(device)
    projector.train()
    
    # Optimizer only updates projector!
    optimizer = optim.AdamW(projector.parameters(), lr=2e-4, weight_decay=1e-4)
    
    # 3. Data Setup
    dataset_name = get_kaggle_dataset_path()
    num_pages = 5000 
    cat_to_idx = {c: i for i, c in enumerate(NODE_TYPES)}
    edge_type_map = {'above': 0, 'below': 1, 'left_of': 2, 'right_of': 3, 'contains': 4, 'reading_order': 5}
    
    epochs = 3
    accumulation_steps = 16
    os.makedirs("checkpoints", exist_ok=True)
    
    # Training Loop
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        pages_generator = load_pages(dataset_name, split="train", limit=num_pages)
        
        epoch_loss = 0.0
        valid_graphs = 0
        optimizer.zero_grad()
        
        for idx, page in enumerate(tqdm(pages_generator, total=num_pages, desc="Tuning")):
            try:
                img_path = f"{dataset_name}/PNG/{page.page_id}.png"
                raw_image = Image.open(img_path).convert("RGB")
            except Exception as e:
                raw_image = Image.new('RGB', (1024, 1024), color='white')
            
            # --- VISION + GRAPH PROCESSING (Frozen) ---
            with torch.no_grad():
                dg = build_graphs([page], raw_images=[raw_image], visual_encoder=vision_encoder)[0]
                G = dg.graph
                node_ids = list(G.nodes())
                if len(node_ids) < 5:
                    continue
                id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
                
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
                
                src_edges, dst_edges, edge_types = [], [], []
                for u, v, data in G.edges(data=True):
                    src_edges.append(id_to_idx[u])
                    dst_edges.append(id_to_idx[v])
                    etype = data.get('edge_type', 'above')
                    edge_types.append(edge_type_map.get(etype, 0))
                    
                edge_index = torch.tensor([src_edges, dst_edges], dtype=torch.long).to(device) if src_edges else torch.empty((2, 0), dtype=torch.long).to(device)
                edge_type = torch.tensor(edge_types, dtype=torch.long).to(device) if edge_types else torch.empty((0,), dtype=torch.long).to(device)
                
                x = jepa.embedder(bbox_features, type_indices, visual_features, has_vision_mask)
                context_embs = jepa.context_encoder(x, edge_index, edge_type)
            
            # --- PROJECTOR (Trainable) ---
            # Forward pass through trainable projector
            graph_tokens = projector(context_embs).unsqueeze(0) # [1, num_nodes, d_llm]
            
            # --- QA SYNTHESIS ---
            # Create a synthetic question based on the document
            categories_present = set([e.category for e in page.elements])
            cat_string = ", ".join(categories_present)
            
            prompt = f"<|system|>\nYou are Lumina, a smart document assistant.<|end|>\n<|user|>\nBased on the document graph provided, what categories of layout elements are on this page?<|end|>\n<|assistant|>\n"
            target = f"The categories of layout elements on this page are: {cat_string}."
            full_text = prompt + target + tokenizer.eos_token
            
            # Tokenize
            prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
            full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"].to(device)
            
            # Labels for Causal LM (ignore prompt tokens)
            labels = full_ids.clone()
            labels[0, :prompt_ids.size(1)] = -100
            
            # --- LLM INJECTION (Frozen) ---
            with torch.no_grad():
                text_embeds = llm.get_input_embeddings()(full_ids)
            
            # Combine
            combined_embeds = torch.cat([graph_tokens, text_embeds], dim=1).to(llm.dtype)
            
            # Prefix labels for graph tokens (-100 because we don't predict graph tokens)
            graph_labels = torch.full((1, graph_tokens.size(1)), -100, dtype=torch.long).to(device)
            combined_labels = torch.cat([graph_labels, labels], dim=1)
            
            # Forward pass through LLM
            outputs = llm(inputs_embeds=combined_embeds, labels=combined_labels)
            
            loss = outputs.loss / accumulation_steps
            loss.backward()
            
            epoch_loss += (loss.item() * accumulation_steps)
            valid_graphs += 1
            
            if (valid_graphs % accumulation_steps) == 0:
                optimizer.step()
                optimizer.zero_grad()
                
        avg_loss = epoch_loss / max(1, valid_graphs)
        print(f"Epoch {epoch+1} Completed. Avg Projector Loss: {avg_loss:.4f}")
        
        # Save Projector
        torch.save(projector.state_dict(), f"checkpoints/projector_epoch_{epoch+1}.pt")
        print(f"Saved checkpoints/projector_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    main()
