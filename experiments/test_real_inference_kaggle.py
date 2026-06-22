import torch
import sys
import os
from pathlib import Path
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.jepa_core import DocumentJEPA
from models.vision import SelectiveVisualEncoder
from models.llm_projector import GraphToLLMProjector
from graph.builder import build_graphs
from data.loader import load_pages

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Lumina-JEPA Core Models
    print("Loading Lumina-JEPA modules...")
    jepa = DocumentJEPA(d_model=256).to(device)
    
    # Load trained weights
    jepa_checkpoint = "checkpoints/jepa_epoch_5.pt"
    if os.path.exists(jepa_checkpoint):
        jepa.load_state_dict(torch.load(jepa_checkpoint, map_location=device))
        print("Successfully loaded trained JEPA weights!")
    else:
        print("Warning: JEPA weights not found. Using untrained JEPA.")

    vision_encoder = SelectiveVisualEncoder().to(device)
    vision_encoder.eval()
    jepa.eval()

    # 2. Load TinyLlama
    print("Loading TinyLlama...")
    llm_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    llm = AutoModelForCausalLM.from_pretrained(llm_name).to(device)
    llm.eval()
    d_llm = llm.config.hidden_size 
    
    # 3. Load Projector
    projector = GraphToLLMProjector(d_model=256, d_llm=d_llm).to(device)
    projector_path = "checkpoints/projector_epoch_3.pt"
    if os.path.exists(projector_path):
        projector.load_state_dict(torch.load(projector_path, map_location=device))
        print("Successfully loaded trained Projector weights!")
    else:
        print("Warning: Projector weights not found. Using untrained Projector.")
    projector.eval()

    # 4. Load a Real Document Page from Kaggle
    dataset_path = "/kaggle/input/datasets/srishanthsriramula/doclaynet-core"
    print(f"Loading sample document from {dataset_path}...")
    
    # load_pages is a generator, get the first page
    pages_iter = load_pages(dataset_path, split="val", limit=1)
    try:
        page = next(pages_iter)
    except StopIteration:
        print("Failed to load page. Make sure the Kaggle dataset path is correct and COCO val.json exists.")
        return

    # Load the corresponding image
    png_dir = os.path.join(dataset_path, "PNG")
    img_path = os.path.join(png_dir, f"{page.page_id}.png")
    
    if not os.path.exists(img_path):
        print(f"Image not found at {img_path}. Proceeding with blank image (vision embeddings will be zero).")
        raw_image = Image.new('RGB', (1024, 1024), color='white')
    else:
        raw_image = Image.open(img_path).convert('RGB')
        print(f"Loaded real document image: {page.page_id}.png")

    # 5. Process Graph through Lumina-JEPA
    print(f"Processing Graph with {len(page.elements)} layout elements through JEPA...")
    with torch.no_grad():
        dg = build_graphs([page], raw_images=[raw_image], visual_encoder=vision_encoder)[0]
        G = dg.graph
        
        node_ids = list(G.nodes())
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        
        # Prepare Tensors
        from config import NODE_TYPES
        cat_to_idx = {c: i for i, c in enumerate(NODE_TYPES)}
        bbox_features, type_indices, visual_features_list, has_vision = [], [], [], []
        
        for nid in node_ids:
            attrs = G.nodes[nid]
            bbox = attrs.get('bbox', (0,0,0,0))
            area = attrs.get('area', 0)
            bbox_features.append([bbox[0]/page.width if page.width else 0, 
                                  bbox[1]/page.height if page.height else 0, 
                                  bbox[2]/page.width if page.width else 0, 
                                  bbox[3]/page.height if page.height else 0, 
                                  area/1000000])
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
        edge_type_map = {'above': 0, 'below': 1, 'left_of': 2, 'right_of': 3, 'contains': 4, 'reading_order': 5}
        src_edges, dst_edges, edge_types = [], [], []
        for u, v, data in G.edges(data=True):
            src_edges.append(id_to_idx[u])
            dst_edges.append(id_to_idx[v])
            etype = data.get('edge_type', 'above')
            edge_types.append(edge_type_map.get(etype, 0))
            
        edge_index = torch.tensor([src_edges, dst_edges], dtype=torch.long).to(device) if src_edges else torch.empty((2, 0), dtype=torch.long).to(device)
        edge_type = torch.tensor(edge_types, dtype=torch.long).to(device) if edge_types else torch.empty((0,), dtype=torch.long).to(device)
        
        # Embed raw features into node features x
        x = jepa.embedder(bbox_features, type_indices, visual_features, has_vision_mask)
        
        # Get context embeddings from JEPA context_encoder
        context_embs = jepa.context_encoder(x, edge_index, edge_type)
        
        # Project to LLM dimensions
        graph_tokens = projector(context_embs) # Shape: [num_nodes, d_llm]
        print(f"Graph Tokens Shape: {graph_tokens.shape}")

    # 6. Interleave with LLM Text
    print("Injecting into TinyLlama...")
    
    # We ask it a question about the document structure
    prompt = f"<|system|>\nYou are Lumina, an intelligent multimodal document analyzer. Answer questions about the provided document layout.<|end|>\n<|user|>\nBased on the layout and visual structure provided, what type of document is this and what is its main subject?<|end|>\n<|assistant|>"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    # Get standard text embeddings from TinyLlama
    text_embeds = llm.get_input_embeddings()(input_ids) # Shape: [1, seq_len, d_llm]
    
    # Prepend graph tokens to text embeddings (like a visual prefix)
    graph_tokens = graph_tokens.unsqueeze(0)
    combined_embeds = torch.cat([graph_tokens, text_embeds], dim=1)
    
    # Cast to LLM's expected dtype (e.g., bfloat16 for TinyLlama)
    combined_embeds = combined_embeds.to(llm.dtype)
    
    # 7. Generate Response
    print("Generating response...")
    with torch.no_grad():
        outputs = llm.generate(
            inputs_embeds=combined_embeds,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
    print("\n\n--- LLM Output ---\n")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("\n------------------")
    print("\nSuccess! The pipeline physically hooks up to real Kaggle data!")

if __name__ == "__main__":
    main()
