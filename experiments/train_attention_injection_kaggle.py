import os
import json
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.jepa import DocumentGraphEncoder
from data.loader import DocLayNetKaggleDataset, collate_fn_doclaynet

KAGGLE_DATA_DIR = "/kaggle/input/datasets/srishanthsriramula/doclaynet-core"

def expand_bias_to_tokens(bias_matrix, box_token_lengths, device):
    """
    Expands a [num_boxes, num_boxes] bias matrix to [seq_len, seq_len]
    based on how many text tokens each box contains.
    """
    seq_len = sum(box_token_lengths)
    token_bias = torch.zeros((seq_len, seq_len), device=device)
    
    idx_i = 0
    for i, len_i in enumerate(box_token_lengths):
        if len_i == 0: continue
        idx_j = 0
        for j, len_j in enumerate(box_token_lengths):
            if len_j == 0: continue
            token_bias[idx_i:idx_i+len_i, idx_j:idx_j+len_j] = bias_matrix[i, j]
            idx_j += len_j
        idx_i += len_i
        
    return token_bias

def create_causal_4d_mask(token_bias, dtype, device):
    """
    Combines the structural bias with a standard Causal Mask.
    Returns shape: [1, 1, seq_len, seq_len]
    """
    seq_len = token_bias.shape[0]
    causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=device))
    
    # Where causal_mask is 0 (future tokens), set to -inf
    # Where causal_mask is 1 (past tokens), set to token_bias (JEPA structural bias)
    custom_mask = torch.where(causal_mask == 1, token_bias, torch.tensor(torch.finfo(dtype).min, device=device))
    
    return custom_mask.unsqueeze(0).unsqueeze(0).to(dtype)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load OCR Texts
    ocr_path = "ocr_texts.json"
    if not os.path.exists(ocr_path):
        print(f"ERROR: {ocr_path} not found. Please run scripts/preprocess_ocr_kaggle.py first.")
        return
    with open(ocr_path, 'r') as f:
        ocr_texts = json.load(f)
        
    # 2. Load Models
    print("Loading Models...")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    llm = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    # FREEZE LLM
    for param in llm.parameters():
        param.requires_grad = False
    llm.eval()
        
    # Load JEPA (Trainable)
    jepa = DocumentGraphEncoder(
        vision_dim=768,
        hidden_dim=256,
        num_layers=4,
        num_heads=8
    ).to(device)
    jepa.train()
    
    optimizer = optim.AdamW(jepa.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Load Dataset
    dataset = DocLayNetKaggleDataset(
        json_path=os.path.join(KAGGLE_DATA_DIR, "COCO", "val.json"),
        image_dir=os.path.join(KAGGLE_DATA_DIR, "PNG"),
        max_samples=5000 # Use a subset for testing, change to None for full run
    )
    
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    epochs = 3
    accumulation_steps = 4
    
    print("Starting End-to-End Graph-Biased Attention Training...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        progress_bar = tqdm(dataset, desc=f"Epoch {epoch+1}/{epochs}")
        for step, (page, raw_image) in enumerate(progress_bar):
            
            # Skip if image wasn't found in OCR JSON
            if page.page_id not in ocr_texts:
                continue
                
            page_ocr = ocr_texts[page.page_id]
            
            # 1. Prepare JEPA inputs
            if not page.elements:
                continue
                
            num_nodes = len(page.elements)
            # Mock visual features for speed (in a real run, run CLIP on crops)
            visual_features = torch.zeros((1, num_nodes, 768), device=device)
            
            # Create dense adjacency matrix
            adj_matrix = torch.ones((1, num_nodes, num_nodes), device=device)
            
            # Collect OCR text and tokenize
            box_token_lengths = []
            all_input_ids = []
            
            for element in page.elements:
                text = page_ocr.get(str(element.id), "")
                if not text.strip():
                    # Fallback text if OCR is empty
                    text = f"[{element.category}]"
                
                # Tokenize this box's text
                tokens = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
                box_token_lengths.append(len(tokens))
                all_input_ids.append(tokens)
                
            if sum(box_token_lengths) == 0:
                continue
                
            input_ids = torch.cat(all_input_ids).unsqueeze(0).to(device) # [1, seq_len]
            
            # 2. Forward JEPA -> Sculpt Attention Bias
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # jepa_output: [1, num_nodes, 256]
                jepa_output = jepa(visual_features, adj_matrix)
                
                # Calculate Cosine Similarity Matrix between all nodes: [num_nodes, num_nodes]
                nodes = jepa_output[0] # [num_nodes, 256]
                nodes_norm = F.normalize(nodes, p=2, dim=-1)
                sim_matrix = torch.matmul(nodes_norm, nodes_norm.transpose(0, 1))
                
                # Scale similarity to attention bias
                # If sim == 1.0 -> bias = 0.0
                # If sim == 0.0 -> bias = -10.0
                bias_matrix = (sim_matrix - 1.0) * 10.0
                
                # Expand to token level
                token_bias = expand_bias_to_tokens(bias_matrix, box_token_lengths, device)
                
                # Create 4D Causal Mask with structural bias
                attention_mask = create_causal_4d_mask(token_bias, llm.dtype, device)
                
                # 3. LLM Forward Pass (Frozen LLM, gradients flow back to JEPA)
                outputs = llm(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids # Next token prediction
                )
                
                loss = outputs.loss / accumulation_steps
                
            # Backward pass (flows through attention mask into JEPA)
            scaler.scale(loss).backward()
            
            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            epoch_loss += loss.item() * accumulation_steps
            progress_bar.set_postfix({'loss': f"{loss.item() * accumulation_steps:.4f}"})
            
        print(f"Epoch {epoch+1} Completed. Avg Loss: {epoch_loss / max(1, step):.4f}")
        torch.save(jepa.state_dict(), f"checkpoints/jepa_attention_epoch_{epoch+1}.pt")
        
    print("Training Complete! Saved JEPA models to checkpoints/")

if __name__ == "__main__":
    main()
