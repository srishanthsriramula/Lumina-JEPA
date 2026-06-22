import json
import os
import sys

def generate_kaggle_notebook():
    notebook_path = os.path.join(os.path.dirname(__file__), "train_jepa_kaggle.ipynb")
    script_path = os.path.join(os.path.dirname(__file__), "train_jepa_kaggle.py")
    
    with open(script_path, "r", encoding="utf-8") as f:
        script_content = f.read()
        
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Vision-SARVAM: JEPA Pretraining (Dual T4 GPU)\n",
                    "This notebook trains the Document JEPA with Selective Vision active from Epoch 1.\n",
                    "Ensure you have cloned the `vision-SARVAM` repository and set the Kaggle accelerator to **GPU T4 x2**."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "!git clone https://github.com/srishanthsriramula/Lumina-JEPA.git\n",
                    "%cd Lumina-JEPA\n",
                    "!pip install -r requirements.txt\n",
                    "!pip install fiftyone\n",
                    "!python data/download.py --dataset doclaynet"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import sys\n",
                    "import torch\n",
                    "import torch.nn as nn\n",
                    "import torch.nn.functional as F\n",
                    "import torch.optim as optim\n",
                    "from torch.utils.data import DataLoader\n",
                    "from tqdm import tqdm\n",
                    "from PIL import Image\n",
                    "\n",
                    "from config import NODE_TYPES\n",
                    "from data.loader import load_pages\n",
                    "from graph.builder import build_graphs\n",
                    "from models.jepa_core import DocumentJEPA\n",
                    "from models.vision import SelectiveVisualEncoder\n",
                    "\n",
                    "def main():\n",
                    "    print(\"Initializing Kaggle T4x2 Training...\")\n",
                    "    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                    "    num_gpus = torch.cuda.device_count()\n",
                    "    print(f\"Using device: {device} | Number of GPUs: {num_gpus}\")\n",
                    "    \n",
                    "    print(\"Loading Models...\")\n",
                    "    vision_encoder = SelectiveVisualEncoder().to(device)\n",
                    "    vision_encoder.eval()\n",
                    "    jepa = DocumentJEPA(d_model=256).to(device)\n",
                    "    if num_gpus > 1:\n",
                    "        jepa = nn.DataParallel(jepa)\n",
                    "        \n",
                    "    # Optimizer & Schedulers\n",
                    "    if num_gpus > 1:\n",
                    "        optimizer = optim.AdamW(jepa.module.context_encoder.parameters(), lr=1e-4, weight_decay=1e-4)\n",
                    "    else:\n",
                    "        optimizer = optim.AdamW(jepa.context_encoder.parameters(), lr=1e-4, weight_decay=1e-4)\n",
                    "    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)\n",
                    "    \n",
                    "    # Using a subset for demonstration. On Kaggle, scale this up.\n",
                    "    num_pages = 5000\n",
                    "    cat_to_idx = {c: i for i, c in enumerate(NODE_TYPES)}\n",
                    "    edge_type_map = {'above': 0, 'below': 1, 'left_of': 2, 'right_of': 3, 'contains': 4, 'reading_order': 5}\n",
                    "    epochs = 5\n",
                    "    accumulation_steps = 16\n",
                    "    os.makedirs(\"checkpoints\", exist_ok=True)\n",
                    "    \n",
                    "    for epoch in range(epochs):\n",
                    "        print(f\"\\n--- Epoch {epoch+1}/{epochs} ---\")\n",
                    "        jepa.train()\n",
                    "        pages_generator = load_pages(\"doclaynet\", split=\"train\", limit=num_pages)\n",
                    "        epoch_loss, valid_graphs = 0.0, 0\n",
                    "        optimizer.zero_grad()\n",
                    "        \n",
                    "        for idx, page in enumerate(tqdm(pages_generator, total=num_pages, desc=\"Training\")):\n",
                    "            raw_image = Image.new('RGB', (1024, 1024), color='white') # Replace with actual image path\n",
                    "            dg = build_graphs([page], raw_images=[raw_image], visual_encoder=vision_encoder)[0]\n",
                    "            G = dg.graph\n",
                    "            node_ids = list(G.nodes())\n",
                    "            if len(node_ids) < 5: continue\n",
                    "                \n",
                    "            id_to_idx = {nid: i for i, nid in enumerate(node_ids)}\n",
                    "            bbox_features, type_indices, visual_features_list, has_vision = [], [], [], []\n",
                    "            for nid in node_ids:\n",
                    "                attrs = G.nodes[nid]\n",
                    "                bbox, area = attrs.get('bbox', (0,0,0,0)), attrs.get('area', 0)\n",
                    "                bbox_features.append([bbox[0]/1024, bbox[1]/1024, bbox[2]/1024, bbox[3]/1024, area/1000000])\n",
                    "                type_indices.append(cat_to_idx.get(attrs.get('type', 'unknown'), len(NODE_TYPES)))\n",
                    "                vis_feat = attrs.get('visual_features')\n",
                    "                if vis_feat is not None:\n",
                    "                    visual_features_list.append(vis_feat.squeeze(0).tolist())\n",
                    "                    has_vision.append(True)\n",
                    "                else:\n",
                    "                    visual_features_list.append([0.0] * 768)\n",
                    "                    has_vision.append(False)\n",
                    "                    \n",
                    "            bbox_features = torch.tensor(bbox_features, dtype=torch.float).to(device)\n",
                    "            type_indices = torch.tensor(type_indices, dtype=torch.long).to(device)\n",
                    "            visual_features = torch.tensor(visual_features_list, dtype=torch.float).to(device)\n",
                    "            has_vision_mask = torch.tensor(has_vision, dtype=torch.bool).to(device)\n",
                    "            \n",
                    "            src_edges, dst_edges, edge_types = [], [], []\n",
                    "            for u, v, data in G.edges(data=True):\n",
                    "                src_edges.append(id_to_idx[u])\n",
                    "                dst_edges.append(id_to_idx[v])\n",
                    "                edge_types.append(edge_type_map.get(data.get('edge_type', 'above'), 0))\n",
                    "                \n",
                    "            if not src_edges: continue\n",
                    "            edge_index = torch.tensor([src_edges, dst_edges], dtype=torch.long).to(device)\n",
                    "            edge_type = torch.tensor(edge_types, dtype=torch.long).to(device)\n",
                    "            \n",
                    "            num_nodes = len(node_ids)\n",
                    "            masked_indices = torch.randperm(num_nodes)[:max(1, int(num_nodes * 0.15))].to(device)\n",
                    "            \n",
                    "            y_pred, y_true = jepa(\n",
                    "                bbox_features=bbox_features, type_indices=type_indices, edge_index=edge_index,\n",
                    "                edge_type=edge_type, masked_node_indices=masked_indices,\n",
                    "                visual_features=visual_features, has_vision_mask=has_vision_mask\n",
                    "            )\n",
                    "            \n",
                    "            loss = F.smooth_l1_loss(y_pred, y_true) / accumulation_steps\n",
                    "            loss.backward()\n",
                    "            epoch_loss += (loss.item() * accumulation_steps)\n",
                    "            valid_graphs += 1\n",
                    "            \n",
                    "            if (valid_graphs % accumulation_steps) == 0:\n",
                    "                optimizer.step()\n",
                    "                optimizer.zero_grad()\n",
                    "                if num_gpus > 1: jepa.module.update_target_encoder()\n",
                    "                else: jepa.update_target_encoder()\n",
                    "                    \n",
                    "        scheduler.step()\n",
                    "        print(f\"Epoch {epoch+1} Completed. Avg Loss: {epoch_loss / max(1, valid_graphs):.4f}\")\n",
                    "        model_to_save = jepa.module if num_gpus > 1 else jepa\n",
                    "        torch.save(model_to_save.state_dict(), f\"checkpoints/jepa_epoch_{epoch+1}.pt\")\n",
                    "\n",
                    "if __name__ == \"__main__\":\n",
                    "    main()\n"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
        
    print(f"Successfully generated Kaggle Notebook at: {notebook_path}")

if __name__ == "__main__":
    generate_kaggle_notebook()
