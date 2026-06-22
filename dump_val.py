import json
import os
from datasets import load_dataset

ds = load_dataset("pierreguillou/DocLayNet-small", split="validation", trust_remote_code=True)

# We want to format this into the COCO format expected by loader.py
coco_output = {
    "images": [],
    "annotations": [],
    "categories": []
}

# Add categories
cats = ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']
for i, name in enumerate(cats):
    coco_output["categories"].append({"id": i + 1, "name": name})

ann_id = 1
for i, sample in enumerate(ds):
    image_id = i + 1
    # HF images usually don't have file_name, but we can generate one
    file_name = f"doclaynet_val_{image_id}.png"
    width = sample.get("coco_width", sample.get("original_width", 1024))
    height = sample.get("coco_height", sample.get("original_height", 1024))
    
    coco_output["images"].append({
        "id": image_id,
        "width": width,
        "height": height,
        "file_name": file_name
    })
    
    bboxes = sample["bboxes_block"]
    cat_ids = sample["categories"]
    
    for bbox, cat_idx in zip(bboxes, cat_ids):
        # HF ClassLabel is 0-indexed, but our COCO categories are 1-indexed
        category_id = cat_idx + 1
        
        coco_output["annotations"].append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox, # [x, y, w, h]
            "area": bbox[2] * bbox[3]
        })
        ann_id += 1

out_dir = "/Users/srishanthsriramula/code/vision-SARVAM/data_store/doclaynet/COCO"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "val.json")

with open(out_path, "w") as f:
    json.dump(coco_output, f)

print(f"Dumped {len(coco_output['images'])} images and {len(coco_output['annotations'])} annotations to {out_path}")
