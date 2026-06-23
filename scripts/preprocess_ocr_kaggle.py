import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from paddleocr import PaddleOCR
import concurrent.futures
from functools import partial

# Dataset Path (Can be overridden by environment variable)
DATA_DIR = os.environ.get("DOCLAYNET_DIR", "./DocLayNet_core")

# Initialize OCR per process to avoid memory/thread issues
ocr_instance = None

def init_worker():
    global ocr_instance
    # Fast mode English OCR
    # Removing 'show_log' to fix ValueError and replacing 'use_angle_cls' with 'use_textline_orientation'
    ocr_instance = PaddleOCR(use_textline_orientation=False, lang='en')

def process_image(img_id, img_info, img_to_anns):
    global ocr_instance
    if ocr_instance is None:
        init_worker()
        
    img_filename = img_info['file_name']
    img_path = os.path.join(DATA_DIR, "PNG", img_filename)
    
    if not os.path.exists(img_path):
        return img_filename, None
        
    try:
        full_image = Image.open(img_path).convert("RGB")
        full_image_np = np.array(full_image)
        page_data = {}
        
        for ann in img_to_anns.get(img_id, []):
            x, y, w, h = [int(v) for v in ann['bbox']]
            x = max(0, x)
            y = max(0, y)
            w = min(w, full_image_np.shape[1] - x)
            h = min(h, full_image_np.shape[0] - y)
            
            if w <= 0 or h <= 0:
                page_data[ann['id']] = ""
                continue
                
            crop_np = full_image_np[y:y+h, x:x+w]
            result = ocr_instance.ocr(crop_np, cls=False)
            
            text_content = ""
            if result and result[0]:
                texts = [line[1][0] for line in result[0] if line is not None]
                text_content = " ".join(texts)
            
            page_data[ann['id']] = text_content
            
        return img_filename, page_data
    except Exception as e:
        print(f"Error processing {img_filename}: {e}")
        return img_filename, None

def main():
    splits = ["train.json", "val.json", "test.json"]
    images = {}
    img_to_anns = {}
    
    for split in splits:
        json_path = os.path.join(DATA_DIR, "COCO", split)
        if not os.path.exists(json_path):
            print(f"Skipping {split}: not found")
            continue
            
        with open(json_path, 'r') as f:
            coco_data = json.load(f)
            
        for img in coco_data['images']:
            images[img['id']] = img
            
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)
            
    if not images:
        print("Error: No images found in any dataset split!")
        return
        
    print(f"Found {len(images)} total images across all splits. Starting multiprocessing...")
    
    output_data = {}
    output_file = "ocr_texts.json"
    
    # Use ProcessPoolExecutor to use all cores on the A100
    max_workers = os.cpu_count() or 4
    print(f"Using {max_workers} cores for OCR extraction.")
    
    process_func = partial(process_image, img_to_anns=img_to_anns)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker) as executor:
        # Map inputs
        futures = {executor.submit(process_func, img_id, img_info): img_filename 
                   for img_id, img_info in images.items()}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(images), desc="Extracting OCR Text"):
            img_filename, page_data = future.result()
            if page_data is not None:
                output_data[img_filename] = page_data
                
            if len(output_data) > 0 and len(output_data) % 500 == 0:
                with open(output_file, 'w') as f:
                    json.dump(output_data, f)
                    
    with open(output_file, 'w') as f:
        json.dump(output_data, f)
        
    print(f"\nSuccessfully extracted text for {len(output_data)} total documents.")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()
