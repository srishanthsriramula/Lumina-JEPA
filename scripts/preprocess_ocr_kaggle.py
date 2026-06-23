import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from paddleocr import PaddleOCR
import concurrent.futures
from functools import partial

# Kaggle Dataset Path
KAGGLE_DATA_DIR = "/kaggle/input/datasets/srishanthsriramula/doclaynet-core"

# Initialize OCR per process to avoid memory/thread issues
ocr_instance = None

def init_worker():
    global ocr_instance
    # Fast mode English OCR
    ocr_instance = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)

def process_image(img_id, img_info, img_to_anns):
    global ocr_instance
    if ocr_instance is None:
        init_worker()
        
    img_filename = img_info['file_name']
    img_path = os.path.join(KAGGLE_DATA_DIR, "PNG", img_filename)
    
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
    json_path = os.path.join(KAGGLE_DATA_DIR, "COCO", "val.json")
    if not os.path.exists(json_path):
        print(f"Error: Could not find {json_path}")
        return
        
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
        
    images = {img['id']: img for img in coco_data['images']}
    annotations = coco_data['annotations']
    
    img_to_anns = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
        
    print(f"Found {len(images)} images in validation set. Starting multiprocessing...")
    
    output_data = {}
    output_file = "ocr_texts.json"
    
    # Use ThreadPoolExecutor or ProcessPoolExecutor
    # Kaggle usually has 4+ cores. ProcessPool is safer for PaddleOCR memory.
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
        
    print(f"\nSuccessfully extracted text for {len(output_data)} documents.")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()
