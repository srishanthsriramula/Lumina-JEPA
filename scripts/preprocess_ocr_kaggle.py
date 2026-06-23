import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from paddleocr import PaddleOCR

# Kaggle Dataset Path
KAGGLE_DATA_DIR = "/kaggle/input/datasets/srishanthsriramula/doclaynet-core"

def main():
    print("Initializing PaddleOCR...")
    # Initialize PaddleOCR (English, fast mode)
    ocr = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)
    
    # Load COCO JSON
    json_path = os.path.join(KAGGLE_DATA_DIR, "COCO", "val.json")
    if not os.path.exists(json_path):
        print(f"Error: Could not find {json_path}")
        return
        
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
        
    images = {img['id']: img for img in coco_data['images']}
    annotations = coco_data['annotations']
    
    # Group annotations by image
    img_to_anns = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
        
    print(f"Found {len(images)} images in validation set.")
    
    output_data = {}
    output_file = "ocr_texts.json"
    
    # Process each image
    # Note: On a massive dataset, we would batch this or use PyTorch DataLoader
    # For now, we process sequentially with a progress bar
    for img_id, img_info in tqdm(images.items(), desc="Extracting OCR Text"):
        img_filename = img_info['file_name']
        img_path = os.path.join(KAGGLE_DATA_DIR, "PNG", img_filename)
        
        if not os.path.exists(img_path):
            continue
            
        try:
            # Load full image
            full_image = Image.open(img_path).convert("RGB")
            full_image_np = np.array(full_image)
            
            page_data = {}
            
            # Process each bounding box in the image
            for ann in img_to_anns.get(img_id, []):
                x, y, w, h = [int(v) for v in ann['bbox']]
                
                # Ensure valid crop coordinates
                x = max(0, x)
                y = max(0, y)
                w = min(w, full_image_np.shape[1] - x)
                h = min(h, full_image_np.shape[0] - y)
                
                if w <= 0 or h <= 0:
                    page_data[ann['id']] = ""
                    continue
                    
                # Crop image array
                crop_np = full_image_np[y:y+h, x:x+w]
                
                # Run PaddleOCR
                # OCR returns a list of lines, each line is [coords, (text, confidence)]
                result = ocr.ocr(crop_np, cls=False)
                
                text_content = ""
                if result and result[0]:
                    # Join all detected text lines in the box
                    texts = [line[1][0] for line in result[0] if line is not None]
                    text_content = " ".join(texts)
                
                page_data[ann['id']] = text_content
                
            output_data[img_filename] = page_data
            
        except Exception as e:
            print(f"Error processing {img_filename}: {e}")
            
        # Periodically save to prevent data loss on massive runs
        if len(output_data) % 100 == 0:
            with open(output_file, 'w') as f:
                json.dump(output_data, f)
                
    # Final save
    with open(output_file, 'w') as f:
        json.dump(output_data, f)
        
    print(f"\nSuccessfully extracted text for {len(output_data)} documents.")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()
