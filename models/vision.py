import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPVisionModel, CLIPImageProcessor

class SelectiveVisualEncoder(nn.Module):
    """
    Takes a full high-resolution document image and bounding boxes of complex visual elements.
    Crops the regions and passes them through a frozen CLIP model.
    This acts as the "Fovea" of the system, only expending compute on areas that need visual nuance.
    """
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        # Frozen feature extractor to save compute
        self.vision_model = CLIPVisionModel.from_pretrained(model_name)
        for param in self.vision_model.parameters():
            param.requires_grad = False
            
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.d_visual = self.vision_model.config.hidden_size # 768 for clip-vit-base
        
    @torch.no_grad()
    def forward(self, image: Image.Image, bboxes: list[tuple[float, float, float, float]]) -> torch.Tensor:
        """
        Args:
            image: A PIL Image of the full document page
            bboxes: List of (x, y, w, h) bounding boxes to crop and encode
            
        Returns:
            (N, d_visual) tensor of visual features for the N boxes
        """
        if not bboxes:
            return torch.empty((0, self.d_visual), device=self.vision_model.device)
            
        crops = []
        for (x, y, w, h) in bboxes:
            # PIL crop expects (left, upper, right, lower)
            # Add small padding to avoid empty crops on edge cases
            w = max(1.0, w)
            h = max(1.0, h)
            crop = image.crop((x, y, x + w, y + h))
            crops.append(crop)
            
        # Process images into tensor
        inputs = self.processor(images=crops, return_tensors="pt")
        inputs = {k: v.to(self.vision_model.device) for k, v in inputs.items()}
        
        # Get pooler_output (the CLS token representation for the whole crop)
        outputs = self.vision_model(**inputs)
        return outputs.pooler_output
