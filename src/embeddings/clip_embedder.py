import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class CLIPEmbedder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def embed_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            vision_outputs = self.model.vision_model(**inputs)
            pooled = vision_outputs.pooler_output  # ✅ tensor

            features = self.model.visual_projection(pooled)  # ✅ project to CLIP space

        # ✅ normalize
        features = features / features.norm(p=2, dim=-1, keepdim=True)

        return features.detach().cpu().numpy()[0]

    def embed_text(self, text):
        inputs = self.processor(text=[text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            text_outputs = self.model.text_model(**inputs)
            pooled = text_outputs.pooler_output  # ✅ tensor

            features = self.model.text_projection(pooled)  # ✅ project to CLIP space

        # ✅ normalize
        features = features / features.norm(p=2, dim=-1, keepdim=True)

        return features.detach().cpu().numpy()[0]