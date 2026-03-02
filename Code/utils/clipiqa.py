# utils/clipiqa.py
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import clip

class ClipIQA:
    def __init__(self, weight_path, device="cuda"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)

        print(f"[ClipIQA] Loading weight: {weight_path}")
        w = torch.load(weight_path, map_location=device)

        if "text_feat" in w:
            self.text_features = w["text_feat"].to(device)

        elif "text_features" in w:
            self.text_features = w["text_features"].to(device)

        elif "attributes" in w:
            self.text_features = w["attributes"].to(device)

        else:
            print("[ClipIQA] WARNING: No text_feat found, using default class prompts.")
            text = ["Good photo.", "Bad photo."]
            with torch.no_grad():
                texts = clip.tokenize(text).to(device)
                self.text_features = F.normalize(self.model.encode_text(texts), dim=-1)

        if "alpha" in w:
            self.alpha = w["alpha"].item() if isinstance(w["alpha"], torch.Tensor) else w["alpha"]
        else:
            self.alpha = 5.0 
        self.text_features = F.normalize(self.text_features, dim=-1)

        print("[ClipIQA] Loaded successfully.")

    def score_tensor(self, img_tensor):
        to_pil = transforms.ToPILImage()
        img = to_pil(img_tensor.cpu())
        return self.score(img)

    def score(self, img_pil):
        image_input = self.preprocess(img_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_feat = F.normalize(self.model.encode_image(image_input), dim=-1)
            sim = (image_feat @ self.text_features.T) * self.alpha
            weights = torch.arange(1, sim.shape[-1] + 1, device=self.device, dtype=sim.dtype)
            score = sim.softmax(dim=-1) @ weights
        return score.item()
