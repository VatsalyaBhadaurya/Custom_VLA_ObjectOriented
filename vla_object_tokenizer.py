# this is object aware tokenizer for vla

import os
import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
import open_clip

os.environ['YOLO_CACHE_DIR'] = 'E:/yolo_cache'
os.environ['HF_HOME'] = 'E:/huggingface_cache'
os.environ['TORCH_HOME'] = 'E:/torch_cache'


class ObjectVLATokenizer:
    def __init__(self):
        print("Loading models...")
        self.yolo = YOLO("yolov8n.pt")  # Object detection
        
        self.clip_model, _, self.clip_proc = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/clip-ViT-B-32")
        
        self.state_proj = torch.nn.Linear(14, 512).cuda() if torch.cuda.is_available() else torch.nn.Linear(14, 512)
        print("✅ Object tokenizer ready!")
    
    def detect_and_embed(self, image):
        """Detect objects + embed crops"""
        results = self.yolo(image)
        object_tokens = []
        
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = box.conf[0]
                if conf > 0.5:  # High confidence
                    bbox = box.xyxy[0].cpu().numpy()
                    obj_name = r.names[cls]
                    
                    # Crop + CLIP embed
                    crop = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                    crop_tensor = self.clip_proc(crop).unsqueeze(0)
                    emb = self.clip_model.encode_image(crop_tensor)
                    
                    object_tokens.append({
                        'name': obj_name, 
                        'bbox': bbox,
                        'embedding': emb
                    })
        
        return object_tokens
    
    def forward(self, image, language, joint_state=None):
        """Full VLA tokenization"""
        # Object tokens
        objects = self.detect_and_embed(image)
        obj_tokens = torch.cat([obj['embedding'] for obj in objects], dim=0)
        
        # Language tokens
        lang_tokens = self.clip_model.encode_text(self.tokenizer(language).input_ids)
        
        # Joint tokens
        if joint_state is None:
            joint_state = np.zeros(14)
        joint_tokens = self.state_proj(torch.tensor(joint_state).float().unsqueeze(0))
        
        # Fuse all
        all_tokens = torch.cat([obj_tokens, lang_tokens.unsqueeze(0), joint_tokens], dim=0)
        return all_tokens, objects

tokenizer = ObjectVLATokenizer()