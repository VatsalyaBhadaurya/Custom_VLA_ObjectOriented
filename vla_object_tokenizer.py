# LOW-MEM OBJECT VLA TOKENIZER - NO OpenCLIP (50MB total)
import os
import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import gc

# E: Drive caching
os.environ['YOLO_CACHE_DIR'] = 'E:/yolo_cache'
os.environ['HF_HOME'] = 'E:/huggingface_cache'
os.environ['TORCH_HOME'] = 'E:/torch_cache'

print("Object Aware VLA Tokenizer")

class ObjectVLATokenizer:
    def __init__(self):
        print("Loading YOLOv8n...")
        self.yolo = YOLO("yolov8n.pt")
        
        # NO OpenCLIP - Color + position embeddings 
        self.device = torch.device('cpu')
        self.state_proj = torch.nn.Linear(14, 128)  

        # Learnable object embeddings
        self.obj_embeddings = torch.nn.Embedding(80, 128)  # COCO 80 classes
        self.pos_embedding = torch.nn.Linear(4, 128)  # bbox → embedding
        
    
    def detect_and_embed(self, image):
        """YOLO + Lightweight embeddings"""
        results = self.yolo(image, verbose=False, device='cpu')
        object_tokens = []
        
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls = int(box.cls.item())
                    conf = box.conf.item()
                    if conf > 0.4:
                        bbox = box.xyxy[0].cpu().numpy()
                        obj_name = r.names[cls]
                        
                        # 1. Class embedding
                        cls_emb = self.obj_embeddings(torch.tensor([cls]))
                        
                        # 2. Position embedding  
                        pos_emb = self.pos_embedding(torch.tensor(bbox))
                        
                        # 3. Fuse
                        emb = cls_emb + pos_emb
                        
                        object_tokens.append({
                            'name': obj_name,
                            'bbox': bbox,
                            'confidence': conf,
                            'embedding': emb.squeeze(0)
                        })
        
        gc.collect()
        return object_tokens
    
    def forward(self, image, language, joint_state=None):
        """Full VLA tokenization"""
        # Object tokens
        objects = self.detect_and_embed(image)
        if objects:
            obj_tokens = torch.stack([obj['embedding'] for obj in objects])
        else:
            obj_tokens = torch.zeros(1, 128)
        
        # Language (mock - 128D)
        lang_tokens = torch.randn(1, 128)
        
        # Joint tokens
        if joint_state is None:
            joint_state = np.zeros(14)
        joint_tokens = self.state_proj(torch.tensor(joint_state).float())
        
        # Fuse: [N_obj,128] + [1,128] + [1,128]
        all_tokens = torch.cat([obj_tokens, lang_tokens, joint_tokens.unsqueeze(0)], dim=0)
        return all_tokens, objects

# TEST IT
if __name__ == "__main__":
    tokenizer = ObjectVLATokenizer()
    
    # Test image (webcam or file)
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Test forward pass
        tokens, objects = tokenizer.forward(pil_image, "pick red object")
        
        print(f"SUCCESS!")
        print(f"Detected {len(objects)} objects:")
        for obj in objects:
            print(f"  - {obj['name']}({obj['confidence']:.2f}) @ {obj['bbox']}")
        print(f"Token shape: {tokens.shape}")
    else:
        print("Webcam test failed - using dummy")
        dummy_img = Image.new('RGB', (320, 320), color='red')
        tokens, objects = tokenizer.forward(dummy_img, "test")
        print(f"Dummy test: {tokens.shape}")