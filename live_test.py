# this is test script to test whole pipeline, yolo -> object tokenizer -> language command -> action output

import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import gc
import os
import time
import keyboard  # pip install keyboard

# E: Drive caching
os.environ['YOLO_CACHE_DIR'] = 'E:/yolo_cache'
os.environ['HF_HOME'] = 'E:/huggingface_cache'
os.environ['TORCH_HOME'] = 'E:/torch_cache'

print("Press ESC to quit")

class LowMemVLATokenizer:
    def __init__(self):
        print("Loading YOLOv8n...")
        self.yolo = YOLO("yolov8n.pt")
        self.state_proj = torch.nn.Linear(14, 256)
        
        self.obj_embeddings = {
            'person': torch.randn(256),
            'cup': torch.randn(256), 
            'book': torch.randn(256),
            'bottle': torch.randn(256),
            'default': torch.randn(256)
        }
        print("✅ YOLO tokenizer ready!")
    
    def detect_objects(self, image):
        results = self.yolo(image, verbose=False, device='cpu')
        objects = []
        
        for r in results:
            for box in r.boxes:
                cls = int(box.cls)
                conf = box.conf.item()
                if conf > 0.3:
                    bbox = box.xyxy[0].cpu().numpy()
                    obj_name = r.names[cls]
                    emb = self.obj_embeddings.get(obj_name, self.obj_embeddings['default'])
                    
                    objects.append({
                        'name': obj_name,
                        'confidence': conf,
                        'bbox': bbox,
                        'embedding': emb
                    })
        gc.collect()
        return objects
    
    def forward(self, image, language, joint_state=None):
        objects = self.detect_objects(image)
        
        if objects:
            obj_tokens = torch.stack([obj['embedding'] for obj in objects]).mean(0).unsqueeze(0)
        else:
            obj_tokens = torch.zeros(1, 256)
        
        lang_token = torch.randn(1, 256)
        joint_state = np.zeros(14) if joint_state is None else joint_state
        joint_token = self.state_proj(torch.tensor(joint_state).float()).unsqueeze(0)
        
        tokens = torch.cat([obj_tokens, lang_token, joint_token], dim=1)
        return tokens, objects

class TinyVLA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.action_head = torch.nn.Linear(768, 7)
    
    def forward(self, tokens):
        return self.action_head(tokens.mean(dim=0))

# INITIALIZE
torch.cuda.empty_cache() if torch.cuda.is_available() else None
tokenizer = LowMemVLATokenizer()
model = TinyVLA()

# Output directory
os.makedirs("E:/vla_detections", exist_ok=True)
frame_count = 0

print("Saving to: E:/vla_detections/")
print("Press ESC to stop")

# MAIN LOOP - NO OPENCV GUI
cap = cv2.VideoCapture(0)
commands = ["avoid perspon", "pick up book", ]

try:
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb).resize((320, 320))
        
        # VLA INFERENCE FOR ALL COMMANDS
        print(f"\n{'='*70}")
        print(f"Frame #{frame_count:06d}")
        
        for cmd in commands:
            tokens, objects = tokenizer.forward(pil_image, cmd)
            actions = model(tokens)
            
            obj_names = [f"{o['name']}({o['confidence']:.1f})" for o in objects]
            print(f"'{cmd}' → Objects: {obj_names} → Actions: {actions.detach().numpy().round(3)}")
        
        # SAVE EVERY 30 FRAMES (1/sec)
        if frame_count % 30 == 0:
            objects = tokenizer.detect_objects(pil_image)
            draw_frame = frame_bgr.copy()
            
            for obj in objects:
                x1, y1, x2, y2 = obj['bbox'].astype(int)
                cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(draw_frame, obj['name'], (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            filename = f"E:/vla_detections/vla_frame_{frame_count:06d}.jpg"
            cv2.imwrite(filename, draw_frame)
            
            print(f"\nSAVED: {filename}")
            print(f"{len(objects)} objects, {len(set([o['name'] for o in objects]))} unique classes")
        
        frame_count += 1
        
        # ESC TO QUIT (Keyboard library - no OpenCV)
        if keyboard.is_pressed('esc'):
            print("\nESC pressed - Stopping VLA...")
            break
            
        time.sleep(0.033)  # 30 FPS
        
except KeyboardInterrupt:
    print("\nCtrl+C - Stopping VLA...")

cap.release()
print(f"Processed: {frame_count} frames")
print(f"Output: E:/vla_detections/")

