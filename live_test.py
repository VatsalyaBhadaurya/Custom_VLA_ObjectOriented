# COMPLETE VLA OBJECT RECOGNITION - FIXED SYNTAX
import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import open_clip
import time

print("🚀 VLA Object Recognition Starting...")

class ObjectVLATokenizer:
    def __init__(self):
        print("Loading YOLO...")
        self.yolo = YOLO("yolov8n.pt")
        
        print("Loading CLIP...")
        self.clip_model, _, self.clip_proc = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.clip_model.eval()
        
        self.state_proj = torch.nn.Linear(14, 512)
        print("✅ Tokenizer ready!")
    
    def detect_objects(self, image):
        results = self.yolo(image)
        objects = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls)
                conf = box.conf.item()
                if conf > 0.4:
                    bbox = box.xyxy[0].cpu().numpy()
                    obj_name = r.names[cls]
                    
                    crop = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                    crop_t = self.clip_proc(crop).unsqueeze(0)
                    
                    with torch.no_grad():
                        emb = self.clip_model.encode_image(crop_t)
                    
                    objects.append({
                        'name': obj_name,
                        'confidence': conf,
                        'bbox': bbox,
                        'embedding': emb.squeeze(0)
                    })
        return objects
    
    def forward(self, image, language, joint_state=None):
        objects = self.detect_objects(image)
        obj_tokens = torch.stack([obj['embedding'] for obj in objects]) if objects else torch.zeros(1, 512)
        
        lang_token = torch.randn(1, 512)
        if joint_state is None:
            joint_state = np.zeros(14)
        joint_token = self.state_proj(torch.tensor(joint_state).float())
        
        tokens = torch.cat([obj_tokens.mean(0).unsqueeze(0), lang_token, joint_token], dim=1)
        return tokens, objects

class MockVLA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.action_head = torch.nn.Linear(1536, 7)
    
    def forward(self, tokens):
        fused = tokens.mean(dim=0)
        return self.action_head(fused)

# INITIALIZE
tokenizer = ObjectVLATokenizer()
model = MockVLA()
print("✅ VLA ready!")

# LIVETEST
cap = cv2.VideoCapture(0)
commands = ["pick red object", "avoid person", "grasp cup"]

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break
    
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    for cmd in commands:
        tokens, objects = tokenizer.forward(pil_image, cmd)
        actions = model(tokens)
        
        # FIXED SYNTAX: No f-string nesting
        obj_list = []
        for o in objects:
            obj_list.append(f"{o['name']}({o['confidence']:.1f})")
        
        print(f"\n🤖 '{cmd}'")
        print(f"   Objects: {obj_list}")
        print(f"   Actions: {actions.detach().numpy().round(3)}")
    
    # Draw detections
    objects = tokenizer.detect_objects(pil_image)  # Re-detect for drawing
    for obj in objects:
        x1, y1, x2, y2 = obj['bbox'].astype(int)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_bgr, obj['name'], (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imshow("VLA Objects Live", frame_bgr)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Test complete!")
