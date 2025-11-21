import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json
import sys
import os

# ==========================================
# CONFIGURATION FOR USER 1
# ==========================================
# User 1's Dataset
DATA_DIR = '../data/PlantVillage' 

# User 2's Model Weights (MobileNetV2)
MODEL_PATH = '../models/model_v2.pth'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import User 2's Architecture
sys.path.append(os.path.abspath('..'))
try:
    from models.model_v2 import get_model_v2
except ImportError:
    print("❌ Critical Error: models/model_v2.py not found. Did you pull User 2's code?")
    sys.exit(1)

# ==========================================
# 1. SETUP DATA
# ==========================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if not os.path.exists(DATA_DIR):
    print(f"❌ Dataset not found at {DATA_DIR}")
    sys.exit(1)

test_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Testing User 2's Model on {len(test_dataset)} images (User 1 Data).")

# ==========================================
# 2. LOAD MODEL (Robust / Partial Load)
# ==========================================
# Initialize MobileNetV2 with OUR class count
model = get_model_v2(num_classes=len(test_dataset.classes))
model = model.to(DEVICE)

try:
    print(f"Loading weights from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model_state = model.state_dict()
    
    # Filter matching layers
    matched_weights = {k: v for k, v in checkpoint.items() if k in model_state and v.size() == model_state[k].size()}
    
    model.load_state_dict(matched_weights, strict=False)
    
    dropped = [k for k in checkpoint.keys() if k not in matched_weights]
    if dropped:
        print(f"⚠️ Mismatch Detected! Dropped layers: {dropped}")
    else:
        print("✅ Full model loaded successfully.")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# ==========================================
# 3. RUN TEST
# ==========================================
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"\nFinal Accuracy: {accuracy:.2%}")

# ==========================================
# 4. SAVE RESULTS
# ==========================================
results = {
    "tested_by": "User 1",
    "model": "model_v2 (MobileNetV2)",
    "dataset": "PlantVillage (User 1)",
    "accuracy": accuracy,
    "notes": "Tested User 2's improved architecture."
}

with open('../results/test_v2_user1.json', 'w') as f:
    json.dump(results, f, indent=4)
print("Saved results/test_v2_user1.json")