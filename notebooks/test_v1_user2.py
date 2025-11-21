import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json
import sys
import os

# ==========================================
# CONFIGURATION FOR USER 2
# ==========================================
# User 2's Dataset (The one you cleaned/filtered)
DATA_DIR = '../data/NewPlantDiseases_Filtered/valid' 

# User 1's Model Weights (ResNet18)
MODEL_PATH = '../models/model_v1.pth'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import User 1's Architecture
sys.path.append(os.path.abspath('..'))
try:
    from models.model_v1 import get_model
except ImportError:
    print("❌ Critical Error: models/model_v1.py not found.")
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

print(f"Testing User 1's Model on {len(test_dataset)} images (User 2 Data).")

# ==========================================
# 2. LOAD MODEL (Robust / Partial Load)
# ==========================================
# We initialize the model with OUR number of classes (38)
model = get_model(num_classes=len(test_dataset.classes))
model = model.to(DEVICE)

try:
    print(f"Loading weights from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model_state = model.state_dict()
    
    # Filter out layers that don't match in size (e.g., the final 15 vs 38 class layer)
    matched_weights = {k: v for k, v in checkpoint.items() if k in model_state and v.size() == model_state[k].size()}
    
    # Load what we can (The "Brain" / Backbone)
    model.load_state_dict(matched_weights, strict=False)
    
    dropped_keys = [k for k in checkpoint.keys() if k not in matched_weights]
    if dropped_keys:
        print(f"⚠️ Mismatch Detected! Dropped layers: {dropped_keys}")
        print("   (This is expected if Class counts differ. Final layer is now random.)")
    else:
        print("✅ Full model loaded successfully (Classes matched perfectly).")

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
    "tested_by": "User 2",
    "model": "model_v1 (ResNet18)",
    "dataset": "New Plant Diseases",
    "accuracy": accuracy,
    "notes": "Partial load performed due to class mismatch." if dropped_keys else "Full load success."
}

with open('../results/test_v1_user2.json', 'w') as f:
    json.dump(results, f, indent=4)
print("Saved results/test_v1_user2.json")