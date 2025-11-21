import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np
import os
from PIL import Image
from tqdm import tqdm  # <--- NEW IMPORT

# ==========================================
# 1. CONFIGURATION (User 2)
# ==========================================
# We need path 1 JUST to read class names (Metadata), not images.
PATH_TO_DATASET_1_REF = '../data/PlantVillage' 

# We need path 2 for the actual training images.
PATH_TO_DATASET_2_TRAIN = '../data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train'

BATCH_SIZE = 32
NUM_EPOCHS = 3
K_FOLDS = 3
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[User 2] Running MobileNetV2 on: {device}")

# ==========================================
# 2. GLOBAL CLASS MAPPING
# ==========================================
def get_global_class_map(paths):
    global_classes = set()
    for path in paths:
        if os.path.exists(path):
            classes = [d.name for d in os.scandir(path) if d.is_dir()]
            global_classes.update(classes)
    
    sorted_classes = sorted(list(global_classes))
    class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted_classes)}
    return sorted_classes, class_to_idx

ALL_CLASSES, GLOBAL_MAP = get_global_class_map([PATH_TO_DATASET_1_REF, PATH_TO_DATASET_2_TRAIN])

print(f"Global Class Map created with {len(ALL_CLASSES)} total classes.")

# ==========================================
# 3. LOCAL DATASET LOADER
# ==========================================
class LocalDataset(Dataset):
    def __init__(self, root_dir, global_map, transform=None):
        self.root_dir = root_dir
        self.global_map = global_map
        self.transform = transform
        self.samples = []
        
        if not os.path.exists(root_dir):
            raise RuntimeError(f"Dataset 2 path not found: {root_dir}")
            
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir): continue
            
            if class_name in self.global_map:
                global_idx = self.global_map[class_name]
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(class_dir, img_file), global_idx))
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# ==========================================
# 4. PREPARE DATA
# ==========================================
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

local_dataset = LocalDataset(PATH_TO_DATASET_2_TRAIN, GLOBAL_MAP, transform=data_transforms)
print(f"User 2 Loaded {len(local_dataset)} images from local dataset.")

# ==========================================
# 5. MODEL SETUP
# ==========================================
def get_mobilenet_model(num_total_classes):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    for param in model.features.parameters():
        param.requires_grad = False
    
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_ftrs, num_total_classes)
    )
    return model.to(device)

# ==========================================
# 6. TRAINING LOOP WITH TQDM
# ==========================================
kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
results = {}

print(f"\n[User 2] Starting Local Training (Classes aligned globally)...")

for fold, (train_ids, val_ids) in enumerate(kfold.split(local_dataset)):
    print(f"\n{'='*20} FOLD {fold + 1}/{K_FOLDS} {'='*20}")
    
    train_sub = Subset(local_dataset, train_ids)
    val_sub = Subset(local_dataset, val_ids)
    
    trainloader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False)
    
    model = get_mobilenet_model(num_total_classes=len(ALL_CLASSES))
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # --- EPOCH LOOP ---
    for epoch in range(NUM_EPOCHS):
        model.train()
        
        # WRAP TRAIN LOADER WITH TQDM
        # This creates the progress bar
        loop = tqdm(trainloader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}', leave=True)
        
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Update the progress bar with the current loss value
            loop.set_postfix(loss=loss.item())
            
    # --- EVALUATION (No TQDM needed here usually, keeping it clean) ---
    print(f"Validating Fold {fold+1}...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    acc = 100 * correct / total
    print(f"Fold {fold+1} Accuracy: {acc:.2f}%")
    results[fold] = acc

print(f"\n[User 2] FINAL AVERAGE ACCURACY: {np.mean(list(results.values())):.2f}%")
