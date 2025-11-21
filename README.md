# collaborative_cnn_team06

Team 06 - Collaborative CNN Development

Team Members:

User 1 (Base Repo): [User 1 Name] - Dataset: Plant Village

User 2 (Fork): [User 2 Name] - Dataset: Kaggle New Plant Diseases

Objective:
Train and cross-test CNN models on distinct plant disease datasets to simulate a federated learning scenario with domain shift.

Models & Results:

Model V1 (ResNet50): Trained on Plant Village. High native accuracy (88.5%), but failed cross-domain transfer (30%).

Model V2 (MobileNetV2): Trained on Kaggle. High native accuracy (92.1%) and successful transfer (81.4%).

Key Findings:
Training on the larger, diverse dataset (User 2) produced a significantly more robust model capable of handling domain shift. A custom global class mapping strategy ensured compatibility across datasets.

Usage:

Install requirements: pip install torch torchvision scikit-learn tqdm pillow

Update dataset paths in train_user2.py.

Run: python train_user2.py

Full Report: See report.md for detailed metrics and analysis.
