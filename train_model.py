import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

# 1. DEFINE THE MULTIMODAL NEURAL NETWORK
class MultimodalFusionNet(nn.Module):
    def __init__(self):
        super(MultimodalFusionNet, self).__init__()
        
        # Text processing branch
        self.text_network = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3) # Dropout prevents overfitting (memorizing the data)
        )
        
        # Image processing branch
        self.image_network = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Fused classification branch (256 + 256 = 512)
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3) # 3 output classes: Negative, Neutral, Positive
        )
        
    def forward(self, text_features, image_features):
        text_out = self.text_network(text_features)
        image_out = self.image_network(image_features)
        
        # Combine them side-by-side
        fused = torch.cat((text_out, image_out), dim=1)
        
        # Predict
        output = self.classifier(fused)
        return output

# 2. CREATE A CUSTOM DATASET HANDLER
class TwitterDataset(Dataset):
    def __init__(self, data_list, feature_dict, label_map):
        self.data_list = data_list
        self.feature_dict = feature_dict
        self.label_map = label_map
        
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx):
        item = self.data_list[idx]
        uid = item['id']
        label_str = item['sentiment']
        
        # Get the pre-extracted tensors
        text_feat = self.feature_dict[uid]['text_feature']
        image_feat = self.feature_dict[uid]['image_feature']
        label_idx = self.label_map[label_str]
        
        return text_feat, image_feat, torch.tensor(label_idx, dtype=torch.long)

def train_and_evaluate():
    print("--- Starting Phase 4: Multimodal AI Training (Optimized) ---")
    
    # 1. Load the Labels and Features
    with open("dataset/master_dataset_labeled.json", 'r', encoding='utf-8') as f:
        labeled_data = json.load(f)
        
    features = torch.load("dataset/features/extracted_features.pt", weights_only=False)
    
    # Map text labels to numbers for PyTorch
    label_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
    
    # 2. Split into Train (80%) and Test (20%)
    train_data, test_data = train_test_split(labeled_data, test_size=0.2, random_state=42)
    print(f"Training on {len(train_data)} items, Testing on {len(test_data)} items.")
    
    # 3. Handle the Class Imbalance!
    train_labels = [label_map[item['sentiment']] for item in train_data]
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(train_labels), 
        y=train_labels
    )
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print(f"Applying Class Weights to fix imbalance: {class_weights}")
    
    # 4. Create DataLoaders
    train_dataset = TwitterDataset(train_data, features, label_map)
    test_dataset = TwitterDataset(test_data, features, label_map)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 5. Initialize Model, Loss, and Optimizer
    model = MultimodalFusionNet()
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    
    # OPTIMIZATION 1: Added weight_decay (L2 Regularization) to heavily penalize memorization
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # OPTIMIZATION 2: Reduced epochs to stop the model before it overfits
    epochs = 8 
    
    # --- TRAINING LOOP ---
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for text_batch, img_batch, label_batch in train_loader:
            optimizer.zero_grad() 
            
            # Forward pass
            outputs = model(text_batch, img_batch)
            loss = criterion(outputs, label_batch)
            
            # Backward pass (Learn)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {running_loss/len(train_loader):.4f}")
        
    # --- EVALUATION LOOP ---
    print("\n--- Final Exam (Evaluation on Test Set) ---")
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for text_batch, img_batch, label_batch in test_loader:
            outputs = model(text_batch, img_batch)
            
            # Find the highest probability class
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.numpy())
            all_targets.extend(label_batch.numpy())
            
    # Print the final report card
    print(classification_report(
        all_targets, 
        all_preds, 
        target_names=["Negative", "Neutral", "Positive"],
        zero_division=0
    ))
    
    # Save the newly trained, robust brain!
    torch.save(model.state_dict(), "dataset/features/multimodal_sentiment_model.pth")
    print("Robust Model successfully saved as 'multimodal_sentiment_model.pth'!")

if __name__ == "__main__":
    train_and_evaluate()