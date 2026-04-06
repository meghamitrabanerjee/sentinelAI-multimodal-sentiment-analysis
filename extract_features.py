import os
import json
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import DistilBertTokenizer, DistilBertModel
from torchvision import models, transforms

def setup_models():
    print("Loading models into memory (this takes a few seconds)...")
    
    # Force CPU usage
    device = torch.device("cpu")
    
    # 1. Setup DistilBERT for Text
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    text_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    text_model.to(device)
    text_model.eval() # Set to evaluation mode
    
    # 2. Setup ResNet-50 for Images
    # We use the default pre-trained weights
    weights = models.ResNet50_Weights.DEFAULT
    img_model = models.resnet50(weights=weights)
    
    # IMPORTANT: We don't want ResNet to classify the image into 1000 categories.
    # We want the raw feature vector BEFORE the classification layer.
    # So, we remove the final Fully Connected (fc) layer.
    img_model = torch.nn.Sequential(*(list(img_model.children())[:-1]))
    img_model.to(device)
    img_model.eval()
    
    # ResNet requires specific image transformations (resize to 224x224, normalize colors)
    img_transforms = weights.transforms()
    
    return device, tokenizer, text_model, img_transforms, img_model

def extract_features():
    print("--- Starting Phase 3: Feature Extraction ---")
    
    device, tokenizer, text_model, img_transforms, img_model = setup_models()
    
    # Load your clean master dataset
    json_path = "dataset/master_dataset_cleaned.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        
    print(f"Loaded {len(dataset)} items. Beginning extraction on CPU...")
    
    extracted_data = {}
    
    # torch.no_grad() is crucial for CPU. It tells PyTorch not to track memory for training.
    with torch.no_grad():
        for item in tqdm(dataset, desc="Processing Data"):
            uid = item['id']
            text = item['clean_text']
            img_path = item['image_path']
            
            try:
                # --- 1. PROCESS TEXT ---
                # Tokenize and pad/truncate to a standard length
                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=128)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Get DistilBERT output. We take the hidden state of the [CLS] token (index 0)
                text_outputs = text_model(**inputs)
                text_feature = text_outputs.last_hidden_state[:, 0, :].squeeze() # Shape: (768,)
                
                # --- 2. PROCESS IMAGE ---
                # Open image, convert to standard RGB (fixes issues with grayscale or PNG transparency)
                image = Image.open(img_path).convert('RGB')
                
                # Apply resizing and normalization
                img_tensor = img_transforms(image).unsqueeze(0).to(device)
                
                # Get ResNet output and flatten it
                img_outputs = img_model(img_tensor)
                img_feature = img_outputs.squeeze() # Shape: (2048,)
                
                # --- 3. SAVE TO DICTIONARY ---
                extracted_data[uid] = {
                    "text_feature": text_feature.cpu(),
                    "image_feature": img_feature.cpu(),
                    "source": item['source']
                }
                
            except Exception as e:
                print(f"\nSkipping ID {uid} due to error: {e}")

    # Create a directory to store our AI features
    os.makedirs("dataset/features", exist_ok=True)
    save_path = "dataset/features/extracted_features.pt"
    
    # Save the dictionary as a PyTorch .pt file
    torch.save(extracted_data, save_path)
    print(f"\n--- Extraction Complete! ---")
    print(f"Successfully extracted and saved features for {len(extracted_data)} items.")
    print(f"Saved to: {save_path}")

if __name__ == "__main__":
    extract_features()