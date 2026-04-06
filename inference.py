import torch
import torch.nn as nn
from PIL import Image
from transformers import DistilBertTokenizer, DistilBertModel
from torchvision import models, transforms
import torch.nn.functional as F

# 1. THE NEURAL NETWORK BLUEPRINT (Must match training exactly)
class MultimodalFusionNet(nn.Module):
    def __init__(self):
        super(MultimodalFusionNet, self).__init__()
        self.text_network = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3))
        self.image_network = nn.Sequential(nn.Linear(2048, 256), nn.ReLU(), nn.Dropout(0.3))
        self.classifier = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 3)
        )
        
    def forward(self, text_features, image_features):
        text_out = self.text_network(text_features)
        image_out = self.image_network(image_features)
        fused = torch.cat((text_out, image_out), dim=1)
        return self.classifier(fused)

class SentimentPredictor:
    def __init__(self, model_path="dataset/features/multimodal_sentiment_model.pth"):
        print("Loading AI Brain and Feature Extractors...")
        self.device = torch.device("cpu")
        
        # Load the trained Fusion Model
        self.model = MultimodalFusionNet()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        
        # Load Text Extractor (DistilBERT)
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.text_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.text_model.eval()
        
        # Load Image Extractor (ResNet50)
        weights = models.ResNet50_Weights.DEFAULT
        resnet = models.resnet50(weights=weights)
        self.img_model = torch.nn.Sequential(*(list(resnet.children())[:-1]))
        self.img_model.eval()
        self.img_transforms = weights.transforms()
        
        self.labels = ["Negative", "Neutral", "Positive"]
        print("System Ready!\n" + "-"*30)

    def predict(self, text, image_path):
        try:
            with torch.no_grad():
                # 1. Process Text
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=128)
                text_outputs = self.text_model(**inputs)
                text_feat = text_outputs.last_hidden_state[:, 0, :].squeeze().unsqueeze(0) # Shape (1, 768)
                
                # 2. Process Image
                image = Image.open(image_path).convert('RGB')
                img_tensor = self.img_transforms(image).unsqueeze(0)
                img_feat = self.img_model(img_tensor).squeeze().unsqueeze(0) # Shape (1, 2048)
                
                # 3. Fusion Prediction
                output = self.model(text_feat, img_feat)
                
                # 4. Get Probabilities (Softmax converts raw output to percentages)
                probabilities = F.softmax(output, dim=1).squeeze().numpy()
                predicted_idx = torch.argmax(output).item()
                
                print(f"TEXT INPUT: '{text}'")
                print(f"IMAGE INPUT: {image_path}")
                print(f"\nPREDICTION: ** {self.labels[predicted_idx].upper()} **")
                print("\nConfidence Breakdown:")
                for i, label in enumerate(self.labels):
                    print(f"  {label}: {probabilities[i]*100:.2f}%")
                    
        except Exception as e:
            print(f"Error during prediction: {e}")

if __name__ == "__main__":
    # Initialize the engine
    predictor = SentimentPredictor()
    
    # ==========================================
    # TEST YOUR AI HERE!
    # Write a fake tweet and provide an image path.
    # You can download a random image from Google to test it.
    # ==========================================
    
    test_text = "🚨 BREAKING ​Donald Trump: Tuesday will be both Power Plant Day and Bridge Day in Iran; something unique is going to happen!; ​Open that f***ing Strait of Hormuz or you will live in hell, watch! Praise be to God."  
    
    # Pick an image from your dataset folder, or download a new one and put the path here!
    test_image = "dataset/twitter_images/donald_trump.jpg" # <--- UPDATE THIS PATH
    
    predictor.predict(test_text, test_image)