# 🌍 SentinelAI - Multimodal Sentiment Analysis

An end-to-end Machine Learning architecture that analyzes the true sentiment of geopolitical discourse by fusing Computer Vision and Natural Language Processing.

## 🧠 The Architecture
* **Text Extraction:** DistilBERT (768-d tensor)
* **Visual Extraction:** ResNet-50 (2048-d tensor)
* **Fusion Strategy:** Mid-level tensor concatenation via a custom PyTorch Neural Network.
* **Explainability (XAI):** VADER-assisted dynamic logic breakdown.

## 🚀 Why this approach?
We identified severe limitations in text-only models (like VADER) when analyzing war and conflict data. Text-only models fail to understand sarcasm, cross-modal contradictions, or visual context (e.g., a tweet with positive text but an image of a destroyed building). This pipeline solves that by forcing the modalities to compete in a specialized fusion layer.

## 🛠️ How to Run
1. Install requirements: `pip install -r requirements.txt`
2. Extract features: `python extract_features.py`
3. Train the model: `python train_model.py`
4. Launch the Dashboard: `python -m streamlit run app.py`

*(Note: The `dataset` folder and `.pth` model weights are excluded from this repo due to size constraints. You must run the data engineering pipeline to generate your own weights.)*
