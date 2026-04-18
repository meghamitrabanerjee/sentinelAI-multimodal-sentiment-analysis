# 🌍 SentinelAI: Multimodal Opinion Intelligence

An end-to-end deep learning architecture that evaluates the true contextual sentiment of geopolitical discourse by fusing Computer Vision and Natural Language Processing. 

This project addresses the critical failure points of text-only sentiment models (e.g., VADER, RoBERTa) when analyzing complex multimodal internet data, specifically detecting cross-modal contradictions (e.g., sarcastic text paired with destructive imagery).

## 🧠 System Architecture

The model utilizes a **Mid-Level Concatenation Architecture** to process and fuse modalities before final classification.

* **Textual Feature Extraction:** `DistilBERT` (Outputs 768-d semantic tensor)
* **Visual Feature Extraction:** `ResNet-50` (Classification head removed; outputs 2048-d visual tensor)
* **Dimensionality Reduction & Fusion:** Both tensors are passed through independent `nn.Linear` layers, compressing them to 256 dimensions to prevent visual modality dominance. They are concatenated into a 512-d fused tensor, passed through a 128-node hidden layer optimized for GPU computation, and classified into 3 states.

## 🚀 Key Engineering & Research Challenges

1. **Custom Data Ingestion:** Engineered a headless browser automation pipeline using Playwright to scrape dynamically rendered single-page applications (Twitter/X React DOM), compiling a highly specific dataset of geopolitical conflicts.
2. **Pseudo-Labeling & Data Distribution:** Utilized VADER lexicon scoring to auto-label the dataset. Identified a severe real-world class imbalance inherent to the war domain: *Negative (71.5%), Positive (26.8%), Neutral (1.6%)*.
3. **Imbalance Optimization:** Mitigated the extreme class imbalance by calculating balanced class weights and feeding them into the PyTorch `CrossEntropyLoss` criterion, forcing the network to learn minority class boundaries.
4. **Defeating Overfitting:** Implemented `Dropout(0.3)` layers and L2 Regularization (`weight_decay=1e-4`) within the Adam optimizer to ensure generalization on subjective opinion data.

## 📊 Empirical Evaluation

The architecture was evaluated on an unseen 20% validation split, establishing a robust baseline for custom multimodal fusion:

* **Overall Accuracy:** 72.58%
* **Weighted F1-Score:** 0.7314
* **Network Calibration:** Mean confidence of 86.9% on correct predictions vs. 78.9% on incorrect predictions.
* **Diagnostic Insight:** The "Neutral" class yielded an F1-score of 0.00 due to extreme data starvation (only 3 validation samples), proving the necessity of massive, balanced datasets for highly polarized domains.

## 🔍 Explainable AI (XAI) Dashboard

The `.pth` weights are deployed via a lazy-loaded Streamlit frontend. To resolve the "black box" nature of neural networks, the dashboard features a dynamic XAI engine. It cross-references linguistic anchors with PyTorch confidence distributions to diagnose whether the text or the visual branch dictated the final prediction.

## 🛠️ How to Run Locally

1. Install dependencies: `pip install -r requirements.txt`
2. Extract baseline features: `python extract_features.py`
3. Train the network: `python train_model.py`
4. Launch the XAI Dashboard: `python -m streamlit run app.py`

*(Note: Due to GitHub's file size limits, the `dataset/` directory and `.pth` model weights are hosted externally. Run the extraction pipeline to generate local tensors).*
