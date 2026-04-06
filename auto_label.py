import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def auto_label_dataset():
    print("--- Starting Phase 3.5: VADER Auto-Labeling ---")
    
    # 1. Load the clean text we generated in Phase 2
    input_path = "dataset/master_dataset_cleaned.json"
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        
    print(f"Loaded {len(dataset)} clean multimodal pairs.")
    
    analyzer = SentimentIntensityAnalyzer()
    labeled_data = []
    
    # Trackers to see our data distribution
    counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    
    # 2. Analyze each text snippet
    for item in dataset:
        text = item['clean_text']
        
        # VADER gives a 'compound' score from -1 (Extremely Negative) to +1 (Extremely Positive)
        scores = analyzer.polarity_scores(text)
        compound = scores['compound']
        
        # 3. Categorize based on standard VADER thresholds
        if compound >= 0.05:
            label = "Positive"
        elif compound <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"
            
        counts[label] += 1
        
        # Create a new dictionary with the label included
        labeled_item = {
            "id": item['id'],
            "text": text,          # Keeping the clean text for reference
            "image_path": item['image_path'],
            "sentiment": label,    # OUR NEW GROUND TRUTH!
            "vader_score": compound # Keeping the raw score just in case
        }
        labeled_data.append(labeled_item)
        
    # 4. Save the new labeled dataset
    output_path = "dataset/master_dataset_labeled.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(labeled_data, f, indent=4, ensure_ascii=False)
        
    print("\n--- Labeling Complete! ---")
    print("Sentiment Distribution:")
    print(f"  Negative : {counts['Negative']}")
    print(f"  Neutral  : {counts['Neutral']}")
    print(f"  Positive : {counts['Positive']}")
    print(f"\nSaved labeled dataset to: {output_path}")

if __name__ == "__main__":
    auto_label_dataset()