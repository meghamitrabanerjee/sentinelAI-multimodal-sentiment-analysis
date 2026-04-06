import os
import glob
import json
import re
import pandas as pd

def clean_tweet_text(text):
    """Cleans raw tweet text for DistilBERT."""
    if not isinstance(text, str):
        return ""
    
    # Remove URLs (http:// or https://)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove @mentions
    text = re.sub(r'\@\w+', '', text)
    
    # Remove the word 'RT' (Retweet tags)
    text = re.sub(r'\bRT\b', '', text)
    
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def validate_image(image_path):
    """Checks if the image exists and is not corrupted (0 bytes)."""
    if os.path.exists(image_path):
        # Check if file size is greater than 0 bytes
        if os.path.getsize(image_path) > 0:
            return True
    return False

def run_preprocessing():
    print("--- Starting Phase 2: Data Preprocessing ---")
    
    # 1. Find all JSON batch files in the dataset folder
    json_files = glob.glob("dataset/twitter_dataset_batch*.json")
    
    if not json_files:
        print("Error: No JSON batch files found in the 'dataset' folder.")
        return

    print(f"Found {len(json_files)} batch files. Merging...")
    
    all_data = []
    
    # 2. Merge all data into one list
    for file in json_files:
        with open(file, 'r', encoding='utf-8') as f:
            try:
                batch_data = json.load(f)
                all_data.extend(batch_data)
            except Exception as e:
                print(f"Error reading {file}: {e}")

    # Convert to a Pandas DataFrame for easy manipulation
    df = pd.DataFrame(all_data)
    print(f"Total raw multimodal pairs collected: {len(df)}")
    
    # 3. Deduplicate based on exact text
    # (Since we generated new UUIDs each run, the IDs are unique, but the text might be the same)
    df.drop_duplicates(subset=['text'], keep='first', inplace=True)
    print(f"Pairs remaining after dropping duplicates: {len(df)}")
    
    # 4. Clean the Text
    print("Cleaning text (removing URLs, mentions, extra spaces)...")
    df['clean_text'] = df['text'].apply(clean_tweet_text)
    
    # Drop rows where the clean text ended up being empty
    df = df[df['clean_text'].str.len() > 5] 
    
    # 5. Validate Images
    print("Validating image files...")
    df['image_valid'] = df['image_path'].apply(validate_image)
    
    # Keep only rows with valid images
    df = df[df['image_valid'] == True]
    
    # Drop the temporary validation column
    df = df.drop(columns=['image_valid'])
    
    print(f"\n--- Preprocessing Complete! ---")
    print(f"Final usable, clean multimodal pairs: {len(df)}")
    
    # 6. Export the final Master Dataset
    master_json_path = "dataset/master_dataset_cleaned.json"
    master_csv_path = "dataset/master_dataset_cleaned.csv"
    
    # Save as JSON for our Python pipeline
    df.to_json(master_json_path, orient='records', indent=4, force_ascii=False)
    
    # Save as CSV so you can easily open it in Excel/Numbers to look at your data
    df.to_csv(master_csv_path, index=False, encoding='utf-8')
    
    print(f"Saved master JSON to: {master_json_path}")
    print(f"Saved master CSV to: {master_csv_path}")

if __name__ == "__main__":
    run_preprocessing()