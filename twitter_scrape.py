import os
import time
import uuid
import json
import requests
from datetime import datetime
from playwright.sync_api import sync_playwright

# Create folder for images
os.makedirs("dataset/twitter_images", exist_ok=True)

def download_twitter_image(image_url, image_id):
    """Downloads the highest quality version of the Twitter image."""
    try:
        clean_url = image_url.split('?')[0] + '?format=jpg&name=large'
        response = requests.get(clean_url, stream=True, timeout=10)

        if response.status_code == 200:
            filepath = f"dataset/twitter_images/{image_id}.jpg"
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return filepath
    except Exception as e:
        print(f"Image download failed for {image_id}: {e}")
    return None

def scrape_twitter(search_query, max_tweets=100):
    scraped_data = []

    with sync_playwright() as p:
        print("Connecting to Chrome on port 9222...")
        browser = p.chromium.connect_over_cdp("http://localhost:9222")
        context = browser.contexts[0]

        if context.pages:
            page = context.pages[0]
        else:
            page = context.new_page()

        search_url = f"https://x.com/search?q={search_query}&src=typed_query&f=live"
        print(f"Opening: {search_url}")

        page.goto(search_url)
        page.wait_for_timeout(8000)

        try:
            page.wait_for_selector('article[data-testid="tweet"]', timeout=15000)
        except:
            print("Tweets not loading. Check your browser window.")
            return []

        processed_texts = set()
        attempts = 0
        max_empty_scrolls = 15

        while len(scraped_data) < max_tweets and attempts < max_empty_scrolls:
            tweets = page.query_selector_all('article[data-testid="tweet"]')

            if not tweets:
                attempts += 1
                time.sleep(3)
                continue

            found_new_tweet_in_this_scroll = False

            for tweet in tweets:
                if len(scraped_data) >= max_tweets:
                    break

                try:
                    # 1. Get Text
                    text_el = tweet.query_selector('[data-testid="tweetText"]')
                    text = text_el.inner_text() if text_el else ""

                    if not text or text in processed_texts:
                        continue

                    # 2. Get Image
                    img_el = tweet.query_selector('[data-testid="tweetPhoto"] img')
                    img_url = img_el.get_attribute("src") if img_el else None

                    # 3. Save if Multimodal
                    if text and img_url:
                        uid = str(uuid.uuid4())[:8]
                        img_path = download_twitter_image(img_url, uid)

                        if img_path:
                            scraped_data.append({
                                "id": uid,
                                "source": "Twitter",
                                "text": text.replace("\n", " "),
                                "image_path": img_path
                            })
                            processed_texts.add(text)
                            found_new_tweet_in_this_scroll = True
                            print(f"Saved [{len(scraped_data)}/{max_tweets}]: Tweet {uid}")

                except Exception as e:
                    pass 

            if found_new_tweet_in_this_scroll:
                attempts = 0
            else:
                attempts += 1

            # BULLETPROOF SCROLLING: Works even if the window is minimized or loses focus
            page.evaluate("window.scrollBy(0, 2500)")
            time.sleep(3.5)

    return scraped_data

if __name__ == "__main__":
    
    # ---------------------------------------------------------
    # BATCH CONFIGURATION
    # Change this number (1 through 20) before each run!
    BATCH_NUMBER = 20
    # ---------------------------------------------------------
    
    # Pre-written queries ensuring distinct, non-overlapping data
    batch_queries = {
        1: "war conflict geopolitics filter:media until:2026-04-05 since:2026-03-15",
        2: "war conflict geopolitics filter:media until:2026-03-14 since:2026-02-20",
        3: "middle east crisis filter:media until:2026-04-05 since:2026-02-01",
        4: "ukraine war frontline filter:media until:2026-04-05 since:2026-02-01",
        5: "military escalation news filter:media until:2026-04-05 since:2026-01-01",
        6: "global conflict updates filter:media until:2026-04-05 since:2026-01-01",
        7: "war zones reporting filter:media until:2026-04-05 since:2026-01-01",
        8: "armed conflict public reaction filter:media until:2026-04-05 since:2025-11-01",
        9: "international security crisis filter:media until:2026-04-05 since:2025-11-01",
        10: "humanitarian crisis war filter:media until:2026-04-05 since:2025-12-01",
        11: "refugees border conflict filter:media until:2026-04-05 since:2025-12-01",
        12: "civilian impact war filter:media until:2026-04-05 since:2025-11-01",
        13: "drone strikes military filter:media until:2026-04-05 since:2025-12-01",
        14: "cyber warfare infrastructure filter:media until:2026-04-05 since:2025-10-01",
        15: "military aid weapons filter:media until:2026-04-05 since:2025-10-01",
        16: "peace talks ceasefire filter:media until:2026-04-05 since:2025-11-01",
        17: "anti-war protests global filter:media until:2026-04-05 since:2025-11-01",
        18: "un security council conflict filter:media until:2026-04-05 since:2025-09-01",
        19: "war economic sanctions filter:media until:2026-04-05 since:2025-09-01",
        20: "nato defense strategy filter:media until:2026-04-05 since:2025-09-01"
    }
    
    query = batch_queries.get(BATCH_NUMBER)
    
    if not query:
        print("Invalid BATCH_NUMBER. Please set it between 1 and 9.")
    else:
        print(f"--- RUNNING BATCH {BATCH_NUMBER} ---")
        print(f"Query: {query}")
        
        # Target 100 per batch. 9 batches = ~900 multimodal pairs
        data = scrape_twitter(query, max_tweets=100) 

        print("\n--- SCRAPING DONE ---")
        
        if data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_path = f"dataset/twitter_dataset_batch{BATCH_NUMBER}_{timestamp}.json"
            
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
                
            print(f"Success! Saved {len(data)} images to 'dataset/twitter_images/'")
            print(f"Success! Saved mapping file to '{json_path}'")
        else:
            print("No tweets were scraped. Check the browser window.")