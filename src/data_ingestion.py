import json
import os
from datetime import datetime

def process_reviews_2023(file_path):
    """
    Processes the Amazon Reviews 2023 dataset (JSON Lines format).
    It maps the new field names to the old structure and processes them efficiently.

    Args:
        file_path (str): The path to the downloaded .jsonl review file.
    """
    # -----------------------------------------------------------
    # 1. New Field Mapping Dictionary for easy reference
    # -----------------------------------------------------------
    FIELD_MAP = {
        'user_id': 'reviewerID',      # new field, maps to old 'reviewerID'
        'parent_asin': 'asin',        # new field, maps to old 'asin'
        'text': 'reviewText',         # new field, maps to old 'reviewText'
        'rating': 'overall',          # new field, maps to old 'overall'
        'title': 'summary',           # new field, maps to old 'summary'
        'sort_timestamp': 'unixReviewTime' # new field, maps to old 'unixReviewTime' (Note: 2023 is in milliseconds!)
    }

    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at path: {file_path}")
        print("Please download one of the category-specific .jsonl files (e.g., 'raw_review_Books.jsonl') and place it in the correct directory.")
        return

    # Counter to track progress and successfully processed lines
    review_count = 0
    successful_count = 0

    print(f"🚀 Starting to process file: {file_path}")
    print("--------------------------------------------------")

    # -----------------------------------------------------------
    # 2. Efficient Line-by-Line Processing (.jsonl format)
    # -----------------------------------------------------------
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            review_count += 1
            
            # Skip empty lines or malformed lines
            if not line.strip():
                continue

            try:
                # Load one JSON object per line
                review_data = json.loads(line)
                
                # Create a structure that mimics the old one, but uses new values
                # If your original code used the OLD field names, the variables 
                # below are what you would use to replace them.
                
                # Get and clean core data
                reviewer_id = review_data.get('user_id')
                asin = review_data.get('parent_asin')
                overall_rating = review_data.get('rating')
                review_text = review_data.get('text')
                summary = review_data.get('title')
                
                # Handle the new timestamp (in milliseconds) and convert to a datetime object
                timestamp_ms = review_data.get('sort_timestamp')
                unix_time_sec = None
                review_time_str = None
                if timestamp_ms is not None:
                    unix_time_sec = int(timestamp_ms / 1000)
                    review_time_str = datetime.fromtimestamp(unix_time_sec).strftime('%Y-%m-%d %H:%M:%S')

                # ***********************************************
                # 3. YOUR ORIGINAL PROCESSING LOGIC GOES HERE
                # ***********************************************
                
                # --- Example of what your original logic might have done: ---
                if reviewer_id and asin and overall_rating and review_text:
                    # You can now use these variables in your existing functions:
                    # e.g., analyze_sentiment(review_text)
                    # e.g., save_to_database(reviewer_id, asin, overall_rating, review_text)
                    
                    if successful_count % 100000 == 0:
                        print(f"✅ Processed {successful_count:,} reviews. Sample: [ASIN: {asin}, Rating: {overall_rating}, Time: {review_time_str}]")
                    
                    successful_count += 1
                # ------------------------------------------------------------
                
            except json.JSONDecodeError as e:
                print(f"⚠️ Warning: JSON Decode Error on line {review_count:,}. Skipping. Error: {e}")
            
            # Optional: Stop after a certain number of reviews for quick testing
            # if review_count > 1000000:
            #     break 

    print("--------------------------------------------------")
    print(f"✨ Processing Complete!")
    print(f"Total lines read: {review_count:,}")
    print(f"Total reviews successfully processed: {successful_count:,}")
    print(f"File: {file_path}")


# =================================================================
# SCRIPT EXECUTION
# =================================================================

# *** IMPORTANT: Change this filename to the category you downloaded ***
# e.g., 'raw_review_Books.jsonl'
DATASET_FILENAME = 'raw_review_Electronics.jsonl' 

if __name__ == "__main__":
    process_reviews_2023(DATASET_FILENAME)