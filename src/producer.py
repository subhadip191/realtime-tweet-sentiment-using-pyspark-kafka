import csv
import time
import json
import sys
import os
from kafka import KafkaProducer

# --- Configuration ---
TOPIC_NAME = "sentiment-tweets"
BOOTSTRAP_SERVERS = 'localhost:9092'

# Robust cross-platform path handling
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(BASE_DIR, '..', 'data', 'tweets.csv')

# Dataset: https://www.kaggle.com/datasets/tariqsays/sentiment-dataset-with-1-million-tweets
# Columns in this dataset: Text, Language, Label
# Classes: positive, negative, litigious, uncertainty
TRAINING_SIZE = 5000  # Must match consumer.py

VALID_LABELS = {'positive', 'negative', 'litigious', 'uncertainty'}


def normalize_label(lbl):
    """Normalizes labels to ensure consistency."""
    if not lbl:
        return None
    lbl = str(lbl).strip().lower()
    # Map common aliases
    if lbl in ['positive', 'pos', '1', '1.0']:
        return 'positive'
    if lbl in ['negative', 'neg', '0', '0.0']:
        return 'negative'
    if lbl in ['litigious']:
        return 'litigious'
    if lbl in ['uncertainty', 'uncertain']:
        return 'uncertainty'
    # Return as-is if it's already a known label
    if lbl in VALID_LABELS:
        return lbl
    return None  # Drop unknown labels


# --- Main Execution ---
if __name__ == "__main__":
    print(f"--- Connecting to Kafka at {BOOTSTRAP_SERVERS} ---")
    try:
        producer = KafkaProducer(
            bootstrap_servers=BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        print("SUCCESS: Connected to Kafka!")
    except Exception as e:
        print(f"ERROR: Kafka connection failed. Is Zookeeper/Kafka running?\nDetails: {e}")
        sys.exit(1)

    try:
        if not os.path.exists(CSV_FILE_PATH):
            print(f"ERROR: File not found at {CSV_FILE_PATH}")
            print("Expected path:", os.path.abspath(CSV_FILE_PATH))
            sys.exit(1)

        print(f"Reading from {CSV_FILE_PATH}...")

        with open(CSV_FILE_PATH, 'r', encoding='utf-8', errors='ignore') as file:
            reader = csv.DictReader(file)

            valid_rows = 0
            sent_count = 0
            skipped_count = 0

            print(f"Skipping first {TRAINING_SIZE} valid rows (reserved for training)...")

            for row in reader:
                # Support both Kaggle column name variants
                text = row.get("Text") or row.get("clean_text") or row.get("text")
                raw_label = row.get("Label") or row.get("category") or row.get("target")

                label = normalize_label(raw_label)

                if text and label:
                    valid_rows += 1

                    # 1. SKIP TRAINING DATA (simulates the "Past")
                    if valid_rows <= TRAINING_SIZE:
                        continue

                    # 2. SEND TEST DATA (simulates the "Future/Stream")
                    message = {"Text": text, "Label": label}
                    producer.send(TOPIC_NAME, message)
                    sent_count += 1

                    if sent_count % 100 == 0:
                        print(f"Sent {sent_count} tweets...")
                        time.sleep(0.5)  # Slow down for demo visibility
                    else:
                        time.sleep(0.05)  # Fast stream
                else:
                    skipped_count += 1

        print(f"--- Done! Sent {sent_count} tweets. ({skipped_count} rows skipped due to missing/unknown labels) ---")
        producer.close()

    except KeyboardInterrupt:
        print("\nStream stopped by user.")
        producer.close()
