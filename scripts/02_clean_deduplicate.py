import os
import pandas as pd
from datasketch import MinHash, MinHashLSH
from ftfy import fix_text
from detoxify import Detoxify
import re

class DataCleaner:
    def __init__(self, toxicity_threshold=0.7):
        self.toxicity_model = Detoxify('original')
        self.toxicity_threshold = toxicity_threshold
        self.lsh = MinHashLSH(threshold=0.85, num_perm=128)
        self.seen_hashes = set()

    def clean_text(self, text):
        # Fix encoding issues
        text = fix_text(text)
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove PII patterns
        text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '[EMAIL]', text)
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
        text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', text)
        return text.strip()

    def is_toxic(self, text):
        results = self.toxicity_model.predict(text)
        return any(score > self.toxicity_threshold for score in results.values())

    def is_duplicate(self, text, doc_id):
        # MinHash for near-duplicate detection
        m = MinHash(num_perm=128)
        for word in text.split():
            m.update(word.encode('utf8'))

        # Check LSH
        result = self.lsh.query(m)
        if result:
            return True

        self.lsh.insert(doc_id, m)
        return False

    def filter_quality(self, text, min_words=50, max_words=10000):
        word_count = len(text.split())
        if word_count < min_words or word_count > max_words:
            return False

        # Check character variety (detect gibberish)
        unique_chars = len(set(text.lower()))
        if unique_chars < 20:
            return False

        return True

def process_pretraining_data():
    cleaner = DataCleaner()

    for file in os.listdir("data/raw"):
        if not file.startswith("pretraining_"):
            continue

        print(f"Processing {file}...")
        df = pd.read_parquet(f"data/raw/{file}")

        cleaned = []
        for idx, row in df.iterrows():
            text = cleaner.clean_text(row['text'])

            # Filter pipeline
            if not cleaner.filter_quality(text):
                continue
            if cleaner.is_toxic(text):
                continue
            if cleaner.is_duplicate(text, f"{file}_{idx}"):
                continue

            cleaned.append({"text": text, "source": file})

        # Save cleaned data
        output_df = pd.DataFrame(cleaned)
        output_path = f"data/processed/{file.replace('.parquet', '_clean.parquet')}"
        output_df.to_parquet(output_path)

        print(f"  Kept {len(cleaned)}/{len(df)} documents ({len(cleaned)/len(df)*100:.1f}%)")

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    process_pretraining_data()
