import pandas as pd
import numpy as np
import argparse
import time
import jellyfish
from rapidfuzz import fuzz
from sentence_transformers import CrossEncoder

def generate_features(input_path, output_path):
    print("======================================")
    print("03 - CROSS-ENCODER & FEATURE ENGINEERING")
    print("======================================")
    
    start_time = time.time()

    # 1. LOAD CANDIDATE PAIRS
    df = pd.read_parquet(input_path)
    print(f"Candidate pairs loaded: {len(df):,}")

    # 2. VECTORIZED TEXT CLEANING (Massive speed boost over .apply)
    print("Cleaning text data...")
    for col in ["name_1", "name_2", "address_1", "address_2"]:
        df[col] = df[col].fillna("").astype(str).str.lower().str.strip()
        # Replace multiple spaces with a single space
        df[col] = df[col].replace(r'\s+', ' ', regex=True)

    # 3. TRUE DEEP LEARNING CROSS-ENCODER
    # Using a lightweight, fast model to protect your AWS budget/RAM
    print("Running Semantic Cross-Encoder (Deep Learning)...")
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
    
    # Combine name and address for deep semantic context
    text_pairs = list(zip(
        df["name_1"] + " " + df["address_1"], 
        df["name_2"] + " " + df["address_2"]
    ))
    
    # Predict in batches to prevent SageMaker Out-of-Memory (OOM) crashes
    df["semantic_cross_score"] = model.predict(text_pairs, batch_size=256, show_progress_bar=True)

    # 4. FAST STRING & PHONETIC FEATURES (For Hari's XGBoost Model)
    print("Calculating phonetic and string similarity features...")
    
    # Extract columns to flat lists for list-comprehension (10x faster than df.apply)
    n1_list, n2_list = df["name_1"].tolist(), df["name_2"].tolist()
    a1_list, a2_list = df["address_1"].tolist(), df["address_2"].tolist()

    # RapidFuzz Ratios
    df["name_fuzz_ratio"] = [fuzz.ratio(a, b) / 100.0 for a, b in zip(n1_list, n2_list)]
    df["name_token_ratio"] = [fuzz.token_set_ratio(a, b) / 100.0 for a, b in zip(n1_list, n2_list)]
    df["address_fuzz_ratio"] = [fuzz.ratio(a, b) / 100.0 for a, b in zip(a1_list, a2_list)]

    # Phonetic Matching (Crucial for regional spelling variations like Kamaraj vs Kamarajar)
    df["name_phonetic_match"] = [
        1 if jellyfish.soundex(a) == jellyfish.soundex(b) else 0 
        for a, b in zip(n1_list, n2_list)
    ]

    # Exact Matches
    df["name_exact_match"] = (df["name_1"] == df["name_2"]).astype(int)
    df["address_exact_match"] = (df["address_1"] == df["address_2"]).astype(int)

    # 5. FINAL OUTPUT (Notice there is NO combined_similarity score)
    output_columns = [
        "entity_id_1", "entity_id_2",
        "name_1", "name_2", "address_1", "address_2",
        "semantic_cross_score",      # Deep learning feature
        "name_fuzz_ratio",           # String feature
        "name_token_ratio",          # String feature
        "address_fuzz_ratio",        # String feature
        "name_phonetic_match",       # Phonetic feature
        "name_exact_match",          # Boolean feature
        "address_exact_match"        # Boolean feature
    ]

    result = df[output_columns]
    result.to_parquet(output_path, index=False)

    elapsed = time.time() - start_time
    print(f"\nFeature engineering completed in {elapsed:.2f} seconds.")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    args = parser.parse_args()
    
    generate_features(args.input, args.output)