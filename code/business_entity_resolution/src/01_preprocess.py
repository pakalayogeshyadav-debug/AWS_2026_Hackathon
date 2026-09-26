import polars as pl
import re
import argparse
from abydos.phonetic import DoubleMetaphone

dmeta = DoubleMetaphone()

# Fail-safe for restricted AWS environments without C-compilers
try:
    from postal.expand import expand_address
    HAS_LIBPOSTAL = True
except ImportError:
    HAS_LIBPOSTAL = False

# Captures French (sarl, sas, eurl, cedex) and standard global entities
ENTITY_REGEX = r"(?i)\b(sarl|sas|sa|eurl|llc|inc|ltd|corp|corporation|gmbh|cedex|pvt|private|limited)\b"

def generate_phonetic_hash(text: str) -> str:
    if not text:
        return ""
    return dmeta.encode(text)[0]

def normalize_address(address: str) -> str:
    if not address:
        return ""
    if HAS_LIBPOSTAL:
        expansions = expand_address(address)
        return expansions[0] if expansions else address
    
    # Aggressive Regex Fallback
    clean = re.sub(r"[^\w\s]", " ", str(address).lower())
    clean = re.sub(r"\b(rue|avenue|boulevard|street|st|ave|blvd|road|rd)\b", "", clean)
    return " ".join(clean.split())

def preprocess_dataset(input_path: str, output_path: str):
    print(f"Processing: {input_path}")
    df = pl.read_csv(input_path, separator='\t', ignore_errors=True)
    
    df = df.with_columns(
        pl.col("business_name").str.to_lowercase().str.replace_all(r"[^\w\s]", " ").alias("name_clean")
    )
    
    df = df.with_columns(
        pl.col("name_clean").str.extract(ENTITY_REGEX, 0).alias("business_type"),
        pl.col("name_clean").str.replace_all(ENTITY_REGEX, "").str.strip_chars().alias("core_name")
    )
    
    df = df.with_columns(
        pl.col("core_name").map_elements(
            lambda x: generate_phonetic_hash(x), 
            return_dtype=pl.String
        ).alias("phonetic_hash")
    )
    
    df = df.with_columns(
        pl.col("business_address").map_elements(
            lambda x: normalize_address(x), 
            return_dtype=pl.String
        ).alias("address_clean")
    )
    
    # Select original hackathon columns plus the engineered features
    df.select([
        "entity_id", 
        "business_name", 
        "business_address", 
        "country", 
        "business_type", 
        "core_name", 
        "phonetic_hash", 
        "address_clean"
    ]).write_parquet(output_path)
    
    print(f"Saved normalized data to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    preprocess_dataset(args.input, args.output)
    
