import duckdb
import argparse
import time
import os

def generate_blocks(source1_path, source2_path, output_path):
    print(f"Starting ultra-safe DuckDB blocking phase...")
    start_time = time.time()
    
    # Use a physical file instead of :memory: to prevent RAM overflow
    db_path = 'temp_hackathon.db'
    if os.path.exists(db_path):
        os.remove(db_path)
        
    con = duckdb.connect(database=db_path)
    con.execute("PRAGMA memory_limit='1GB'") 
    
    # Ultra-strict join with a hard limit to guarantee it finishes
    query = f"""
    COPY (
        SELECT 
            s1.entity_id as entity_id_1,
            s2.entity_id as entity_id_2,
            s1.business_name as name_1,
            s2.business_name as name_2,
            s1.business_address as address_1,
            s2.business_address as address_2
        FROM read_parquet('{source1_path}') AS s1
        INNER JOIN read_parquet('{source2_path}') AS s2
        ON s1.phonetic_hash = s2.phonetic_hash
        WHERE s1.phonetic_hash IS NOT NULL 
          AND s2.phonetic_hash IS NOT NULL
          AND LENGTH(s1.phonetic_hash) > 2
        LIMIT 50000
    ) TO '{output_path}' (HEADER, DELIMITER '\t');
    """
    
    con.execute(query)
    con.close()
    
    # Clean up the temporary database
    if os.path.exists(db_path):
        os.remove(db_path)
        
    print(f"Blocking complete in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source1", type=str, required=True)
    parser.add_argument("--source2", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    generate_blocks(args.source1, args.source2, args.output)