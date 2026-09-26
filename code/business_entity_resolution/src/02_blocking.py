import duckdb
import argparse
import time

def generate_blocks(source1_path, source2_path, output_path):
    print(f"Starting DuckDB blocking phase...")
    start_time = time.time()
    
    # Connect to an in-memory DuckDB instance
    con = duckdb.connect(database=':memory:')
    
    # Block on phonetic_hash OR address_clean to ensure high recall
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
        ON (s1.phonetic_hash = s2.phonetic_hash AND s1.phonetic_hash != '')
        OR (s1.address_clean = s2.address_clean AND s1.address_clean != '')
    ) TO '{output_path}' (FORMAT PARQUET);
    """
    
    con.execute(query)
    print(f"Blocking complete in {time.time() - start_time:.2f} seconds.")
    print(f"Candidate pairs saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source1", type=str, required=True)
    parser.add_argument("--source2", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    generate_blocks(args.source1, args.source2, args.output)
