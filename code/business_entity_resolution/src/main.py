import os
import sys
import yaml
import subprocess
import pandas as pd

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yml")

def load_config():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def run_pipeline(config):
    print("=" * 60)
    print(f"Starting {config['project']['name']} Pipeline")
    print("=" * 60)
    
    src_dir = os.path.dirname(__file__)
    
    # Step 1: Blocking
    blocking_script = os.path.join(src_dir, "02_blocking.py")
    cmd_blocking = [
        sys.executable, blocking_script,
        "--source1", config["paths"]["raw_data_source1"],
        "--source2", config["paths"]["raw_data_source2"],
        "--output", config["paths"]["candidate_pairs"]
    ]
    print("\n[Step 1/2] Running DuckDB candidate blocking...")
    subprocess.run(cmd_blocking, check=True)

    # Step 2: Cross-Encoder Inference
    encoder_script = os.path.join(src_dir, "03_cross_encoder.py")
    cmd_encoder = [
        sys.executable, encoder_script,
        "--input", config["paths"]["candidate_pairs"],
        "--output", config["paths"]["matching_results"]
    ]
    print("\n[Step 2/2] Running Cross-Encoder scoring...")
    subprocess.run(cmd_encoder, check=True)

    print("\nPipeline execution complete.")

def print_summary(config):
    results_path = config["paths"]["matching_results"]
    if os.path.exists(results_path):
        df = pd.read_csv(results_path, sep="\t")
        print("=" * 60)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 60)
        print(f"Total scored pairs : {len(df):,}")
        if "score" in df.columns:
            matches = (df["score"] >= config["model"]["match_threshold"]).sum()
            print(f"High-confidence matches (>= {config['model']['match_threshold']}): {matches:,}")
        print(f"Artifact location   : {results_path}")
    else:
        print(f"No results found at {results_path}. Run the pipeline first.")

if __name__ == "__main__":
    cfg = load_config()
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        print_summary(cfg)
    else:
        run_pipeline(cfg)