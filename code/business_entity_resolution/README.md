Enterprise Business Entity Resolution EngineState Zero Hackathon SubmissionA high-performance, memory-optimized entity resolution framework designed to identify, block, and match duplicate business records across large-scale, heterogeneous datasets under strict resource ceilings.🚀 Key FeaturesMemory-Bounded DuckDB Blocking (02_blocking.py): Utilizes an embedded SQL engine with strict memory caps and disk-backed temp storage to bypass $O(N^2)$ computational explosion and prevent Out-Of-Memory (OOM) crashes on resource-constrained hardware (ml.t3.medium).Deep Semantic Cross-Encoding (03_cross_encoder.py): Employs state-of-the-art transformer models (cross-encoder/ms-marco-MiniLM-L-6-v2) via HuggingFace sentence-transformers for deep contextual similarity scoring.Hybrid Feature Engineering: Combines deep semantic vectors with high-speed string distance metrics (Jaro-Winkler, Levenshtein distance) using rapidfuzz and jellyfish.Config-Driven Architecture: Fully decoupled parameters, data paths, and confidence thresholds managed via a centralized config.yml.Automated Pipeline Orchestration (main.py): Seamless end-to-end pipeline execution and verification driver.📁 Project StructurePlaintextbusiness_entity_resolution/
├── config.yml                 # Centralized configuration & hyperparameters
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation & overview
├── Documentation_template.md  # Detailed technical architecture document
├── output/                    # Processed TSV artifacts and intermediate matches
└── src/
    ├── 01_preprocessing.py    # Data cleaning & phonetic hashing
    ├── 02_blocking.py         # Out-of-core DuckDB candidate pair generation
    ├── 03_cross_encoder.py    # Neural semantic scoring & feature engineering
    ├── 04_clustering.py       # Graph-based connected-components resolution
    └── main.py                # Pipeline execution driver & summary CLI
🛠️ Installation & SetupClone the repository and enter the project root:Bashcd ~/shared/State_Zero_submission/code/business_entity_resolution/
Install required dependencies:Bashpip install -r requirements.txt
⚙️ Configuration (config.yml)Operational settings, file paths, and memory limits are controlled via config.yml:YAMLproject:
  name: "State_Zero_Entity_Resolution"
  version: "1.0.0"

paths:
  candidate_pairs: "output/candidate_pairs.tsv"
  matching_results: "output/matching_results.tsv"

model:
  name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  match_threshold: 0.75
🏃 Usage & Execution1. Run the Full PipelineExecute the end-to-end pipeline via the orchestrator:Bashpython src/main.py
2. View Execution SummaryCheck output metrics and high-confidence match distributions:Bashpython src/main.py --summary
📊 Output Artifactsoutput/candidate_pairs.tsv: Screened entity pairs generated during the blocking phase.output/matching_results.tsv: Final scored pairs containing semantic similarity probabilities and string alignment features.
