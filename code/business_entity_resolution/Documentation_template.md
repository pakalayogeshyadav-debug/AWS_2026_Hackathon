# Technical Documentation: Business Entity Resolution

## 1. Project Overview
This project provides an end-to-end entity resolution pipeline designed to identify and link identical business entities across disparate datasets. Built for the State Zero Hackathon, this system emphasizes high-accuracy semantic matching while strictly adhering to severe memory constraints.

## 2. System Architecture & Pipeline

The pipeline operates in four distinct sequential phases:

### Phase 1: Preprocessing & Standardization (`01_preprocessing.py`)
* **Text Normalization:** Standardizes casing, strips non-alphanumeric punctuation, and expands common abbreviations (e.g., "Corp" to "Corporation").
* **Phonetic Hashing:** Generates Soundex/Metaphone hashes for business names to enable rapid clustering in the blocking phase.

### Phase 2: Memory-Optimized Blocking (`02_blocking.py`)
* **Objective:** Reduce the $O(N^2)$ complexity of pairwise comparisons by grouping highly probable matches.
* **Mechanism:** Utilizes DuckDB for out-of-core SQL execution.
* **Hardware Optimization:** Constrained by a 4GB RAM ceiling, the DuckDB engine is explicitly capped via `PRAGMA memory_limit='1GB'` and forced to use a physical disk-backed database (`temp_hackathon.db`) to prevent Cartesian explosion and out-of-memory (OOM) crashes. Output is capped at 50,000 highly confident candidate pairs to guarantee system stability.

### Phase 3: Semantic Cross-Encoder Matching (`03_cross_encoder.py`)
* **Objective:** Score candidate pairs based on semantic and string similarity.
* **Mechanism:** Employs the deep learning model `cross-encoder/ms-marco-MiniLM-L-6-v2` via HuggingFace `sentence-transformers`.
* **Feature Engineering:** Combines the neural network's semantic similarity score with deterministic string-distance metrics (Jaro-Winkler, Levenshtein distance) calculated via `jellyfish` and `rapidfuzz`.

### Phase 4: Cluster Resolution (`04_clustering.py`)
* **Objective:** Aggregate pairwise matches into unified global entities.
* **Mechanism:** Uses connected-components graph algorithms to assign a single `global_entity_id` to clusters of linked candidate pairs whose match scores exceed the `0.75` confidence threshold.

## 3. Configuration Management
All environmental variables, file paths, and hyperparameters are decoupled from the codebase and managed via `config.yml`. The primary execution driver, `main.py`, orchestrates the sequence of the pipeline using these configurations.

## 4. Scalability and Future Work
While currently hard-capped at 50,000 rows to survive the `ml.t3.medium` memory bottleneck, the architecture is designed to scale. By migrating to a higher-memory instance (e.g., 16GB+ RAM) and removing the `LIMIT` clause in the blocking script, the pipeline can natively process millions of candidate pairs using the exact same codebase.
