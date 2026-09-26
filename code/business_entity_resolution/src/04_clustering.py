import pandas as pd
import argparse
import time


# ============================================================
# MATCH THRESHOLDS
# ============================================================

HIGH_CONFIDENCE = 0.85
MEDIUM_CONFIDENCE = 0.70


def classify_pair(row):
    """
    Convert similarity features into a Match / Not Match decision.

    Precision is prioritized because false merges are costly
    in entity resolution.
    """

    name_score = row["name_similarity"]
    address_score = row["address_similarity"]

    name_token = row["name_token_similarity"]
    address_token = row["address_token_similarity"]

    name_exact = row["name_exact_match"]
    address_exact = row["address_exact_match"]

    combined = row["combined_similarity"]

    # --------------------------------------------------------
    # VERY STRONG MATCH
    # --------------------------------------------------------

    if name_exact == 1 and address_exact == 1:
        return 1.0, "MATCH"

    # Very strong business name + reasonable address
    if (
        name_score >= HIGH_CONFIDENCE
        and address_score >= 0.60
    ):
        return combined, "MATCH"

    # Strong token-level name match + address
    if (
        name_token >= 0.90
        and address_score >= 0.65
    ):
        return combined, "MATCH"

    # --------------------------------------------------------
    # MEDIUM CONFIDENCE
    # --------------------------------------------------------

    if (
        combined >= HIGH_CONFIDENCE
        and name_score >= 0.80
    ):
        return combined, "MATCH"

    # --------------------------------------------------------
    # EVERYTHING ELSE
    # --------------------------------------------------------

    return combined, "NOT_MATCH"


def create_clusters(df):
    """
    Create cluster IDs for matched records.

    Each Source-1 entity forms the root of its own
    matching cluster.
    """

    matched = df[
        df["match_decision"] == "MATCH"
    ].copy()

    if matched.empty:
        df["cluster_id"] = ""
        return df

    # Cluster based on Source-1 entity
    cluster_map = {}

    cluster_number = 1

    for entity_id in matched["entity_id_1"].unique():

        cluster_id = f"C{cluster_number:06d}"

        cluster_map[entity_id] = cluster_id

        cluster_number += 1

    df["cluster_id"] = (
        df["entity_id_1"]
        .map(cluster_map)
        .fillna("")
    )

    return df


def process_clustering(input_path, output_path):

    print("======================================")
    print("04 - CLASSIFICATION & CLUSTERING")
    print("======================================")

    start_time = time.time()

    # --------------------------------------------------
    # LOAD FEATURES
    # --------------------------------------------------

    df = pd.read_parquet(input_path)

    print(f"Feature rows loaded: {len(df):,}")

    required_columns = [
        "entity_id_1",
        "entity_id_2",
        "name_similarity",
        "name_token_similarity",
        "address_similarity",
        "address_token_similarity",
        "name_exact_match",
        "address_exact_match",
        "combined_similarity"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # --------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------

    print("Classifying candidate pairs...")

    results = df.apply(
        classify_pair,
        axis=1
    )

    df["match_score"] = results.apply(
        lambda x: x[0]
    )

    df["match_decision"] = results.apply(
        lambda x: x[1]
    )

    # --------------------------------------------------
    # SORT BY SCORE
    # --------------------------------------------------

    df = df.sort_values(
        by=[
            "entity_id_1",
            "match_score"
        ],
        ascending=[
            True,
            False
        ]
    )

    # --------------------------------------------------
    # CLUSTER CREATION
    # --------------------------------------------------

    print("Creating clusters...")

    df = create_clusters(df)

    # --------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------

    output_columns = [
        "entity_id_1",
        "entity_id_2",

        "name_1",
        "name_2",

        "address_1",
        "address_2",

        "name_similarity",
        "name_token_similarity",

        "address_similarity",
        "address_token_similarity",

        "combined_similarity",

        "match_score",
        "match_decision",

        "cluster_id"
    ]

    df[output_columns].to_parquet(
        output_path,
        index=False
    )

    # --------------------------------------------------
    # STATISTICS
    # --------------------------------------------------

    total = len(df)

    matches = (
        df["match_decision"] == "MATCH"
    ).sum()

    not_matches = (
        df["match_decision"] == "NOT_MATCH"
    ).sum()

    clusters = (
        df.loc[
            df["cluster_id"] != "",
            "cluster_id"
        ]
        .nunique()
    )

    elapsed = time.time() - start_time

    print()
    print("Classification completed.")
    print("--------------------------------------")
    print(f"Total candidate pairs : {total:,}")
    print(f"Matches               : {matches:,}")
    print(f"Not matches           : {not_matches:,}")
    print(f"Clusters              : {clusters:,}")
    print(f"Time taken            : {elapsed:.2f} seconds")
    print(f"Output                : {output_path}")
    print("--------------------------------------")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        type=str
    )

    parser.add_argument(
        "--output",
        required=True,
        type=str
    )

    args = parser.parse_args()

    process_clustering(
        args.input,
        args.output
    )