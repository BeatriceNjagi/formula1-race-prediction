import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.utils.io_utils import load_raw


def run():
    """
    Simple cleaning pipeline.

    Loads race results and qualifying, fixes types,
    creates target labels, merges and saves.
    """

    # --- Load raw files ---
    results_df    = load_raw("ergast_race_results.csv")
    qualifying_df = load_raw("ergast_qualifying.csv")

    print(f"Race results shape: {results_df.shape}")
    print(f"Qualifying shape: {qualifying_df.shape}")

    # --- Clean race results ---

    # Convert date to datetime
    results_df["race_date"] = pd.to_datetime(results_df["race_date"], errors="coerce")

    # Create target labels
    # is_winner: 1 if driver finished in position 1, else 0
    results_df["is_winner"] = (results_df["finish_position"] == 1).astype(int)

    # is_top3: 1 if driver finished in positions 1, 2 or 3, else 0
    results_df["is_top3"]   = (results_df["finish_position"] <= 3).astype(int)

    # Remove duplicates — one row per driver per race
    results_df = results_df.drop_duplicates(subset=["season", "round", "driver_id"])

    # --- Clean qualifying ---

    # Remove duplicates
    qualifying_df = qualifying_df.drop_duplicates(subset=["season", "round", "driver_id"])

    # Keep only the columns we need from qualifying
    qualifying_df = qualifying_df[["season", "round", "driver_id", "qualifying_position"]]

    # --- Merge ---
    # Left join so we keep all race results even if qualifying data is missing
    df = pd.merge(
        results_df,
        qualifying_df,
        on  = ["season", "round", "driver_id"],
        how = "left"
    )

    # Fill missing qualifying positions with 20 (last place)
    df["qualifying_position"] = df["qualifying_position"].fillna(20)

    # --- Sort chronologically ---
    df = df.sort_values(["season", "round", "finish_position"]).reset_index(drop=True)

    # --- Keep only essential columns ---
    df = df[[
        "season",
        "round",
        "race_name",
        "race_date",
        "circuit_id",
        "driver_id",
        "driver_code",
        "constructor_id",
        "grid",
        "qualifying_position",
        "finish_position",
        "points",
        "is_winner",
        "is_top3"
    ]]

    # --- Save ---
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/model_table.csv", index=False)

    print(f"\nModel table saved.")
    print(f"Shape: {df.shape}")
    print(f"\nColumn types:")
    print(df.dtypes)
    print(f"\nSample:")
    print(df.head())


if __name__ == "__main__":
    run()