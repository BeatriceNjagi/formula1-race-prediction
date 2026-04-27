import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


def run():
    """
    Simple feature engineering pipeline.

    Adds one rolling feature to the model table:
    avg_finish_last5 — average finish position over last 5 races per driver.

    Final features used by the model:
    1. qualifying_position   — where the driver starts the race
    2. avg_finish_last5      — how the driver has been performing recently
    """

    # --- Load model table ---
    df = pd.read_csv("data/processed/model_table.csv")
    df["race_date"] = pd.to_datetime(df["race_date"])

    print(f"Loaded model table: {df.shape}")

    # --- Sort chronologically ---
    df = df.sort_values(["season", "round", "driver_id"]).reset_index(drop=True)

    # --- Add rolling average finish position ---
    # For each driver look at their last 5 races and average the finish position
    # shift(1) means the current race is never included in its own calculation
    # min_periods=1 means we still calculate if fewer than 5 races are available
    df["avg_finish_last5"] = (
        df.groupby("driver_id")["finish_position"]
        .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
    )

    # For a driver's very first race they have no history at all
    # Fill with the global average finish position as a fallback
    global_avg = df["finish_position"].mean()
    df["avg_finish_last5"] = df["avg_finish_last5"].fillna(global_avg)

    # --- Select final columns ---
    df = df[[
        # Identity — needed for filtering and display
        "season",
        "round",
        "race_name",
        "race_date",
        "circuit_id",
        "driver_id",
        "driver_code",
        "constructor_id",

        # Features — inputs to the model
        "qualifying_position",   # strongest predictor
        "avg_finish_last5",      # recent form

        # Targets — what the model predicts
        "is_winner",
        "is_top3",

        # Extra — useful for analysis and Streamlit app
        "finish_position",
        "points"
    ]]

    # --- Validation ---
    print(f"\nValidation:")
    print(f"  Shape: {df.shape}")
    print(f"  Missing values:\n{df.isnull().sum()}")
    print(f"\n  is_top3 rate: {round(df['is_top3'].mean()*100, 1)}%")
    print(f"  is_winner rate: {round(df['is_winner'].mean()*100, 1)}%")
    print(f"\n  Feature preview:")
    print(df[["driver_id", "qualifying_position", "avg_finish_last5", "is_top3"]].head(10))

    # --- Save ---
    df.to_csv("data/processed/features.csv", index=False)

    print(f"\nSaved to data/processed/features.csv")
    print(f"Final shape: {df.shape}")


if __name__ == "__main__":
    run()