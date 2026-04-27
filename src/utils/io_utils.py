import os
import pandas as pd

def save_raw(df, filename, raw_dir="data/raw"):
    """
    Saves a dataframe to the data/raw folder as a CSV file.
    Creates the folder if it doesn't exist yet.
    
    df       : the pandas dataframe to save
    filename : what to call the file e.g. "ergast_results_2023.csv"
    raw_dir  : the folder to save into, defaults to data/raw
    """

    # Create the folder if it doesn't already exist
    # exist_ok=True means it won't throw an error if the folder is already there
    os.makedirs(raw_dir, exist_ok=True)

    # Build the full file path by joining folder + filename
    filepath = os.path.join(raw_dir, filename)

    # Save the dataframe as a CSV without the row numbers pandas adds by default
    df.to_csv(filepath, index=False)

    # Print confirmation so you know it worked
    print(f"Saved {len(df)} rows to {filepath}")

def load_raw(filename, raw_dir="data/raw"):
    """
    Loads a CSV file from data/raw/ into a pandas dataframe.

    filename : the file to load e.g. "ergast_results.csv"
    raw_dir  : folder to look in, defaults to data/raw
    """

    # Build the full file path
    filepath = os.path.join(raw_dir, filename)

    # Raise a clear error if the file doesn't exist yet
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No file found at {filepath}. Run ingestion first.")

    # Load the CSV into a pandas dataframe
    df = pd.read_csv(filepath)

    print(f"Loaded {len(df)} rows from {filepath}")

    return df