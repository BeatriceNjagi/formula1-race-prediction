import requests
import pandas as pd
import yaml
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.utils.io_utils import save_raw


def load_config():
    """
    Opens and reads config/settings.yaml.
    Returns a dictionary with all the settings inside.
    """
    with open("config/settings.yaml", "r") as f:
        return yaml.safe_load(f)


def fetch_race_results(base_url, season):
    """
    Pulls all race results for one season with pagination.
    Jolpica returns max 100 results per page so we keep
    fetching until there are no more pages left.
    """

    rows = []
    offset = 0    # offset tells the API where to start from
    limit  = 100  # number of results per page

    while True:
        # Build URL with limit and offset for pagination
        url = f"{base_url}/{season}/results.json?limit={limit}&offset={offset}"
        print(f"  Fetching race results: season={season} offset={offset}")

        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        races = data["MRData"]["RaceTable"]["Races"]

        # If no races returned we have reached the end — stop looping
        if not races:
            break

        for race in races:
            season_year = race["season"]
            race_round  = race["round"]
            race_name   = race["raceName"]
            race_date   = race["date"]
            circuit_id  = race["Circuit"]["circuitId"]

            for result in race["Results"]:
                row = {
                    "season"          : season_year,
                    "round"           : race_round,
                    "race_name"       : race_name,
                    "race_date"       : race_date,
                    "circuit_id"      : circuit_id,
                    "driver_id"       : result["Driver"]["driverId"],
                    "driver_code"     : result["Driver"].get("code", None),
                    "constructor_id"  : result["Constructor"]["constructorId"],
                    "grid"            : result["grid"],
                    "finish_position" : result["position"],
                    "points"          : result["points"],
                    "status"          : result["status"],
                    "laps_completed"  : result["laps"]
                }
                rows.append(row)

        # Check total results available from the API response
        total = int(data["MRData"]["total"])

        # If we have fetched everything stop looping
        if offset + limit >= total:
            break

        # Otherwise move to the next page
        offset += limit

        # Pause briefly between page requests
        time.sleep(0.3)

    return rows


def fetch_qualifying(base_url, season):
    """
    Pulls all qualifying results for one season with pagination.
    """

    rows = []
    offset = 0
    limit  = 100

    while True:
        url = f"{base_url}/{season}/qualifying.json?limit={limit}&offset={offset}"
        print(f"  Fetching qualifying: season={season} offset={offset}")

        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        races = data["MRData"]["RaceTable"]["Races"]

        if not races:
            break

        for race in races:
            race_round = race["round"]
            race_name  = race["raceName"]

            for result in race["QualifyingResults"]:
                row = {
                    "season"              : season,
                    "round"               : race_round,
                    "race_name"           : race_name,
                    "driver_id"           : result["Driver"]["driverId"],
                    "constructor_id"      : result["Constructor"]["constructorId"],
                    "qualifying_position" : result["position"],
                    "q1_time"             : result.get("Q1", None),
                    "q2_time"             : result.get("Q2", None),
                    "q3_time"             : result.get("Q3", None)
                }
                rows.append(row)

        total = int(data["MRData"]["total"])

        if offset + limit >= total:
            break

        offset += limit
        time.sleep(0.3)

    return rows


def run(start_season, end_season):
    """
    Master function that loops over every season and pulls
    race results and qualifying data then saves to data/raw/.
    """

    config   = load_config()
    base_url = config["ergast"]["base_url"]

    all_results    = []
    all_qualifying = []

    for season in range(start_season, end_season + 1):
        print(f"\nProcessing season {season}...")

        # --- Race results ---
        results = fetch_race_results(base_url, season)
        all_results.extend(results)

        # --- Qualifying ---
        qualifying = fetch_qualifying(base_url, season)
        all_qualifying.extend(qualifying)

        # Pause between seasons
        time.sleep(1)

    # Convert to dataframes
    race_results_df    = pd.DataFrame(all_results)
    qualifying_df = pd.DataFrame(all_qualifying)

    # Save both to data/raw/
    save_raw(race_results_df,    "ergast_race_results.csv")
    save_raw(qualifying_df, "ergast_qualifying.csv")

    print("\nErgast ingestion complete.")


if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 2021
    end   = int(sys.argv[2]) if len(sys.argv) > 2 else 2025
    run(start, end)