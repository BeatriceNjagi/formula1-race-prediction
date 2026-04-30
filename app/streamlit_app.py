import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- Fix working directory ---
# This ensures all file paths work correctly on Streamlit Cloud
# by always pointing to the project root regardless of where
# Streamlit runs the app from
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Page config ---
st.set_page_config(
    page_title = "F1 Race Prediction",
    page_icon  = "🏎️",
    layout     = "wide"
)


# --- Helper functions ---

def load_model(filepath):
    """
    Loads a saved model or scaler from the artifacts/ folder.
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)


@st.cache_data
def get_data():
    """
    Loads the feature table built by build_features.py.
    Cached so it only loads once per session.
    """
    df = pd.read_csv("data/processed/features.csv")
    df["race_date"] = pd.to_datetime(df["race_date"])
    return df


@st.cache_resource
def get_models():
    """
    Loads all four model artifacts.
    Cached so models only load once per session.
    """
    rf_top3   = load_model("artifacts/rf_is_top3.pkl")
    rf_winner = load_model("artifacts/rf_is_winner.pkl")
    return rf_top3, rf_winner


def get_predictions(df, season, round_num, rf_top3, rf_winner):
    """
    Generates top3 and winner predictions for all drivers
    in a selected race.

    df        : full feature dataframe
    season    : selected season year
    round_num : selected round number
    rf_top3   : trained Random Forest model for is_top3
    rf_winner : trained Random Forest model for is_winner
    """

    # Filter to selected race
    race_df = df[
        (df["season"] == season) &
        (df["round"]  == round_num)
    ].copy()

    if race_df.empty:
        return None

    # Features the model uses
    feature_cols = ["qualifying_position", "avg_finish_last5"]
    X = race_df[feature_cols]

    # Get probability of top3 finish for each driver
    race_df["top3_probability"] = rf_top3.predict_proba(X)[:, 1]

    # Get probability of winning for each driver
    race_df["winner_probability"] = rf_winner.predict_proba(X)[:, 1]

    # Sort by top3 probability descending
    race_df = race_df.sort_values(
        "top3_probability", ascending=False
    ).reset_index(drop=True)

    return race_df


# --- App ---

st.title("🏎️ F1 Race Prediction System")
st.markdown(
    "Predicting podium finishes and race winners using "
    "qualifying position and recent form."
)

# --- Load data and models ---
try:
    df = get_data()
    rf_top3, rf_winner = get_models()

except FileNotFoundError as e:
    st.error(f"Required file not found: {e}")
    st.stop()

except Exception as e:
    st.error(f"Error loading data or models: {e}")
    st.stop()

# --- Sidebar ---
st.sidebar.header("🔍 Select a Race")

# Season selector
seasons = sorted(df["season"].unique(), reverse=True)
selected_season = st.sidebar.selectbox("Season", seasons)

# Race selector filtered by season
season_df    = df[df["season"] == selected_season]
race_options = (
    season_df
    .drop_duplicates(subset=["round"])
    [["round", "race_name"]]
    .sort_values("round")
)
race_labels = {
    row["race_name"]: row["round"]
    for _, row in race_options.iterrows()
}

selected_race_name = st.sidebar.selectbox(
    "Race", list(race_labels.keys())
)
selected_round = race_labels[selected_race_name]

# --- Race header ---
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"📍 {selected_race_name}")
    st.caption(f"Season {selected_season} — Round {selected_round}")

with col2:
    race_row  = df[
        (df["season"] == selected_season) &
        (df["round"]  == selected_round)
    ].iloc[0]
    race_date = pd.to_datetime(race_row["race_date"]).strftime("%d %B %Y")
    st.subheader(f"📅 {race_date}")
    st.caption(f"Circuit: {race_row['circuit_id'].replace('_', ' ').title()}")

st.divider()

# --- Generate predictions ---
predictions = get_predictions(
    df, selected_season, selected_round, rf_top3, rf_winner
)

if predictions is None:
    st.error("No data found for this race.")
    st.stop()

# --- Predicted Podium ---
st.subheader("🏆 Predicted Podium")

top3 = predictions.head(3)

pod_col1, pod_col2, pod_col3 = st.columns(3)

with pod_col1:
    st.metric(
        label = "🥇 Predicted Winner",
        value = top3.iloc[0]["driver_code"].upper(),
        delta = f"{round(top3.iloc[0]['winner_probability']*100, 1)}% win probability"
    )

with pod_col2:
    st.metric(
        label = "🥈 Predicted P2",
        value = top3.iloc[1]["driver_code"].upper(),
        delta = f"{round(top3.iloc[1]['top3_probability']*100, 1)}% podium probability"
    )

with pod_col3:
    st.metric(
        label = "🥉 Predicted P3",
        value = top3.iloc[2]["driver_code"].upper(),
        delta = f"{round(top3.iloc[2]['top3_probability']*100, 1)}% podium probability"
    )

st.divider()

# --- Full driver predictions table ---
st.subheader("📊 All Driver Predictions")

display_df = predictions[[
    "driver_code",
    "constructor_id",
    "qualifying_position",
    "avg_finish_last5",
    "top3_probability",
    "winner_probability",
    "is_top3",
    "is_winner"
]].copy()

display_df.columns = [
    "Driver",
    "Constructor",
    "Qualifying Position",
    "Avg Finish Last 5",
    "Podium Probability",
    "Win Probability",
    "Actual Top 3",
    "Actual Winner"
]

display_df["Podium Probability"] = display_df["Podium Probability"].apply(
    lambda x: f"{round(x*100, 1)}%"
)
display_df["Win Probability"] = display_df["Win Probability"].apply(
    lambda x: f"{round(x*100, 1)}%"
)
display_df["Avg Finish Last 5"] = display_df["Avg Finish Last 5"].apply(
    lambda x: round(x, 1)
)
display_df["Actual Top 3"]  = display_df["Actual Top 3"].apply(
    lambda x: "✅" if x == 1 else ""
)
display_df["Actual Winner"] = display_df["Actual Winner"].apply(
    lambda x: "🏆" if x == 1 else ""
)
display_df["Driver"]      = display_df["Driver"].str.upper()
display_df["Constructor"] = display_df["Constructor"].str.title()

st.dataframe(display_df, use_container_width=True, hide_index=True)

st.divider()

# --- Driver form chart ---
st.subheader("📈 Driver Form — Avg Finish Last 5 Races")
st.caption("Lower is better — position 1 is the best finish")

form_df = predictions[["driver_code", "avg_finish_last5"]].copy()
form_df = form_df.sort_values("avg_finish_last5", ascending=True)
form_df.columns = ["Driver", "Avg Finish Last 5 Races"]
form_df["Driver"] = form_df["Driver"].str.upper()

st.bar_chart(form_df.set_index("Driver"))

st.divider()

# --- Model performance info ---
st.subheader("ℹ️ About This Model")

info_col1, info_col2 = st.columns(2)

with info_col1:
    st.markdown("""
    **Model:** Random Forest Classifier

    **Features used:**
    - Qualifying position
    - Average finish position last 5 races

    **Trained on:** Seasons 2021 — 2024

    **Tested on:** Season 2025
    """)

with info_col2:
    st.markdown("""
    **Model Performance (2025 test season):**

    | Metric | Podium | Winner |
    |---|---|---|
    | Accuracy | 86.2% | 89.1% |
    | Recall | 90.3% | 100% |
    | Precision | 52.4% | 31.6% |
    | F1 Score | 66.3% | 48.0% |
    """)

st.divider()
st.caption("Built with FastF1, Jolpica API, scikit-learn and Streamlit")