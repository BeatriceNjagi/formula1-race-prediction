import pandas as pd
import numpy as np
import os
import sys
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


def load_features():
    """
    Loads the feature table built by build_features.py.
    """
    filepath = "data/processed/features.csv"

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No file at {filepath}. Run build_features.py first.")

    df = pd.read_csv(filepath)
    print(f"Loaded features: {df.shape}")
    return df


def split_data(df, target):
    """
    Splits data into train and test sets using a time aware split.

    We train on seasons 2021-2024 and test on 2025.
    This mirrors real world usage — you train on past seasons
    and predict the current or future season.

    Never use random splitting for time series data because
    it allows future data to leak into training.

    df     : feature dataframe
    target : column name to predict e.g. "is_top3" or "is_winner"
    """

    # Define the two feature columns the model uses
    feature_cols = ["qualifying_position", "avg_finish_last5"]

    # Train on 2021-2024
    train = df[df["season"] <= 2024]

    # Test on 2025
    test  = df[df["season"] == 2025]

    print(f"\nTrain set: {len(train)} rows (seasons 2021-2024)")
    print(f"Test set:  {len(test)} rows (season 2025)")

    # Separate features from target
    X_train = train[feature_cols]
    y_train = train[target]

    X_test  = test[feature_cols]
    y_test  = test[target]

    print(f"\nTarget: {target}")
    print(f"  Train positive rate: {round(y_train.mean()*100, 1)}%")
    print(f"  Test positive rate:  {round(y_test.mean()*100, 1)}%")

    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train):
    """
    Trains a Logistic Regression model.

    This is the baseline model. It is simple, fast, and interpretable.
    We scale the features first because Logistic Regression is sensitive
    to feature magnitudes — qualifying_position ranges 1-20 while
    avg_finish_last5 also ranges roughly 1-20 so scaling helps.

    Returns the trained model and the scaler used to transform features.
    """

    print("\nTraining Logistic Regression...")

    # Scale features to have mean 0 and standard deviation 1
    # This helps Logistic Regression converge faster and perform better
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # class_weight="balanced" tells the model to pay more attention
    # to the minority class (podium finishers) since they are only 15%
    # of the data. Without this the model would just predict "no podium"
    # for everyone and get 85% accuracy without learning anything useful.
    model = LogisticRegression(
        class_weight = "balanced",
        max_iter     = 1000,
        random_state = 42
    )

    model.fit(X_scaled, y_train)
    print("  Logistic Regression trained ✓")

    return model, scaler


def train_random_forest(X_train, y_train):
    """
    Trains a Random Forest model.

    This is the advanced model. It builds many decision trees and
    combines their predictions. It handles non-linear relationships
    and does not need feature scaling.

    For example it can learn that a driver who qualifies 1st AND
    has been finishing well recently is much more likely to win
    than a driver who qualifies 1st but has been finishing 10th.
    Logistic Regression struggles with these interactions.

    Returns the trained model.
    """

    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators  = 200,    # number of trees — more trees = more stable
        max_depth     = 5,      # limits tree depth to prevent overfitting
        class_weight  = "balanced",
        random_state  = 42,
        n_jobs        = -1      # use all CPU cores for faster training
    )

    model.fit(X_train, y_train)
    print("  Random Forest trained ✓")

    return model


def evaluate_model(model, X_test, y_test, model_name, scaler=None):
    """
    Evaluates a trained model on the test set and prints metrics.

    model      : trained sklearn model
    X_test     : test features
    y_test     : true labels
    model_name : name to display in output
    scaler     : if provided, scales X_test before predicting
                 needed for Logistic Regression but not Random Forest
    """

    print(f"\n--- {model_name} Results ---")

    # Scale test features if a scaler was provided
    # Important: use transform not fit_transform on test data
    # We only fit the scaler on training data
    if scaler:
        X_eval = scaler.transform(X_test)
    else:
        X_eval = X_test

    # Get predictions
    y_pred = model.predict(X_eval)

    # Get probability scores — useful for ranking drivers
    y_prob = model.predict_proba(X_eval)[:, 1]

    # Calculate metrics
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    cm        = confusion_matrix(y_test, y_pred)

    print(f"  Accuracy:  {round(accuracy*100, 1)}%")
    print(f"  Precision: {round(precision*100, 1)}%")
    print(f"  Recall:    {round(recall*100, 1)}%")
    print(f"  F1 Score:  {round(f1*100, 1)}%")
    print(f"\n  Confusion Matrix:")
    print(f"  {cm}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return {
        "model_name" : model_name,
        "accuracy"   : accuracy,
        "precision"  : precision,
        "recall"     : recall,
        "f1"         : f1,
        "confusion_matrix" : cm.tolist()
    }


def save_artifacts(model, filename, scaler=None):
    """
    Saves a trained model to the artifacts/ folder as a .pkl file.
    pkl (pickle) is Python's standard format for saving objects to disk.

    model    : trained sklearn model to save
    filename : name for the saved file e.g. "lr_is_top3.pkl"
    scaler   : if provided, saves the scaler alongside the model
    """

    os.makedirs("artifacts", exist_ok=True)

    # Save the model
    model_path = f"artifacts/{filename}"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved model to {model_path}")

    # Save the scaler if provided
    if scaler:
        scaler_name = filename.replace(".pkl", "_scaler.pkl")
        scaler_path = f"artifacts/{scaler_name}"
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"  Saved scaler to {scaler_path}")


def run():
    """
    Master function that trains and evaluates both models
    for both prediction targets.

    Targets:
    - is_top3   : predict podium finish
    - is_winner : predict race winner

    Models:
    - Logistic Regression (baseline)
    - Random Forest (advanced)
    """

    # --- Load features ---
    df = load_features()

    # --- Install sklearn if not already installed ---
    # Run: uv add scikit-learn

    # --- Train and evaluate for each target ---
    for target in ["is_top3", "is_winner"]:

        print(f"\n{'='*50}")
        print(f"TARGET: {target}")
        print(f"{'='*50}")

        # Split data
        X_train, X_test, y_train, y_test = split_data(df, target)

        # --- Logistic Regression ---
        lr_model, scaler = train_logistic_regression(X_train, y_train)
        lr_metrics = evaluate_model(
            lr_model, X_test, y_test,
            model_name = f"Logistic Regression ({target})",
            scaler     = scaler
        )
        save_artifacts(lr_model, f"lr_{target}.pkl", scaler=scaler)

        # --- Random Forest ---
        rf_model = train_random_forest(X_train, y_train)
        rf_metrics = evaluate_model(
            rf_model, X_test, y_test,
            model_name = f"Random Forest ({target})"
        )
        save_artifacts(rf_model, f"rf_{target}.pkl")

        # --- Feature importance from Random Forest ---
        feature_cols = ["qualifying_position", "avg_finish_last5"]
        importances  = rf_model.feature_importances_

        print(f"\nFeature Importances ({target}):")
        for feature, importance in zip(feature_cols, importances):
            print(f"  {feature}: {round(importance*100, 1)}%")

    print(f"\n{'='*50}")
    print("Training complete.")
    print("Artifacts saved to artifacts/ folder.")
    print(f"{'='*50}")


if __name__ == "__main__":
    run()