import os
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import randint
#prueba
pd.set_option("display.max_columns", None)

# Adjust this path if needed
COMPETITION_PATH = "."


def load_competition_datasets(data_dir, sample_frac=None, random_state=None):
    """
    Load train and test datasets, optionally sample a fraction of the training set,
    concatenate, and reset index.
    """
    print("Loading competition datasets from:", data_dir)
    train_file = os.path.join(data_dir, "competition_data/train_data.txt")
    test_file = os.path.join(data_dir, "competition_data/test_data.txt")

    # Load training data and optionally subsample
    train_df = pd.read_csv(train_file, sep="\t", low_memory=False)
    if sample_frac is not None:
        train_df = train_df.sample(frac=sample_frac, random_state=random_state)

    # Load test data
    test_df = pd.read_csv(test_file, sep="\t", low_memory=False)

    # Concatenate and reset index
    combined = pd.concat([train_df, test_df], ignore_index=True)
    print(f"  → Concatenated DataFrame: {combined.shape[0]} rows")
    return combined


def cast_column_types(df):
    """
    Cast columns to efficient dtypes and parse datetime fields.
    """
    print("Casting column types and parsing datetime fields...")
    dtype_map = {
        "platform": "category",
        "conn_country": "category",
        "ip_addr": "category",
        "master_metadata_track_name": "category",
        "master_metadata_album_artist_name": "category",
        "master_metadata_album_album_name": "category",
        "reason_end": "category",
        "username": "category",
        "spotify_track_uri": "string",
        "episode_name": "string",
        "episode_show_name": "string",
        "spotify_episode_uri": "string",
        "audiobook_title": "string",
        "audiobook_uri": "string",
        "audiobook_chapter_uri": "string",
        "audiobook_chapter_title": "string",
        "shuffle": bool,
        "offline": bool,
        "incognito_mode": bool,
        "obs_id": int,
    }

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["offline_timestamp"] = pd.to_datetime(
        df["offline_timestamp"], unit="s", errors="coerce", utc=True
    )
    df = df.astype(dtype_map)
    print("  → Column types cast successfully.")
    return df


def split_train_valid_test(df, X, y, test_mask):
    """
    Split features and labels into train/valid/test based on masks.
    """
    print("Splitting data into train/valid/test sets...")


    # Identificar train (no test)
    not_test_mask = ~test_mask
    train_df = df[not_test_mask]

    # Calcular punto de corte para validación
    n_total = len(train_df)
    n_valid = int(0.3 * n_total)

    # Último 30% de train = validación
    valid_df = train_df.iloc[-n_valid:]
    train_df = train_df.iloc[:-n_valid]

    # Construimos test
    test_df = df[test_mask]

    # Separamos X e y
    y_train = train_df["target"].to_numpy()
    y_valid = valid_df["target"].to_numpy()
    y_test  = test_df["target"].to_numpy()

    X_train = train_df.drop(columns=["target"])
    X_valid = valid_df.drop(columns=["target"])
    X_test  = test_df.drop(columns=["target"])

    print(f"→ Train: {X_train.shape[0]} filas")
    print(f"→ Valid: {X_valid.shape[0]} filas ({0.3*100:.0f}% del train)")
    print(f"→ Test:  {X_test.shape[0]} filas")

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def train_classifier_random_search(X_train, y_train, n_iter=10, n_splits=4, random_state=22):
    print("Training model with RandomizedSearchCV...")

    rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)

    param_dist = {
        "n_estimators": randint(50, 300),
        "max_depth": randint(3, 50),
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 20),
        "criterion": ["gini", "entropy", "log_loss"],
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False]
    }

    # TimeSeriesSplit para respetar orden temporal
    tscv = TimeSeriesSplit(n_splits=n_splits)

    rs = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=tscv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )

    rs.fit(X_train, y_train)
    print(f"Best params: {rs.best_params_}")
    print(f"Best CV ROC AUC: {rs.best_score_:.4f}")
    return rs





def main():
    print("=== Starting pipeline ===")

    # Load and preprocess data
    df = load_competition_datasets(
        COMPETITION_PATH, sample_frac=0.2, random_state=1234
    )
    df = cast_column_types(df)

    # Generate user order column
    df = df.sort_values(["username", "ts"])
    df["user_order"] = df.groupby("username", observed=True).cumcount() + 1
    df = df.sort_values(["obs_id"])

    
    # Create target and test mask

    print("Creating 'target' and 'is_test' columns...")
    df["target"] = (df["reason_end"] == "fwdbtn").astype(int)
    df["is_test"] = df["reason_end"].isna()
    df.drop(columns=["reason_end"], inplace=True)
    print("  → 'target' and 'is_test' created, dropped 'reason_end' column.")

    to_keep = [
        "obs_id",
        "target",
        "is_test",
        "user_order",
        "shuffle",	
        "offline",
        "incognito_mode"
    ] #MODIFICAR
    df = df[to_keep]

   # Build feature matrix and get feature names
    y = df["target"].to_numpy()
    X = df.drop(columns=["target"])
    feature_names = X.columns
    test_mask = df["is_test"].to_numpy()
    print(df)

    # Split data
    X_train, X_valid, X_test, y_train, y_valid, _ = split_train_valid_test(df, X, y, test_mask)
     
    # Train model
    model = train_classifier_random_search(X_train, y_train)

    # Evaluate on validation set
    print("Evaluating on validation set...")
    y_val_pred = model.predict_proba(X_valid)[:, 1]  # probabilidades para la clase positiva
    val_auc = roc_auc_score(y_valid, y_val_pred)
    print(f"  → Validation ROC AUC: {val_auc:.4f}")

    # Display top 20 feature importances
    print("Extracting and sorting feature importances...")
    importances = model.best_estimator_.feature_importances_
    imp_series = pd.Series(importances, index=feature_names)
    imp_sorted = imp_series.sort_values(ascending=False)
    print("\nTop 20 feature importances:")
    print(imp_sorted.head(20))

    # Predict on test set
    print("Generating predictions for test set...")
    test_obs_ids = X_test["obs_id"]
    preds_proba = model.predict_proba(X_test)[:, 1]
    preds_df = pd.DataFrame({"obs_id": test_obs_ids, "pred_proba": preds_proba})
    preds_df.to_csv("modelo_valid_rand.csv", index=False, sep=",")
    print(f"  → Predictions written to 'modelo_valid_rand.csv'")

    print("=== Pipeline complete ===")


if __name__ == "__main__":
    main()