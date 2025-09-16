import os
import pandas as pd
import numpy as np
from scipy.stats import uniform, randint
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterSampler
import xgboost as xgb
import time

pd.set_option("display.max_columns", None)

COMPETITION_PATH = "."

# =========================
# Loaders & Preprocessing
# =========================

def load_competition_datasets(data_dir, sample_frac=None, random_state=None):
    print("Loading competition datasets from:", data_dir)
    train_file = os.path.join(data_dir, "competition_data/train_data.txt")
    test_file = os.path.join(data_dir, "competition_data/test_data.txt")

    train_df = pd.read_csv(train_file, sep="\t", low_memory=False)
    if sample_frac is not None:
        train_df = train_df.sample(frac=sample_frac, random_state=random_state)

    test_df = pd.read_csv(test_file, sep="\t", low_memory=False)

    combined = pd.concat([train_df, test_df], ignore_index=True)
    print(f"  → Concatenated DataFrame: {combined.shape[0]} rows")
    return combined

def cast_column_types(df):
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

# =========================
# Feature Engineering
# =========================

def category_to_int(df, category):
    df[f"int_{category}"] = pd.factorize(df[category])[0] + 1
    return df

def add_engineered_features(df):
    # --- OS name ---
    df["platform_os"] = df["platform"].str.split().str[0]

    # --- Local time ---
    offsets = {"AR": -3, "US": -5}
    df["local_ts"] = df.apply(
        lambda row: row["ts"] + pd.Timedelta(hours=offsets.get(row["conn_country"], 0)),
        axis=1,
    )

    # --- Time of day categories ---
    def categorize_hour(hour):
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 15:
            return "noon"
        elif 15 <= hour < 21:
            return "afternoon"
        else:
            return "night"

    df["time_of_day"] = df["local_ts"].dt.hour.map(categorize_hour)
    df = category_to_int(df, "platform_os")
    df = category_to_int(df, "time_of_day")

    # --- Temporal dynamics ---
    df["day_of_week"] = df["local_ts"].dt.dayofweek
    df["hour"] = df["local_ts"].dt.hour

    return df

def add_user_daytime_counts(df):
    user_day_counts = (
        df.groupby(["username", "time_of_day"]).size().reset_index(name="count")
    )
    user_day_counts = user_day_counts.pivot(
        index="username", columns="time_of_day", values="count"
    ).fillna(0)
    user_day_counts = user_day_counts.add_prefix("count_").reset_index()
    df = df.merge(user_day_counts, on="username", how="left")
    return df

def add_session_features(df):
    df = df.sort_values(["username", "ts"])
    df["ts_diff"] = df.groupby("username")["ts"].diff().dt.total_seconds().fillna(0)
    df["new_session"] = (df["ts_diff"] > 1800).astype(int)
    df["session_id"] = df.groupby("username")["new_session"].cumsum()
    df["session_pos"] = df.groupby(["username", "session_id"]).cumcount() + 1
    df["session_len"] = df.groupby(["username", "session_id"])["obs_id"].transform("count")
    return df

def add_repetition_features(df):
    df = df.sort_values(["username", "ts"])
    df["user_track_count"] = df.groupby(["username", "int_master_metadata_track_name"]).cumcount()
    df["user_artist_count"] = df.groupby(["username", "int_master_metadata_album_artist_name"]).cumcount()
    return df

def add_skip_history(df: pd.DataFrame) -> pd.DataFrame:
    # User’s rolling skip rate up to the current observation
    df["user_skip_rate"] = (
        df.groupby("username")["target"]
          .transform(lambda x: x.shift().expanding().mean())
    )
    
    # Fill NaN for first observation
    df["user_skip_rate"] = df["user_skip_rate"].fillna(0.0)

    # Recent skip streak: last N plays skipped (e.g., 3)
    df["user_recent_skip_sum"] = (
        df.groupby("username")["target"]
          .transform(lambda x: x.shift().rolling(3, min_periods=1).sum())
    )

    return df

# =========================
# Train / Eval
# =========================

def split_train_valid_test(df):
    print("Splitting data into train/valid/test sets...")
    test_mask = df["is_test"].to_numpy()
    train_valid_df = df[~test_mask]
    n_valid = int(0.1 * len(train_valid_df))

    train_df = train_valid_df.iloc[:-n_valid]
    valid_df = train_valid_df.iloc[-n_valid:]
    test_df = df[test_mask]

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"].to_numpy()
    X_valid = valid_df.drop(columns=["target"])
    y_valid = valid_df["target"].to_numpy()
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"].to_numpy()

    print(f"→ Train: {X_train.shape[0]} filas")
    print(f"→ Valid: {X_valid.shape[0]} filas")
    print(f"→ Test:  {X_test.shape[0]} filas")

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def train_classifier(X_train, y_train, X_valid, y_valid, params=None):
    start = time.time()
    best_score = 0
    model = None
    iterations = 20
    for g in ParameterSampler(params, n_iter=iterations, random_state=1234):
        clf_xgb = xgb.XGBClassifier(
            objective="binary:logistic", seed=1234, eval_metric="auc", **g
        )
        clf_xgb.fit(X_train, y_train, verbose=False, eval_set=[(X_valid, y_valid)])

        y_pred = clf_xgb.predict_proba(X_valid)[:, 1]
        auc_roc = roc_auc_score(y_valid, y_pred)
        if auc_roc > best_score:
            print(f'Mejor valor de ROC-AUC encontrado: {auc_roc}')
            best_score = auc_roc
            best_grid = g
            model = clf_xgb

    end = time.time()
    print('ROC-AUC: %0.5f' % best_score)
    print('Grilla:', best_grid)
    print(f'Tiempo transcurrido: {str(end - start)} segundos')
    print(f'Tiempo de entrenamiento por iteración: {str(round((end - start) / iterations, 2))} segundos')
    return model

# =========================
# Main
# =========================

def main():
    print("=== Starting pipeline ===")
    df = load_competition_datasets(COMPETITION_PATH, sample_frac=1, random_state=1234)
    df = cast_column_types(df)
    df = df.sort_values(["obs_id"])

    # Basic encodings
    df = category_to_int(df, "master_metadata_album_album_name")
    df = category_to_int(df, "master_metadata_album_artist_name")
    df = category_to_int(df, "master_metadata_track_name")
    df = category_to_int(df, "platform")
    df = category_to_int(df, "username")

    # Feature engineering
    df = add_engineered_features(df)
    df = add_user_daytime_counts(df)

    # Create target/test
    print("Creating 'target' and 'is_test' columns...")
    df["target"] = (df["reason_end"] == "fwdbtn").astype(int)
    df["is_test"] = df["reason_end"].isna()
    df.drop(columns=["reason_end"], inplace=True)
    print("  → 'target' and 'is_test' created, dropped 'reason_end' column.")

    # Session + repetition + temporal + skip history
    df = add_session_features(df)
    df = add_repetition_features(df)
    df = add_skip_history(df)

    # Columns to keep
    to_keep = [
        "obs_id", "int_platform", "int_platform_os", "int_time_of_day",
        "int_master_metadata_track_name", "int_master_metadata_album_artist_name",
        "int_master_metadata_album_album_name", "target", "is_test",
        "shuffle", "offline", "incognito_mode", "int_username",
        "count_morning", "count_noon", "count_afternoon", "count_night",
        "day_of_week", "hour",
        "ts_diff", "session_pos", "session_len",
        "user_track_count", "user_artist_count", "user_skip_rate"
    ]
    df = df[to_keep]

    # Split
    X_train, y_train, X_valid, y_valid, X_test, y_test = split_train_valid_test(df)

    # Train model
    print("Training XGBoosting model...")
    params = {
        "max_depth": list(range(3, 10)),
        "learning_rate": uniform(0.01, 0.2),
        "subsample": uniform(0.5, 0.5),        # <= más bajo, más robusto
        "colsample_bytree": uniform(0.5, 0.5), # <= más bajo, más robusto
        "min_child_weight": randint(5, 20),    # evita sobreajuste a outliers
        "gamma": uniform(0, 5),
        "reg_lambda": uniform(5, 20),          # más regularización L2
        "reg_alpha": uniform(0, 5),            # agregá regularización L1
        "n_estimators": list(range(200, 1001, 100))
    }
    model = train_classifier(X_train, y_train, X_valid, y_valid, params)

    model.fit(X_train, y_train, verbose=100, eval_set=[(X_valid, y_valid)])

    print("Evaluating on training set...")
    train_pred = model.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, train_pred)
    print(f"  → Training ROC AUC: {train_auc:.4f}")

    print("Evaluating on validation set...")
    y_val_pred = model.predict_proba(X_valid)[:, 1]
    val_auc = roc_auc_score(y_valid, y_val_pred)
    print(f"  → Validation ROC AUC: {val_auc:.4f}")

    # Importances
    importances = model.feature_importances_
    imp_series = pd.Series(importances, index=X_train.columns)
    imp_sorted = imp_series.sort_values(ascending=False)
    print("\nTop 20 feature importances:")
    print(imp_sorted.head(20))

    # Predictions
    print("Generating predictions for test set...")
    test_obs_ids = X_test["obs_id"]
    preds_proba = model.predict_proba(X_test)[:, 1]
    preds_df = pd.DataFrame({"obs_id": test_obs_ids, "pred_proba": preds_proba})
    preds_df.to_csv("modelo_masvariables2.csv", index=False)
    print("  → Predictions written to 'modelo_masvariables.csv'")

    print("=== Pipeline complete ===")

if __name__ == "__main__":
    main()
