import os
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,roc_curve
from scipy.stats import uniform, randint
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, ParameterSampler
from scipy.stats import randint
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
import time

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
    # df['tp_local'] = df['ts'].dt.tz_convert('Europe/Madrid')
    # print("\nFechas convertidas a Madrid:")
    # print(df['timestamp_madrid'])
    df = df.astype(dtype_map)
    print("  → Column types cast successfully.")
    return df

def add_engineered_features(df):
    # --- 1. Extract OS name from platform ---
    df["platform_os"] = df["platform"].str.split().str[0]

    # --- 2. Convert ts to local time based on country ---
    # Define offsets: Argentina ~ UTC-3, US ~ UTC-5 (rough, no DST handling)
    # You can refine with pytz if exact TZ needed
    offsets = {"AR": -3, "US": -5}

    df["local_ts"] = df.apply(
        lambda row: row["ts"] + pd.Timedelta(hours=offsets.get(row["conn_country"], 0)),
        axis=1,
    )

    # --- 3. Create time-of-day categories ---
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

    # --- 4. Encode to int for model ---
    df = category_to_int(df, "platform_os")
    df = category_to_int(df, "time_of_day")

    return df

def split_train_valid_test(df):
    """
    Split features and labels into train/valid/test based on masks.
    """
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
    print(f"→ Valid: {X_valid.shape[0]} filas ({0.1*100:.0f}% del train)")
    print(f"→ Test:  {X_test.shape[0]} filas")

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def train_classifier(X_train, y_train,X_valid, y_valid, params=None):
    start = time.time()
    best_score = 0
    model = None
    iterations = 20
    for g in ParameterSampler(params, n_iter = iterations, random_state = 1234):
        clf_xgb = xgb.XGBClassifier(objective = 'binary:logistic', seed = 1234, eval_metric = 'auc', **g)
        clf_xgb.fit(X_train, y_train, verbose=False,eval_set=[(X_valid, y_valid)])

        y_pred = clf_xgb.predict_proba(X_valid)[:, 1] # Obtenemos la probabilidad de una de las clases (cualquiera).
        auc_roc = roc_auc_score(y_valid, y_pred)
        # Guardamos si es mejor.
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

def category_to_int(df, category):
    df[f"int_{category}"] = pd.factorize(df[category])[0] + 1
    return df

def add_user_daytime_counts(df):
    # Contar escuchas por usuario y franja horaria
    user_day_counts = (
        df.groupby(["username", "time_of_day"])
          .size()
          .reset_index(name="count")
    )

    # Pivot: columnas = franjas, filas = usuarios
    user_day_counts = user_day_counts.pivot(
        index="username", columns="time_of_day", values="count"
    ).fillna(0)

    # Renombrar columnas para claridad
    user_day_counts = user_day_counts.add_prefix("count_").reset_index()

    # Merge back al df original
    df = df.merge(user_day_counts, on="username", how="left")

    return df


def main():
    print("=== Starting pipeline ===")

    # Load and preprocess data
    df = load_competition_datasets(
        COMPETITION_PATH, sample_frac=1,random_state=1234
    )
    df = cast_column_types(df)
    df = df.sort_values(["obs_id"])
    df = category_to_int(df, "master_metadata_album_album_name")
    df = category_to_int(df, "master_metadata_album_artist_name")
    df = category_to_int(df, "master_metadata_track_name")
    df =category_to_int(df, "platform")
    df =category_to_int(df, "username")
    df = add_engineered_features(df)
    df = add_user_daytime_counts(df)
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
        "int_platform",
        "int_platform_os",        
        "int_time_of_day",      
        "int_master_metadata_track_name",
        "int_master_metadata_album_artist_name",
        "int_master_metadata_album_album_name",
        "target",
        "is_test",
        "user_order",
        "shuffle",
        "offline",
        "incognito_mode",
        "int_username",
        "count_morning",
        "count_noon",
        "count_afternoon",
        "count_night",
    ]
    df = df[to_keep]

    # Split data
    X_train, y_train, X_valid, y_valid, X_test, y_test = split_train_valid_test(df)

  
    # Train Gradient Boosting model
    print("Training XGBoosting model...")

    params = {
        "max_depth": list(range(2, 16)),                # allow shallower + deeper trees
        "learning_rate": uniform(0.01, 0.3),            # [0.01, 0.31] slower to faster learning
        "gamma": uniform(0, 10),                        # [0, 10]
        "reg_lambda": uniform(0, 20),                   # stronger regularization range
        "subsample": uniform(0.3, 0.7),                 # [0.3, 1.0]
        "min_child_weight": randint(1, 15),             # integer, 1–15
        "colsample_bytree": uniform(0.3, 0.7),          # [0.3, 1.0]
        "n_estimators": list(range(50, 1001, 50))       # 50 up to 1000
    }
    model = train_classifier(X_train, y_train, X_valid, y_valid, params)

    model.fit(X_train, y_train, verbose=100,eval_set=[(X_valid, y_valid)])
 
    # Evaluate on trainig & validation set
    print("Evaluating on training set...")
    train_pred = model.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, train_pred)
    print(f"  → Trainig ROC AUC: {train_auc:.4f}")

    print("Evaluating on validation set...")
    y_val_pred = model.predict_proba(X_valid)[:, 1]
    val_auc = roc_auc_score(y_valid, y_val_pred)
    print(f"  → Validation ROC AUC: {val_auc:.4f}")

    # Feature importances
    importances = model.feature_importances_
    imp_series = pd.Series(importances, index=X_train.columns)
    imp_sorted = imp_series.sort_values(ascending=False)
    print("\nTop 20 feature importances:")
    print(imp_sorted.head(20))

    # Predict on test set
    print("Generating predictions for test set...")
    test_obs_ids = X_test["obs_id"]
    preds_proba = model.predict_proba(X_test)[:, 1]
    preds_df = pd.DataFrame({"obs_id": test_obs_ids, "pred_proba": preds_proba}) 
    preds_df.to_csv("modelo_masvariables.csv", index=False)
    print(f"  → Predictions written to 'modelo_masvariables.csv")

    print("=== Pipeline complete ===")

if __name__ == "__main__":
    main()