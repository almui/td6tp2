import os
import json
import random
import pandas as pd
import numpy as np
from scipy.stats import uniform, randint
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterSampler
import xgboost as xgb
import time
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
import time
from hyperopt import fmin, tpe, hp, STATUS_OK

pd.set_option("display.max_columns", None)

COMPETITION_PATH = "."

# =========================
# Loaders & Preprocessing
# =========================

def load_competition_datasets(data_dir, sample_frac=None, random_state=None, max_files=None):
    print("Loading competition datasets from:", data_dir)
    train_file = os.path.join(data_dir, "competition_data/train_data.txt")
    test_file = os.path.join(data_dir, "competition_data/test_data.txt")

    train_df = pd.read_csv(train_file, sep="\t", low_memory=False)
    if sample_frac is not None:
        train_df = train_df.sample(frac=sample_frac, random_state=random_state)

    test_df = pd.read_csv(test_file, sep="\t", low_memory=False)

    combined = pd.concat([train_df, test_df], ignore_index=True)
    print(f"  → Concatenated DataFrame: {combined.shape[0]} rows")
    lista_track, lista_episode = [], []

    api_dir = os.path.join(data_dir, "competition_data/spotify_api_data")
    archivos = os.listdir(api_dir)

    if max_files:
        max_files = min(max_files, len(archivos))  # por si pones un número mayor al total
        print(f"Loading {max_files/len(archivos)}% of the Spotify API data")
        random.seed(random_state)
        archivos = random.sample(archivos, max_files)

    for i, nombre_archivo in enumerate(archivos, 1):
        ruta_archivo = os.path.join(api_dir, nombre_archivo)
        if not (nombre_archivo.startswith('spotify_track') or nombre_archivo.startswith('spotify_episode')):
            continue

        with open(ruta_archivo, 'r') as f:
            datos_json = json.load(f)

        df = pd.json_normalize(datos_json, sep='.')
        if nombre_archivo.startswith('spotify_track'):
            lista_track.append(df)
        else:
            lista_episode.append(df)

        print(f"[{i}/{len(archivos)}] Loaded {nombre_archivo}")

    df_track = pd.concat(lista_track, ignore_index=True) if lista_track else pd.DataFrame()
    df_episode = pd.concat(lista_episode, ignore_index=True) if lista_episode else pd.DataFrame()
    return combined, df_track, df_episode

def cast_column_types(df, df_t, df_e):
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
    # ---Tracks---
    columnas_a_eliminar_t = [
        "href", "id", "uri", "preview_url", "type",
        "album.external_urls.spotify", "album.href", "album.id",
        "album.images", "album.uri",
        "external_ids.isrc", "external_urls.spotify", "album.artists", "artists"
    ]
    df_t = df_t.drop(columnas_a_eliminar_t, errors="ignore", axis=1)

    dtype_map_t = {
        "name": "category",
        "album.name": "category",
        "album.album_type": "category",
        "album.total_tracks": int,
        "album.type": "category",
        "album.release_date_precision": "category",
        "explicit": bool,
        "is_local": bool,
        "duration_ms": int,
        "disc_number": int,
        "track_number": int,
        "popularity": int,
        "external_ids.isrc": "string",
        "album.release_date": "string",
        "album.available_markets": "string",
        "available_markets": "string"
    }

    # Aplicás tipos sin iterar uno por uno
    for col, dtype in dtype_map_t.items():
        if col in df_t.columns:
            df_t[col] = df_t[col].astype(dtype)

    # ---Episodes---
    columnas_a_eliminar_e = [
        "href", "id", "uri", "external_urls.spotify",
        "show.external_urls.spotify", "show.href", "show.id",
        "show.images", "audio_preview_url", "show.uri", "show.html_description", "show.description",
        "show.copyrights", "html_description", "description", "images"
    ]
    df_e = df_e.drop(columns=columnas_a_eliminar_e, errors="ignore", axis=1)
    dtype_map_e = {
        "name": "category",
        "language": "category",
        "languages": "string",
        "release_date": "string",
        "release_date_precision": "category",
        "explicit": bool,
        "is_externally_hosted": bool,
        "is_playable": bool,
        "duration_ms": int,
        "show.name": "category",
        "show.publisher": "category",
        "show.type": "category",
        "show.total_episodes": int,
        "show.media_type": "category",
        "show.languages": "string",
        "show.is_externally_hosted ": bool,
        "show.explicit": bool,
        "show.available_markets": "string",
        "type": "category"
    }

    for col, dtype in dtype_map_e.items():
        if col in df_e.columns:
            df_e[col] = df_e[col].astype(dtype)
    print("  → Column types cast successfully.")

    return df, df_t, df_e

# =========================
# Feature Engineering
# =========================

def category_to_int(df, category):
    df[f"int_{category}"] = pd.factorize(df[category])[0] + 1
    return df

def one_hot_encoding(df, category):
    encoder = OneHotEncoder(sparse_output=False, dtype = int)
    one_hot_encoded = encoder.fit_transform(df[[category]])
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out([category]))
    df = pd.concat([df, one_hot_df], axis=1)

    nuevas_variables = one_hot_df.columns.tolist()

    return df, nuevas_variables
def add_engineered_features(df):
    # --- OS name ---
    df["platform_os"] = df["platform"].str.split().str[0].str.lower()
    df["platform_os"] = df["platform_os"].str.replace("webplayer", "web_player")

    # --- Local time ---
    offsets = {
        "AR": -3, "ES": 1, "CL": -3, "NL": 1, "BE": 1, "DE": 1, "CZ": 1, "LT": 2,
        "LV": 2, "EE": 2, "IE": 0, "GB": 0, "BR": -3, "FR": 1, "ZZ": 0, "US": -5,
        "PY": -4, "UY": -3, "CO": -5, "CR": -6, "GR": 2, "FI": 2, "ZA": 2, "DK": 1,
        "RU": 3, "AT": 1, "JM": -5, "MT": 1, "IT": 1, "AU": 10, "SK": 1, "RO": 2,
        "GE": 4, "LU": 1, "SG": 8, "AL": 1, "CH": 1, "TW": 8, "HR": 1, "BG": 2,
        "PE": -5, "PA": -5, "MX": -6, "NO": 1,
    }
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
        elif 15 <= hour < 20:
            return "afternoon"
        else:
            return "night"

    df["time_of_day"] = df["local_ts"].dt.hour.map(categorize_hour)
    df, plataformas = one_hot_encoding(df, "platform_os")
    df = category_to_int(df, "time_of_day")

    # --- Temporal dynamics ---
    df["day_of_week"] = df["local_ts"].dt.dayofweek
    df["hour"] = df["local_ts"].dt.hour
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df, plataformas


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
    test_obs_ids = test_df["obs_id"].to_numpy()
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"].to_numpy()

    print(f"→ Train: {X_train.shape[0]} filas")
    print(f"→ Valid: {X_valid.shape[0]} filas")
    print(f"→ Test:  {X_test.shape[0]} filas")

    return X_train, y_train, X_valid, y_valid, X_test, y_test, test_obs_ids


def train_classifier(X_train, y_train, X_valid, y_valid, params=None):
    def objective(params):
        # Convertir a enteros si es necesario
        params['max_depth'] = int(params['max_depth'])
        params['min_child_weight'] = int(params['min_child_weight'])
        params['n_estimators'] = int(params['n_estimators'])
        clf = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            seed=1234,
            **params
        )
        clf.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False
        )
        y_pred = clf.predict_proba(X_valid)[:, 1]
        auc = roc_auc_score(y_valid, y_pred)
        return {'loss': -auc, 'status': STATUS_OK}
    

    start = time.time()
    best = fmin(
        fn=objective,
        space=params,
        algo=tpe.suggest,
        max_evals=80,
        rstate=np.random.default_rng(1234)  # semilla reproducible
    )
    final_params = {
        "max_depth": int(best["max_depth"]),
        "learning_rate": best["learning_rate"],
        "subsample": best["subsample"],
        "colsample_bytree": best["colsample_bytree"],
        "min_child_weight": int(best["min_child_weight"]),
        "gamma": best["gamma"],
        "reg_lambda": best["reg_lambda"],
        "reg_alpha": best["reg_alpha"],
        "n_estimators": int(best["n_estimators"])
    }

    end = time.time()
    
    print('Grilla:', final_params)
    print(f'Tiempo transcurrido: {str(end - start)} segundos')
    best_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        seed=1234,
        **final_params
    )
    best_model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False
        )
    y_pred = best_model.predict_proba(X_valid)[:, 1]
    best_score= roc_auc_score(y_valid, y_pred)
    print('ROC-AUC: %0.5f' % best_score)
    return best_model

# =========================
# Main
# =========================

def main():
    print("=== Starting pipeline ===")
    df, df_t, df_s = load_competition_datasets(COMPETITION_PATH, sample_frac=1, random_state=1234, max_files=20500)
    df, df_t, df_s = cast_column_types(df, df_t, df_s)
    df = df.sort_values(["obs_id"]) 
    df_t = df_t.rename(columns={"name": "master_metadata_track_name"})
    df_s = df_s.rename(columns={"name": "episode_name"})

    df = df.merge(df_t, on="master_metadata_track_name", how="left", suffixes=("", "_track"))
    df = df.merge(df_s, on="episode_name", how="left", suffixes=("", "_episode"))
    for col in ["explicit", "duration_ms"]:
        track_col = f"{col}_track"
        episode_col = f"{col}_episode"
        if track_col in df.columns or episode_col in df.columns:
            df[col] = df.get(track_col, df.get(episode_col))
            df[col] = df[col].combine_first(df.get(episode_col))
        df = df.drop(columns=[c for c in [track_col, episode_col] if c in df.columns])
    


    # Basic encodings
    df = category_to_int(df, "master_metadata_album_album_name")
    df = category_to_int(df, "master_metadata_album_artist_name")
    df = category_to_int(df, "master_metadata_track_name")
    df = category_to_int(df, "platform")
    df = category_to_int(df, "username")
    df = category_to_int(df, "episode_name")
    df = category_to_int(df, "episode_show_name")    
    df = category_to_int(df, "conn_country")
    df = category_to_int(df, "ip_addr")
    df = category_to_int(df, "album.type")    
    df = category_to_int(df, "release_date_precision")
    df = category_to_int(df, "explicit")    
    df = category_to_int(df, "is_local")    
    df = category_to_int(df, "is_playable")
    df = category_to_int(df, "type")
    df = category_to_int(df, "language")
    df = category_to_int(df, "show.media_type")


    # Feature engineering
    df, platafromas = add_engineered_features(df)
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

    df = df.sort_values(["obs_id"])
    to_keep = [
        "obs_id", "int_time_of_day",
        "int_username", *platafromas, "int_conn_country", "int_episode_name",
        "int_master_metadata_track_name", "int_master_metadata_album_artist_name", "target", "is_test",
        "shuffle", "offline", "incognito_mode", 
        "count_morning", "count_noon", "count_afternoon", "count_night",
        "day_of_week", "hour",
        "ts_diff", "session_pos", "session_len",
        "user_track_count", "user_artist_count", "is_weekend", 
        "disc_number", "int_explicit","int_is_local", "int_is_playable", 
        "popularity", "track_number", "duration_ms", "int_album.type", "int_release_date_precision",
        "int_type", "album.total_tracks", "int_language", "int_show.media_type", 
    ]
    df = df[to_keep]

    # Split
    X_train, y_train, X_valid, y_valid, X_test, y_test, test_obs_ids = split_train_valid_test(df)

    # Balance de clases
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f"→ scale_pos_weight = {scale_pos_weight:.2f}")

    
    # Train model
    print("Training XGBoosting model...")
    params = {
        "max_depth": hp.uniform("max_depth", 3, 10),
        "learning_rate": hp.uniform("learning_rate", 0.01, 0.2),
        "subsample": hp.uniform("subsample", 0.5, 0.8),        # <= más bajo, más robusto
        "colsample_bytree": hp.uniform("colsample_bytree",0.5, 1.0), # <= más bajo, más robusto
        "min_child_weight": hp.uniform('min_child_weight', 5, 15),    # evita sobreajuste a outliers
        "gamma": hp.uniform("gamma",2.0, 7.0),
        "reg_lambda": hp.uniform('reg_lambda', 5.0, 20.0),          # más regularización L2
        "reg_alpha": hp.uniform("reg_alpha", 0.0, 5.0),            # agregá regularización L1
        "n_estimators": hp.uniform("n_estimators", 100, 800)
    }


    model = train_classifier(X_train, y_train, X_valid, y_valid, params)

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
    preds_proba = model.predict_proba(X_test)[:, 1]
    preds_df = pd.DataFrame({"obs_id": test_obs_ids, "pred_proba": preds_proba})

    # Eliminar posibles obs_id duplicados
    preds_df = preds_df.drop_duplicates(subset="obs_id", keep="last")

    preds_df.to_csv("modelo_masvariables4.csv", index=False)
    print("  → Predictions written to 'modelo_masvariables4.csv'")

    print("=== Pipeline complete ===")

if __name__ == "__main__":
    main()
