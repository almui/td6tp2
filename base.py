import os
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,roc_curve
from scipy.stats import uniform
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
    df = df.astype(dtype_map)
    print("  → Column types cast successfully.")
    return df


def split_train_valid_test(df):
    """
    Split features and labels into train/valid/test based on masks.
    """
    print("Splitting data into train/valid/test sets...")

    test_mask = df["is_test"].to_numpy()
    train_valid_df = df[~test_mask]
    n_valid = int(0.1 * len(train_valid_df))

    valid_df = train_valid_df.iloc[-n_valid:]
    train_df = train_valid_df.iloc[:-n_valid]
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
    ]
    df = df[to_keep]

    # Split data
    X_train, y_train, X_valid, y_valid, X_test, y_test = split_train_valid_test(df)

    # Train Gradient Boosting model
    print("Training Gradient Boosting model...")

    params = {'max_depth': list(range(1, 40)),
          'learning_rate': uniform(scale = 0.2),
          'gamma': uniform(scale = 2),
          'reg_lambda': uniform(scale = 5),        # Parámetro de regularización.
          'subsample': uniform(0.5, 0.5),          # Entre 0.5 y 1.
          'min_child_weight': uniform(scale = 5),
          'colsample_bytree': uniform(0.75, 0.25), # Entre 0.75 y 1.
          'n_estimators': list(range(1, 1000))
         }
    start = time.time()
    best_score = 0
    best_estimator = None
    iterations = 20
    for g in ParameterSampler(params, n_iter = iterations, random_state = 1234):
        clf_xgb = xgb.XGBClassifier(objective = 'binary:logistic', seed = 1234, eval_metric = 'auc', **g)
        clf_xgb.fit(X_train, y_train, eval_set = [(X_valid, y_valid)], verbose = False)

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

    # xgb_params = {'colsample_bytree': 0.75,
    #           'gamma': 0.5,
    #           'learning_rate': 0.075,
    #           'max_depth': 8,
    #           'min_child_weight': 1,
    #           'n_estimators': 500,
    #           'reg_lambda': 0.5,
    #           'subsample': 0.75,
    #           }

    # model = xgb.XGBClassifier(objective = 'binary:logistic',
    #                         seed = 1234,
    #                         eval_metric = 'auc',
    #                         **xgb_params)
     
    model.fit(X_train, y_train, verbose=100, eval_set=[(X_train,y_train), (X_valid, y_valid)])

    # Evaluate on validation set
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

 
    preds_df.to_csv("modelo_xgboost_iter.csv", index=False)
    print(f"  → Predictions written to 'modelo_xgboost_iter.csv")

    print("=== Pipeline complete ===")

if __name__ == "__main__":
    main()