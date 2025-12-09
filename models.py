import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import optuna


def preprocess_model_data(df):
    # Step 1: Convert datetime into numeric index
    df["tourney_datetime"] = pd.to_datetime(df["tourney_datetime"])
    df["tourney_time_idx"] = (df["tourney_datetime"] - pd.Timestamp("1970-01-01")).dt.days

    # Step 2: One-hot encode categoricals
    categorical = ["player_hand", "opponent_hand", "surface", "tourney_level", "round"]
    df = pd.get_dummies(df, columns=categorical, drop_first=True)

    # Step 3: Drop unwanted columns
    df = df.drop(columns=["tourney_datetime"])

    # Step 4: Separate target
    y = df["df_rate"].values
    X = df.drop(columns=["df_rate"])

    # Identify continuous features
    continuous_cols = [
        c for c in X.columns
        if c not in df.filter(regex="^(player_hand|opponent_hand|surface|tourney_level|round)_").columns
    ]

    # Step 5: Scale ONLY continuous features
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[continuous_cols] = scaler.fit_transform(X[continuous_cols])

    return X, X_scaled, y


def train_elastic_net(x_scaled, y):
    # Convert to numpy arrays for sklearn
    X = x_scaled.values
    y = y.astype(float)

    # --- Time-Series Cross Validation ---
    tscv = TimeSeriesSplit(n_splits=5)

    fold = 1
    results = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # --- Train Elastic Net ---
        model = ElasticNet(alpha=0.001, l1_ratio=0.1)  # you can tune these
        model.fit(X_train, y_train)

        # --- Predict ---
        preds = model.predict(X_test)

        # --- Metrics ---
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        results.append((mae, mse, r2))

        print(f"\n===== Fold {fold} =====")
        print(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"R²:  {r2:.4f}")

        fold += 1

    # --- Average performance across folds ---
    avg_mae = np.mean([r[0] for r in results])
    avg_mse = np.mean([r[1] for r in results])
    avg_r2 = np.mean([r[2] for r in results])

    print("\n===== Overall Performance (5-Fold Time Series CV) =====")
    print(f"Average MAE: {avg_mae:.4f}")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average R²:  {avg_r2:.4f}")


def tune_elastic_net(x_scaled, y):
    X = x_scaled.values
    y = y.astype(float)

    # Time-series split
    tscv = TimeSeriesSplit(n_splits=5)

    # Hyperparameter grid
    alphas = [0.001, 0.01, 0.1, 0.5, 1, 5]
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]

    fold = 1
    results = []
    best_params = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        best_score = float("inf")
        best_alpha = None
        best_l1 = None

        # --- Grid search inside this fold ---
        for alpha in alphas:
            for l1 in l1_ratios:
                model = ElasticNet(alpha=alpha, l1_ratio=l1)
                model.fit(X_train, y_train)
                preds = model.predict(X_train)

                mse = mean_squared_error(y_train, preds)
                if mse < best_score:
                    best_score = mse
                    best_alpha = alpha
                    best_l1 = l1

        # --- Train using best hyperparameters ---
        final_model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1)
        final_model.fit(X_train, y_train)
        preds = final_model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        results.append((mae, mse, r2))
        best_params.append((best_alpha, best_l1))

        print(f"\n===== Fold {fold} =====")
        print(f"Best alpha: {best_alpha}, Best l1_ratio: {best_l1}")
        print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}")

        fold += 1

    # --- Aggregate results ---
    avg_mae = np.mean([r[0] for r in results])
    avg_mse = np.mean([r[1] for r in results])
    avg_r2 = np.mean([r[2] for r in results])
    avg_alpha = np.mean([p[0] for p in best_params])
    avg_l1 = np.mean([p[1] for p in best_params])

    print("\n===== Overall Performance After Hyperparameter Tuning =====")
    print(f"Avg Best Alpha: {avg_alpha:.4f}")
    print(f"Avg Best L1 Ratio: {avg_l1:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average R²:  {avg_r2:.4f}")

    print(results)
    print(best_params)


def train_xgboost(x, y):
    X = x.values  # ensure numpy array
    y = y.astype(float)

    tscv = TimeSeriesSplit(n_splits=5)
    fold = 1

    results = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # === XGBoost Regressor ===
        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            tree_method="hist"  # fast on CPUs
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        results.append((mae, mse, r2))

        print(f"\n===== Fold {fold} =====")
        print(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"R²:  {r2:.4f}\n")

        fold += 1

    # === Averages across folds ===
    avg_mae = np.mean([r[0] for r in results])
    avg_mse = np.mean([r[1] for r in results])
    avg_r2 = np.mean([r[2] for r in results])

    print("===== Overall XGBoost CV Performance =====")
    print(f"Average MAE: {avg_mae:.4f}")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average R²:  {avg_r2:.4f}")

    print(results)


def objective(trial, x, y):
    # --- Search Space ---
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "objective": "reg:squarederror",
        "tree_method": "hist"
    }

    tscv = TimeSeriesSplit(n_splits=5)
    r2_scores = []

    X_np = x.values
    y_np = y.astype(float)

    # --- Perform Time-Series CV ---
    for train_idx, test_idx in tscv.split(X_np):
        X_train, X_test = X_np[train_idx], X_np[test_idx]
        y_train, y_test = y_np[train_idx], y_np[test_idx]

        model = XGBRegressor(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        r2_scores.append(r2)

    return np.mean(r2_scores)  # Optuna minimizes this!


def run_optuna_tuning(x, y, n_trials):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, x, y), n_trials=n_trials)

    print("\n===== BEST PARAMETERS FOUND =====")
    print(study.best_params)
    print(f"Best R^2: {study.best_value:.4f}")

    return study.best_params


hist_df = pd.read_csv("data/processed/player_level_with_history.csv")
cols_to_extract = [
    "player_age", "player_ht", "player_hand", "player_rank",
    "opponent_age", "opponent_ht", "opponent_hand", "opponent_rank",
    "surface", "tourney_level", "tourney_datetime", "round",
    "tourney_month", "tourney_year",
    "rank_diff", "ht_diff",
    "hist_df_rate_avg", "hist_first_serve_pct_avg", "hist_first_serve_win_pct_avg",
    "hist_second_serve_win_pct_avg", "hist_ace_rate_avg", "hist_bp_pressure_avg",
    "hist_bp_clutch_avg", "hist_opp_ace_rate_avg", "hist_opp_bp_pressure_avg",
    "hist_opp_bp_clutch_avg", "df_rate"
]

# subset_df = hist_df[cols_to_extract].copy()
# print(subset_df)
# subset_df.to_csv("data/processed/model_features_with_target.csv", index=False)
subset_df = pd.read_csv("data/processed/model_features_with_target.csv")
final_model_data, final_model_data_scaled, final_model_data_response = preprocess_model_data(subset_df)

#train_elastic_net(final_model_data_scaled, final_model_data_response)
# tune_elastic_net(final_model_data_scaled, final_model_data_response)

best_params = run_optuna_tuning(final_model_data, final_model_data_response, n_trials=100)



