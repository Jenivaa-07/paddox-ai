import os
import json
import hashlib
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
import joblib
import shap

def load_data(split):
    try:
        return pd.read_csv(f"data/processed/predictive/{split}_fantasy_final.csv")
    except:
        return pd.DataFrame()

def ndcg_score_at_k(y_true, y_pred, k=5):
    """NDCG for one ranking group (one race)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return 0.0
    # PADDOX fantasy scoring can assign negative DNF/DSQ points. NDCG
    # requires non-negative relevance, so shift relevance within each race.
    y_true = y_true - np.min(y_true)
    k = min(int(k), y_true.size)
    discounts = np.log2(np.arange(2, k + 2))
    order = np.argsort(y_pred)[::-1][:k]
    ideal_order = np.argsort(y_true)[::-1][:k]
    dcg = float(np.sum(y_true[order] / discounts))
    idcg = float(np.sum(y_true[ideal_order] / discounts))
    return dcg / idcg if idcg > 0 else 0.0


def mean_race_ndcg(df, true_col, pred_col, k=5):
    """Macro-average NDCG across races; each race is a separate ranking task."""
    scores = [
        ndcg_score_at_k(group[true_col].to_numpy(), group[pred_col].to_numpy(), k)
        for _, group in df.groupby(["season", "round"], sort=False)
    ]
    return float(np.mean(scores)) if scores else 0.0


def paired_race_bootstrap_delta(df, true_col, model_col, baseline_col,
                                n_iterations=2000, seed=42):
    """95% CI for model MAE minus baseline MAE using paired race clusters."""
    groups = [group for _, group in df.groupby(["season", "round"], sort=False)]
    if len(groups) < 2:
        return None
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_iterations):
        indices = rng.integers(0, len(groups), size=len(groups))
        sample = pd.concat([groups[i] for i in indices], ignore_index=True)
        model_mae = mean_absolute_error(sample[true_col], sample[model_col])
        baseline_mae = mean_absolute_error(sample[true_col], sample[baseline_col])
        deltas.append(float(model_mae - baseline_mae))
    return [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]

def bootstrap_ci_by_race(df, true_col, pred_col, metric_fn, n_iterations=1000, ci=95):
    """Deterministic race-clustered bootstrap confidence interval."""
    races = df[['season', 'round']].drop_duplicates()
    scores = []
    
    for iteration in range(n_iterations):
        sample_races = races.sample(frac=1.0, replace=True, random_state=42 + iteration)
        sample_df = sample_races.merge(df, on=['season', 'round'], how='left')
        if len(sample_df) > 0:
            score = metric_fn(sample_df[true_col].values, sample_df[pred_col].values)
            scores.append(score)
            
    if not scores:
        return [0, 0]
        
    lower = np.percentile(scores, (100 - ci) / 2.0)
    upper = np.percentile(scores, 100 - (100 - ci) / 2.0)
    return [lower, upper]

def _mae_wrapper(y_true, y_pred): return mean_absolute_error(y_true, y_pred)
def _rmse_wrapper(y_true, y_pred): return np.sqrt(mean_squared_error(y_true, y_pred))
def _spearman_wrapper(y_true, y_pred): return spearmanr(y_true, y_pred).correlation

def train_rf_model():
    print("Training Random Forest Fantasy Predictor...")
    run_id = "run_" + os.urandom(4).hex()
    out_dir = f"artifacts/predictive/rf_fantasy/{run_id}"
    plot_dir = f"{out_dir}/plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    train_df = load_data('train')
    val_df = load_data('val')
    test_df = load_data('test')
    
    if train_df.empty or test_df.empty:
        print("Missing data splits.")
        return
        
    numeric_features = ['qualifying_position', 'rolling_avg_finish']
    categorical_features = ['constructor_id']
    target = 'fantasy_points_target'
    
    # We combine train and val for RF to use "all eligible training races" as requested
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    
    X_train = train_val_df[numeric_features + categorical_features]
    y_train = train_val_df[target].values
    
    X_test = test_df[numeric_features + categorical_features]
    y_test = test_df[target].values
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, 
                               min_samples_leaf=2, max_features='sqrt', random_state=42)
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', rf)])
    
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, f"{out_dir}/model.joblib")
    
    y_pred = pipeline.predict(X_test)
    test_df['pred_points'] = y_pred
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    spearman, _ = spearmanr(y_test, y_pred)
    
    # Ranking is evaluated within each race, not across the entire test set.
    ndcg5 = mean_race_ndcg(test_df.assign(_truth=y_test), '_truth', 'pred_points', k=5)
    ndcg10 = mean_race_ndcg(test_df.assign(_truth=y_test), '_truth', 'pred_points', k=10)

    # Explicit baselines: preserve the existing grid-position proxy and add
    # train-only constant baselines. Never estimate a baseline from test labels.
    grid_proxy = 25 - X_test['qualifying_position'].fillna(20.0).values
    train_mean = float(np.mean(y_train))
    train_median = float(np.median(y_train))
    test_df['baseline_grid_proxy'] = grid_proxy
    test_df['baseline_train_mean'] = train_mean
    test_df['baseline_train_median'] = train_median
    mae_grid_proxy = mean_absolute_error(y_test, grid_proxy)
    mae_train_mean = mean_absolute_error(y_test, np.full(len(y_test), train_mean))
    mae_train_median = mean_absolute_error(y_test, np.full(len(y_test), train_median))

    mae_ci = bootstrap_ci_by_race(test_df, target, 'pred_points', _mae_wrapper)
    rmse_ci = bootstrap_ci_by_race(test_df, target, 'pred_points', _rmse_wrapper)

    metrics = {
        "MAE": float(mae),
        "MAE_CI": mae_ci,
        "RMSE": float(rmse),
        "RMSE_CI": rmse_ci,
        "R2": float(r2),
        "Spearman": float(spearman),
        "NDCG_5": float(ndcg5),
        "NDCG_10": float(ndcg10),
        "NDCG_aggregation": "macro-average across individual races",
        "baseline_definition": {
            "grid_position_proxy": "25 - qualifying_position; heuristic proxy, not a mean baseline",
            "training_target_mean": "constant mean fantasy target calculated from train plus validation data",
            "training_target_median": "constant median fantasy target calculated from train plus validation data"
        },
        "baselines": {
            "grid_position_proxy_MAE": float(mae_grid_proxy),
            "training_target_mean_MAE": float(mae_train_mean),
            "training_target_median_MAE": float(mae_train_median),
            "model_minus_grid_proxy_MAE_95pct_paired_race_bootstrap_CI": paired_race_bootstrap_delta(
                test_df, target, 'pred_points', 'baseline_grid_proxy'
            ),
            "model_minus_training_mean_MAE_95pct_paired_race_bootstrap_CI": paired_race_bootstrap_delta(
                test_df, target, 'pred_points', 'baseline_train_mean'
            ),
            "model_minus_training_median_MAE_95pct_paired_race_bootstrap_CI": paired_race_bootstrap_delta(
                test_df, target, 'pred_points', 'baseline_train_median'
            )
        }
    }
    
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    test_df.to_csv(f"{out_dir}/heldout_predictions.csv", index=False)
    
    # Plot predicted-versus-actual
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Actual Points')
    plt.ylabel('Predicted Points')
    plt.savefig(f"{plot_dir}/predicted_vs_actual.png")
    plt.close()
    
    # Residual plot
    plt.figure()
    plt.scatter(y_pred, y_test - y_pred, alpha=0.5)
    plt.hlines(0, min(y_pred), max(y_pred), colors='r', linestyles='dashed')
    plt.xlabel('Predicted Points')
    plt.ylabel('Residuals')
    plt.savefig(f"{plot_dir}/residuals.png")
    plt.close()
    
    # Model-vs-Baseline plot
    plt.figure()
    plt.bar(['Random Forest', 'Grid proxy', 'Train mean', 'Train median'],
            [mae, mae_grid_proxy, mae_train_mean, mae_train_median])
    plt.savefig(f"{plot_dir}/model_vs_baseline.png")
    plt.close()
    
    print("Calculating permutation importance...")
    result = permutation_importance(pipeline, X_test, y_test, n_repeats=10, random_state=42, n_jobs=1)
    
    plt.figure()
    sorted_idx = result.importances_mean.argsort()
    features = np.array(numeric_features + categorical_features)[sorted_idx]
    plt.boxplot(result.importances[sorted_idx].T, vert=False, tick_labels=features)
    plt.title("Permutation Importance")
    plt.savefig(f"{plot_dir}/permutation_importance.png")
    plt.close()
    
    print("Generating SHAP summary (sample)...")
    X_test_transformed = preprocessor.transform(X_test)
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test_transformed[:200])
    
    plt.figure()
    shap.summary_plot(shap_values, X_test_transformed[:200], show=False)
    plt.savefig(f"{plot_dir}/shap_summary.png")
    plt.close()
    
    import shutil
    shutil.copy("config/fantasy_scoring_v1.yaml", f"{out_dir}/fantasy_scoring_v1.yaml")
    
    feature_schema = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "description": "Tabular features for Random Forest Fantasy Predictor"
    }
    with open(f"{out_dir}/feature_schema.json", "w") as f:
        json.dump(feature_schema, f, indent=4)

    split_files = {}
    for split_name, split_frame in (("train", train_df), ("val", val_df), ("test", test_df)):
        split_path = f"data/processed/predictive/{split_name}_fantasy_final.csv"
        with open(split_path, "rb") as split_file:
            split_hash = hashlib.sha256(split_file.read()).hexdigest()
        split_files[split_name] = {
            "path": split_path,
            "sha256": split_hash,
            "rows": int(len(split_frame))
        }
    run_manifest = {
        "run_id": run_id,
        "random_seed": 42,
        "training_rows_including_validation": int(len(train_val_df)),
        "test_rows": int(len(test_df)),
        "split_files": split_files,
        "baseline_definitions": {
            "grid_position_proxy": "25 - qualifying_position",
            "training_target_mean": train_mean,
            "training_target_median": train_median
        },
        "ranking_metric_aggregation": "macro-average of per-race NDCG",
        "metrics_file": "metrics.json",
        "predictions_file": "heldout_predictions.csv",
        "model_file": "model.joblib"
    }
    with open(f"{out_dir}/run_manifest.json", "w") as f:
        json.dump(run_manifest, f, indent=4)
    for manifest_path in (
        "data/processed/predictive/fantasy_split_manifest.json",
        "data/manifests/predictive_dataset_manifest.json"
    ):
        if os.path.isfile(manifest_path):
            shutil.copy2(manifest_path, out_dir)

    with open(f"{out_dir}/training_log.txt", "w") as f:
        f.write("Training Random Forest Fantasy Predictor...\n")
        f.write("Calculating permutation importance...\n")
        f.write("Generating SHAP summary...\n")
    
    with open("artifacts/predictive/rf_fantasy/current_model.json", "w") as f:
        json.dump({"run_id": run_id}, f)

if __name__ == "__main__":
    train_rf_model()
