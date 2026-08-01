import os
import json
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
    order = np.argsort(y_pred)[::-1][:k]
    dcg = sum(y_true[i] / np.log2(idx + 2) for idx, i in enumerate(order))
    
    ideal_order = np.argsort(y_true)[::-1][:k]
    idcg = sum(y_true[i] / np.log2(idx + 2) for idx, i in enumerate(ideal_order))
    
    return dcg / idcg if idcg > 0 else 0.0

def bootstrap_ci_by_race(df, true_col, pred_col, metric_fn, n_iterations=1000, ci=95):
    races = df[['season', 'round']].drop_duplicates()
    scores = []
    
    for _ in range(n_iterations):
        sample_races = races.sample(frac=1.0, replace=True)
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
    
    ndcg5 = ndcg_score_at_k(y_test, y_pred, k=5)
    ndcg10 = ndcg_score_at_k(y_test, y_pred, k=10)
    
    # Baseline: linear mapping of grid position to fantasy points (inverse)
    # Simple proxy: 25 - grid
    base_pred = 25 - X_test['qualifying_position'].fillna(20.0).values
    test_df['baseline_points'] = base_pred
    mae_base = mean_absolute_error(y_test, base_pred)
    
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
        "baseline_MAE": float(mae_base)
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
    plt.bar(['Model MAE', 'Baseline MAE'], [mae, mae_base])
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
        
    with open(f"{out_dir}/training_log.txt", "w") as f:
        f.write("Training Random Forest Fantasy Predictor...\n")
        f.write("Calculating permutation importance...\n")
        f.write("Generating SHAP summary...\n")
    
    with open("artifacts/predictive/rf_fantasy/current_model.json", "w") as f:
        json.dump({"run_id": run_id}, f)

if __name__ == "__main__":
    train_rf_model()
