import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss
from scipy.stats import spearmanr
from lstm_race_model import LSTMRacePredictor

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    
    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))
    
    nonzero = bin_total != 0
    if not np.any(nonzero):
        return 0.0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    
    ece = np.sum(np.abs(prob_true - prob_pred) * (bin_total[nonzero] / len(y_true)))
    return ece

def bootstrap_ci_by_race(df, true_col, pred_col, metric_fn, n_iterations=1000, ci=95):
    """ Race-clustered bootstrap confidence interval """
    races = df[['season', 'round']].drop_duplicates()
    scores = []
    
    for _ in range(n_iterations):
        sample_races = races.sample(frac=1.0, replace=True)
        # Merge to get all rows for sampled races
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

def plot_checkpoint_perf(df, out_dir):
    checkpoints = df['checkpoint_type'].unique()
    chk_maes = []
    chk_names = []
    for chk in checkpoints:
        chk_df = df[df['checkpoint_type'] == chk]
        mae = mean_absolute_error(chk_df['final_position'], chk_df['pred_position'])
        chk_maes.append(mae)
        chk_names.append(chk)
        
    plt.figure()
    sns.barplot(x=chk_names, y=chk_maes)
    plt.title('MAE by Checkpoint')
    plt.ylabel('MAE')
    plt.savefig(f"{out_dir}/checkpoint_mae.png")
    plt.close()

def plot_calibration(df, out_dir, n_bins=10):
    y_true = df['top10']
    y_prob = df['pred_top10_prob']
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    
    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))
    
    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    
    plt.figure()
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.plot(prob_pred, prob_true, "s-", label="Model")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.savefig(f"{out_dir}/calibration_curve.png")
    plt.close()

def load_data(split):
    try:
        df = pd.read_csv(f"data/processed/predictive/{split}_race_seq.csv")
    except:
        return pd.DataFrame() # Return empty if doesn't exist
    import ast
    df['seq_laps'] = df['seq_laps'].apply(ast.literal_eval)
    df['mask'] = df['mask'].apply(ast.literal_eval)
    # Add normalized position target
    df['field_size'] = df['field_size'].fillna(20.0)
    df['normalized_position'] = (df['final_position'] - 1) / (df['field_size'] - 1)
    # clip just in case of weird data
    df['normalized_position'] = df['normalized_position'].clip(0.0, 1.0)
    return df

def train_model():
    print("Training LSTM Race Predictor...")
    run_id = "run_" + os.urandom(4).hex()
    out_dir = f"artifacts/predictive/lstm_race/{run_id}"
    plot_dir = f"{out_dir}/plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    train_df = load_data('train')
    val_df = load_data('val')
    test_df = load_data('test')
    
    if train_df.empty or val_df.empty or test_df.empty:
        print("Missing data splits.")
        return
    
    features = ['grid_position', 'rolling_avg_finish']
    scaler = StandardScaler()
    train_feat = scaler.fit_transform(train_df[features].fillna(20.0))
    val_feat = scaler.transform(val_df[features].fillna(20.0))
    test_feat = scaler.transform(test_df[features].fillna(20.0))
    
    from utils.scaling import scale_lap_time
    def to_tensors(df, feat):
        X = []
        for i in range(len(df)):
            laps = df.iloc[i]['seq_laps']
            f = feat[i]
            # Scale laps using the shared config
            seq = [[scale_lap_time(laps[j]), f[0], f[1]] for j in range(10)]
            X.append(seq)
        y_reg = df['normalized_position'].values
        y_cls = df['top10'].values
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y_reg, dtype=torch.float32).unsqueeze(1), torch.tensor(y_cls, dtype=torch.float32).unsqueeze(1)

    X_train, y_reg_train, y_cls_train = to_tensors(train_df, train_feat)
    X_val, y_reg_val, y_cls_val = to_tensors(val_df, val_feat)
    X_test, y_reg_test, y_cls_test = to_tensors(test_df, test_feat)
    
    train_loader = DataLoader(TensorDataset(X_train, y_reg_train, y_cls_train), batch_size=32, shuffle=True)
    
    model = LSTMRacePredictor(input_dim=3, hidden_dim=64, num_layers=2)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    reg_criterion = nn.MSELoss()
    cls_criterion = nn.BCEWithLogitsLoss()
    reg_weight = 0.5
    cls_weight = 0.5
    
    best_val_loss = float('inf')
    epochs = 20
    train_losses = []
    val_losses = []
    train_log_lines = []
    
    for epoch in range(epochs):
        model.train()
        ep_loss = 0
        for x, yr, yc in train_loader:
            optimizer.zero_grad()
            pred_r, pred_c = model(x)
            loss_r = reg_criterion(pred_r, yr)
            loss_c = cls_criterion(pred_c, yc)
            loss = reg_weight * loss_r + cls_weight * loss_c
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            
        model.eval()
        with torch.no_grad():
            pred_r_val, pred_c_val = model(X_val)
            val_loss_r = reg_criterion(pred_r_val, y_reg_val)
            val_loss_c = cls_criterion(pred_c_val, y_cls_val)
            val_loss = reg_weight * val_loss_r + cls_weight * val_loss_c
            
        train_losses.append(ep_loss / len(train_loader))
        val_losses.append(val_loss.item())
        
        log_line = f"Epoch {epoch+1}/{epochs} | Train Loss: {ep_loss/len(train_loader):.4f} | Val Loss: {val_loss.item():.4f} (Reg: {val_loss_r.item():.4f}, Cls: {val_loss_c.item():.4f})"
        print(log_line)
        train_log_lines.append(log_line)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{out_dir}/model.pt")
            
    with open(f"{out_dir}/training_log.txt", "w") as f:
        f.write("\n".join(train_log_lines))
            
    # Plot loss
    plt.figure()
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.legend()
    plt.savefig(f"{plot_dir}/loss_curve.png")
    plt.close()
    
    print("Training Complete. Evaluating on Test Split...")
    model.load_state_dict(torch.load(f"{out_dir}/model.pt"))
    model.eval()
    
    # Calculate threshold on validation set to avoid test set data leakage
    from sklearn.metrics import f1_score
    with torch.no_grad():
        _, best_val_c = model(X_val)
        val_probs = torch.sigmoid(best_val_c).numpy().flatten()
    best_thresh = 0.5
    best_f1 = 0
    for th in np.linspace(0.1, 0.9, 9):
        f1 = f1_score(y_cls_val.numpy().flatten(), (val_probs >= th).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = th
    print(f"Selected classification threshold from validation set: {best_thresh:.2f}")

    with torch.no_grad():
        pred_r_test, pred_c_test = model(X_test)
        
    pred_r_np = pred_r_test.numpy().flatten()
    pred_c_np = torch.sigmoid(pred_c_test).numpy().flatten()
    
    # Inverse transform normalized prediction
    # predicted_position = 1 + normalized_prediction × (field_size - 1)
    test_df['pred_normalized'] = pred_r_np
    test_df['pred_position'] = 1 + test_df['pred_normalized'] * (test_df['field_size'] - 1)
    test_df['pred_top10_prob'] = pred_c_np
    
    # Baseline comparison (Linear Regression on tabular)
    lr = LinearRegression()
    lr.fit(train_feat, train_df['final_position'])
    test_df['pred_baseline_tabular'] = lr.predict(test_feat)
    
    mae = mean_absolute_error(test_df['final_position'], test_df['pred_position'])
    rmse = np.sqrt(mean_squared_error(test_df['final_position'], test_df['pred_position']))
    spear = spearmanr(test_df['final_position'], test_df['pred_position']).correlation
    
    mae_ci = bootstrap_ci_by_race(test_df, 'final_position', 'pred_position', _mae_wrapper)
    rmse_ci = bootstrap_ci_by_race(test_df, 'final_position', 'pred_position', _rmse_wrapper)
    
    from sklearn.metrics import confusion_matrix
    
    # Classification metrics thresholding using the validation-derived threshold
    pred_cls = (test_df['pred_top10_prob'] >= best_thresh).astype(int)
    prec = precision_score(test_df['top10'], pred_cls, zero_division=0)
    rec = recall_score(test_df['top10'], pred_cls, zero_division=0)
    f1 = f1_score(test_df['top10'], pred_cls, zero_division=0)
    roc_auc = roc_auc_score(test_df['top10'], test_df['pred_top10_prob'])
    brier = brier_score_loss(test_df['top10'], test_df['pred_top10_prob'])
    ece = expected_calibration_error(test_df['top10'].values, test_df['pred_top10_prob'].values)
    cm = confusion_matrix(test_df['top10'], pred_cls).tolist()
    
    metrics = {
        "regression": {
            "MAE": float(mae),
            "MAE_CI": mae_ci,
            "RMSE": float(rmse),
            "RMSE_CI": rmse_ci,
            "Spearman": float(spear)
        },
        "classification": {
            "Precision": float(prec),
            "Recall": float(rec),
            "F1": float(f1),
            "ROC_AUC": float(roc_auc),
            "Brier_Score": float(brier),
            "ECE": float(ece),
            "Confusion_Matrix": cm
        },
        "baselines": {
            "Grid_MAE": float(mean_absolute_error(test_df['final_position'], test_df['grid_position'])),
            "Rolling_Avg_MAE": float(mean_absolute_error(test_df['final_position'], test_df['rolling_avg_finish'])),
            "Tabular_LR_MAE": float(mean_absolute_error(test_df['final_position'], test_df['pred_baseline_tabular']))
        }
    }
    
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    test_df.to_csv(f"{out_dir}/heldout_predictions.csv", index=False)
    plot_checkpoint_perf(test_df, plot_dir)
    plot_calibration(test_df, plot_dir)
    
    # Model config
    from utils.scaling import SCALING_CONFIG
    config = {"input_dim": 3, "hidden_dim": 64, "num_layers": 2}
    config.update(SCALING_CONFIG)
    with open(f"{out_dir}/model_config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    feature_schema = {
        "features": features,
        "input_format": "sequence_length_10",
        "description": "Left-padded 10-timestep sequence of [lap_time_ms, grid_position, rolling_avg_finish]",
        "scaling": SCALING_CONFIG
    }
    with open(f"{out_dir}/feature_schema.json", "w") as f:
        json.dump(feature_schema, f, indent=4)
        
    # Generate current_model.json
    with open("artifacts/predictive/lstm_race/current_model.json", "w") as f:
        json.dump({"run_id": run_id}, f)
        
if __name__ == "__main__":
    train_model()
