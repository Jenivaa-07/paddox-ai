import os
import json
import time
import uuid
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset

# Constants
MODEL_NAME = "distilbert-base-uncased"
PADDOX_MODEL_DESC = "PADDOX motorsport-domain-adapted DistilBERT sentiment classifier"
LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
LABEL_TO_ID = {l: i for i, l in enumerate(LABELS)}
ID_TO_LABEL = {i: l for i, l in enumerate(LABELS)}

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "sentiment")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "sentiment")
ARTIFACTS_PATH = os.path.join(BASE_DIR, "artifacts", "sentiment")
REPORTS_PATH = os.path.join(BASE_DIR, "reports", "sentiment")

def ensure_dirs():
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    os.makedirs(ARTIFACTS_PATH, exist_ok=True)
    os.makedirs(REPORTS_PATH, exist_ok=True)

class LossLoggingCallback(TrainerCallback):
    def __init__(self):
        self.log_history = []
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            self.log_history.append(logs)

def generate_smoke_test_data():
    """Generates a 10-row dataset only for code-path smoke testing."""
    data = [
        {"text": "The new aero package on the Mercedes is terrible.", "label": "NEGATIVE", "source": "fanhub", "timestamp": "2026-07-26T10:00:00Z", "source_id": "anon_1", "annotator": "auto"},
        {"text": "Max Verstappen's pace in sector 2 was absolutely stunning!", "label": "POSITIVE", "source": "fanhub", "timestamp": "2026-07-26T10:01:00Z", "source_id": "anon_2", "annotator": "auto"},
        {"text": "The pit stop took 2.4 seconds.", "label": "NEUTRAL", "source": "fanhub", "timestamp": "2026-07-26T10:02:00Z", "source_id": "anon_3", "annotator": "auto"},
        {"text": "Ferrari strategy ruins the race again.", "label": "NEGATIVE", "source": "fanhub", "timestamp": "2026-07-26T10:03:00Z", "source_id": "anon_4", "annotator": "auto"},
        {"text": "What a masterclass by Lando Norris.", "label": "POSITIVE", "source": "fanhub", "timestamp": "2026-07-26T10:04:00Z", "source_id": "anon_5", "annotator": "auto"},
        {"text": "Tyre degradation is higher than expected.", "label": "NEUTRAL", "source": "fanhub", "timestamp": "2026-07-26T10:05:00Z", "source_id": "anon_6", "annotator": "auto"},
        {"text": "Worst steward decision ever.", "label": "NEGATIVE", "source": "fanhub", "timestamp": "2026-07-26T10:06:00Z", "source_id": "anon_7", "annotator": "auto"},
        {"text": "Incredible defending from Fernando Alonso!", "label": "POSITIVE", "source": "fanhub", "timestamp": "2026-07-26T10:07:00Z", "source_id": "anon_8", "annotator": "auto"},
        {"text": "Rain is expected in 15 minutes.", "label": "NEUTRAL", "source": "fanhub", "timestamp": "2026-07-26T10:08:00Z", "source_id": "anon_9", "annotator": "auto"},
        {"text": "The race starts at 15:00 local time.", "label": "NEUTRAL", "source": "fanhub", "timestamp": "2026-07-26T10:09:00Z", "source_id": "anon_10", "annotator": "auto"},
        # Added duplicate for testing deduplication
        {"text": "The race starts at 15:00 local time.", "label": "NEUTRAL", "source": "fanhub", "timestamp": "2026-07-26T10:10:00Z", "source_id": "anon_10", "annotator": "auto"}
    ]
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(RAW_DATA_PATH, "smoke_test.csv"), index=False)
    return df

def clean_and_prepare_data(df):
    # Deduplicate
    initial_len = len(df)
    df = df.drop_duplicates(subset=['text'], keep='first')
    df = df.dropna(subset=['text', 'label'])
    
    # Map labels
    df['label_id'] = df['label'].map(LABEL_TO_ID)
    df = df.dropna(subset=['label_id'])
    df['label_id'] = df['label_id'].astype(int)
    
    print(f"Data cleaned. Deduplicated from {initial_len} to {len(df)} rows.")
    return df

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None, labels=[0, 1, 2], zero_division=0)
    macro_f1 = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)[2]
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
    
    metrics = {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'precision_NEGATIVE': precision[0],
        'recall_NEGATIVE': recall[0],
        'f1_NEGATIVE': f1[0],
        'precision_NEUTRAL': precision[1],
        'recall_NEUTRAL': recall[1],
        'f1_NEUTRAL': f1[1],
        'precision_POSITIVE': precision[2],
        'recall_POSITIVE': recall[2],
        'f1_POSITIVE': f1[2],
    }
    
    return metrics, cm.tolist()

class CustomTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Calculate loss with class weights
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def train(data_path=None, is_smoke_test=False):
    ensure_dirs()
    run_id = f"run_{uuid.uuid4().hex[:8]}_{int(time.time())}"
    run_artifact_dir = os.path.join(ARTIFACTS_PATH, run_id)
    run_report_dir = os.path.join(REPORTS_PATH, run_id)
    os.makedirs(run_artifact_dir, exist_ok=True)
    os.makedirs(run_report_dir, exist_ok=True)
    
    print(f"Starting Training Run: {run_id}")
    
    if is_smoke_test or data_path is None:
        print("Using SMOKE TEST dataset.")
        df = generate_smoke_test_data()
    else:
        df = pd.read_csv(data_path)
        
    df = clean_and_prepare_data(df)
    
    # Save cleaned dataset
    cleaned_path = os.path.join(PROCESSED_DATA_PATH, f"{run_id}_cleaned.csv")
    df.to_csv(cleaned_path, index=False)
    
    # Split: 80% train, 10% val, 10% test
    # If it's a smoke test with 10 items, we handle tiny sizes
    if len(df) < 15:
        train_df = df
        val_df = df
        test_df = df
    else:
        train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label_id'], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label_id'], random_state=42)
        
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    
    def tokenize_func(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
        
    columns_to_remove = ["text", "label", "source", "timestamp", "source_id", "annotator"]
    train_dataset = Dataset.from_pandas(train_df).map(tokenize_func, batched=True).rename_column("label_id", "labels").remove_columns(columns_to_remove)
    val_dataset = Dataset.from_pandas(val_df).map(tokenize_func, batched=True).rename_column("label_id", "labels").remove_columns(columns_to_remove)
    test_dataset = Dataset.from_pandas(test_df).map(tokenize_func, batched=True).rename_column("label_id", "labels").remove_columns(columns_to_remove)
    
    # Class weights for imbalance
    classes = np.array([0, 1, 2])
    train_labels = train_df['label_id'].values
    if len(np.unique(train_labels)) > 1:
        cw = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
        # map back to all 3 classes
        class_weights = torch.tensor([cw[np.where(np.unique(train_labels) == c)[0][0]] if c in np.unique(train_labels) else 1.0 for c in classes], dtype=torch.float32)
    else:
        class_weights = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = class_weights.to(device)
    
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID
    )
    
    training_args = TrainingArguments(
        output_dir=run_artifact_dir,
        num_train_epochs=1 if is_smoke_test else 3,
        per_device_train_batch_size=2 if is_smoke_test else 16,
        per_device_eval_batch_size=2 if is_smoke_test else 16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(run_report_dir, "logs"),
        logging_steps=1 if is_smoke_test else 10,
        seed=42, # Reproducible random seed
        use_cpu=not torch.cuda.is_available()
    )
    
    loss_callback = LossLoggingCallback()
    
    # Wrap standard compute_metrics to capture cm
    global latest_cm
    latest_cm = None
    
    def wrapped_compute_metrics(eval_pred):
        global latest_cm
        mets, cm = compute_metrics(eval_pred)
        latest_cm = cm
        return mets

    trainer = CustomTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=wrapped_compute_metrics,
        callbacks=[loss_callback]
    )
    
    print("Beginning Training...")
    trainer.train()
    
    print("Running Final Evaluation on Test Set...")
    eval_results = trainer.evaluate(eval_dataset=test_dataset)
    
    # Save Model
    model.save_pretrained(run_artifact_dir)
    tokenizer.save_pretrained(run_artifact_dir)
    
    # Save Metrics & Logs
    metrics_path = os.path.join(run_report_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(eval_results, f, indent=4)
        
    cm_path = os.path.join(run_report_dir, "confusion_matrix.json")
    with open(cm_path, "w") as f:
        json.dump(latest_cm, f, indent=4)
        
    # Extract loss history
    train_loss = [log.get('loss') for log in loss_callback.log_history if 'loss' in log]
    eval_loss = [log.get('eval_loss') for log in loss_callback.log_history if 'eval_loss' in log]
    
    # Plot Graphs
    plt.figure(figsize=(10,5))
    if train_loss:
        plt.plot(train_loss, label='Training Loss')
    if eval_loss:
        # Eval occurs less frequently, this is a rough alignment for the plot
        eval_x = np.linspace(0, len(train_loss)-1, len(eval_loss)) if train_loss else range(len(eval_loss))
        plt.plot(eval_x, eval_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Logging Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(run_report_dir, "loss_graph.png"))
    
    # Manifest
    manifest = {
        "run_id": run_id,
        "run_type": "smoke_test" if is_smoke_test else "production",
        "dataset_type": "synthetic_test_fixture" if is_smoke_test else "real_annotated",
        "eligible_for_paper": not is_smoke_test,
        "eligible_for_production": not is_smoke_test,
        "model_description": PADDOX_MODEL_DESC,
        "timestamp": time.time(),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df)
    }
    with open(os.path.join(run_report_dir, "dataset_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=4)
        
    print(f"Training Complete. Artifacts saved to {run_artifact_dir}")
    print(f"Reports saved to {run_report_dir}")

if __name__ == "__main__":
    train(is_smoke_test=True)
