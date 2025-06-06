import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from transformers import DistilBertTokenizer
from model import ActionClassifier
from dataset import ADCDataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os

# --- Custom Collate Function ---
def collate_fn(batch):
    batched_data = {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'action_label': torch.stack([item['action_label'] for item in batch]),
        'query_labels': torch.stack([item['query_labels'] for item in batch]),
        'touch_coords': torch.stack([item['touch_coords'] for item in batch]),
        'lift_coords': torch.stack([item['lift_coords'] for item in batch]),
        'bounding_boxes': [item['bounding_boxes'] for item in batch]  # Keep as list
    }
    return batched_data

# --- Config ---
config = {
    "batch_size": 32,
    "epochs": 10,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "num_workers": 4,
    "freeze_bert": False,
    "save_path": "./saved_models",
    "train_path": "short_train_json_dataset",
    "val_path": "short_val_json_dataset",
    "test_path": "short_test_json_dataset",
}

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load Data ---
tokenizer = DistilBertTokenizer.from_pretrained("models/distilbert-base-uncased")
train_dataset = ADCDataset(config["train_path"], tokenizer)
val_dataset = ADCDataset(config["val_path"], tokenizer)
test_dataset = ADCDataset(config["test_path"], tokenizer)

# --- DataLoaders with Custom Collate ---
train_loader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=config["num_workers"],
    collate_fn=collate_fn  # <-- KEY CHANGE
)
val_loader = DataLoader(
    val_dataset,
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    collate_fn=collate_fn
)
test_loader = DataLoader(
    test_dataset,
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    collate_fn=collate_fn
)

# --- Model, Optimizer, Loss ---
model = ActionClassifier(num_classes=11, freeze_bert=config["freeze_bert"]).to(device)
optimizer = AdamW(
    model.parameters(),
    lr=config["learning_rate"],
    weight_decay=config["weight_decay"]
)
loss_fn = CrossEntropyLoss()

# --- Metrics ---
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    return {"accuracy": acc, "f1": f1}

# --- Training Loop ---
def train():
    best_val_acc = 0.0
    best_test_acc = 0.0

    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["action_label"].to(device) - 1  # Convert 1-11 to 0-10

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        val_metrics = evaluate(val_loader)
        test_metrics = evaluate(test_loader)

        print(
            f"Epoch {epoch + 1}/{config['epochs']} | "
            f"Train Loss: {train_loss / len(train_loader):.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Test Acc: {test_metrics['accuracy']:.4f}"
        )

        # Save best model
        if test_metrics["accuracy"] > best_test_acc or (
            test_metrics["accuracy"] == best_test_acc and val_metrics["accuracy"] > best_val_acc
        ):
            best_test_acc = test_metrics["accuracy"]
            best_val_acc = val_metrics["accuracy"]
            os.makedirs(config["save_path"], exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(config["save_path"], "best_model.pt")
            )
            print(f"Saved new best model with Test Acc: {best_test_acc:.4f}")

# --- Evaluation ---
def evaluate(loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["action_label"].to(device) - 1

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    return compute_metrics(y_true, y_pred)

if __name__ == "__main__":
    train()
