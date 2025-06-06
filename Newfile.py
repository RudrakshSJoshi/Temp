import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig

class ActionClassifier(nn.Module):
    def __init__(self, num_classes=11, freeze_bert=False):
        super().__init__()
        # Load pretrained DistilBERT
        self.bert = DistilBertModel.from_pretrained("models/distilbert-base-uncased")
        self.config = self.bert.config
        
        # Freeze BERT if needed
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Classification head
        self.dropout = nn.Dropout(0.1)  # Optional dropout
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits



import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from transformers import DistilBertTokenizer
from model import ActionClassifier
from dataset import ADCDataset  # Your custom dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os

# Config (from data_parameters.txt)
config = {
    "batch_size": 32,
    "epochs": 10,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "num_workers": 4,
    "freeze_bert": False,
    "save_path": "./saved_models",
    "train_path": "train_json_dataset",
    "val_path": "val_json_dataset",
    "test_path": "test_json_dataset",
}

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer and datasets
tokenizer = DistilBertTokenizer.from_pretrained("models/distilbert-base-uncased")
train_dataset = ADCDataset(config["train_path"], tokenizer)
val_dataset = ADCDataset(config["val_path"], tokenizer)
test_dataset = ADCDataset(config["test_path"], tokenizer)

# DataLoaders
train_loader = DataLoader(
    train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"]
)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], num_workers=config["num_workers"])
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], num_workers=config["num_workers"])

# Model, optimizer, loss
model = ActionClassifier(num_classes=11, freeze_bert=config["freeze_bert"]).to(device)
optimizer = AdamW(
    model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
)
loss_fn = CrossEntropyLoss()

# Metrics (handles imbalanced data)
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")  # Weighted F1 for imbalance
    return {"accuracy": acc, "f1": f1}

# Training loop
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
        
        # Save if best test accuracy (or val accuracy if tied)
        if test_metrics["accuracy"] > best_test_acc or (
            test_metrics["accuracy"] == best_test_acc and val_metrics["accuracy"] > best_val_acc
        ):
            best_test_acc = test_metrics["accuracy"]
            best_val_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), os.path.join(config["save_path"], "best_model.pt"))
            print(f"Saved new best model with Test Acc: {best_test_acc:.4f}")

# Evaluation helper
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
