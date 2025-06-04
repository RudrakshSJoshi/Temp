import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from transformers import DistilBertModel, DistilBertTokenizer, DistilBertConfig
from pydantic import BaseModel
import glob
from tqdm import tqdm
import datetime

def collate_fn(batch):
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'action_label': torch.stack([item['action_label'] for item in batch]),
        'query_labels': torch.stack([item['query_labels'] for item in batch]),
        'touch_coords': torch.stack([item['touch_coords'] for item in batch]),
        'lift_coords': torch.stack([item['lift_coords'] for item in batch]),
        'bounding_boxes': [item['bounding_boxes'] for item in batch]
    }

class Results(BaseModel):
    action_type: int
    type_action: str
    touch_coords: List[float]
    lift_coords: List[float]

class DataEntry(BaseModel):
    prompt: str
    coordinates: List[List[float]]
    results: Results

class ADC_Agent(nn.Module):
    def __init__(self, 
                model_path: str = "models/distilbert-base-uncased",
                lstm_hidden_size: int = 256,
                max_seq_length: int = 512,
                freeze_bert: bool = False,
                dropout: float = 0.1):
        super().__init__()
        
        if not os.path.exists(model_path):
            raise ValueError(f"Model path {model_path} does not exist")
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        bert_config = DistilBertConfig.from_pretrained(model_path)
        self.dropout = nn.Dropout(dropout)
        self.bert = DistilBertModel.from_pretrained(model_path, config=bert_config)
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        bert_output_size = bert_config.hidden_size
        self.max_seq_length = max_seq_length
        self.lstm_hidden_size = lstm_hidden_size

        # Separate projection heads
        self.action_proj = nn.Sequential(
            nn.Linear(bert_output_size, bert_output_size),
            nn.GELU(),
            nn.LayerNorm(bert_output_size)
        )
        self.query_proj = nn.Sequential(
            nn.Linear(bert_output_size, bert_output_size),
            nn.GELU(),
            nn.LayerNorm(bert_output_size)
        )
        self.coord_proj = nn.Sequential(
            nn.Linear(bert_output_size, bert_output_size),
            nn.GELU(),
            nn.LayerNorm(bert_output_size)
        )

        # Action classification
        self.action_codes = {3, 4, 5, 6, 7, 10, 11}
        self.num_actions = max(self.action_codes) + 1
        self.action_head = nn.Linear(bert_output_size, self.num_actions)
        
        # Query generation
        self.query_decoder = nn.LSTM(
            input_size=bert_output_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )
        self.query_output = nn.Linear(lstm_hidden_size, bert_config.vocab_size)
        
        # Coordinate prediction (simplified to MLP)
        self.coord_head = nn.Sequential(
            nn.Linear(bert_output_size, lstm_hidden_size),
            nn.ReLU(),
            nn.Linear(lstm_hidden_size, 4),  # x1,y1,x2,y2
            nn.Sigmoid()
        )
        
        # Constants
        self.gesture_actions = {4, 5, 6, 7, 10, 11}
        self.search_action = 3
        self.max_query_length = 32

    def forward_training(self, input_text: Union[str, Dict[str, torch.Tensor]], device=None):
        if device is None:
            device = next(self.parameters()).device
        
        if isinstance(input_text, str):
            inputs = self._preprocess_text(input_text)
            inputs = {k: v.to(device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(device) for k, v in input_text.items()}
        
        # BERT processing
        bert_output = self.bert(**inputs)
        pooled_output = bert_output.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        
        # Action prediction
        action_feats = self.action_proj(pooled_output)
        action_logits = self.action_head(action_feats)
        action_probs = F.softmax(action_logits, dim=-1)
        action_code = torch.argmax(action_probs, dim=-1)
        
        outputs = {
            'code': action_code,
            'action_logits': action_logits,
            'query_logits': None,
            'coordinates': None
        }
        
        # Generate outputs only for relevant actions
        if action_code.item() == self.search_action:
            outputs['query_logits'] = self._generate_search_query_training(
                self.query_proj(pooled_output), 
                device=device
            )
        elif action_code.item() in self.gesture_actions:
            outputs['coordinates'] = self.coord_head(self.coord_proj(pooled_output))
            
        return outputs

    def _generate_search_query_training(self, context, device):
        batch_size = context.size(0)
        hidden = context.unsqueeze(0)
        cell = torch.zeros_like(hidden)
        input_seq = context.unsqueeze(1)
        
        all_logits = []
        for _ in range(self.max_query_length):
            lstm_out, (hidden, cell) = self.query_decoder(input_seq, (hidden, cell))
            logits = self.query_output(lstm_out.squeeze(1))
            all_logits.append(logits)
            
            # Teacher forcing handled in loss, just continue sequence
            next_token = torch.argmax(logits, dim=-1)
            token_emb = self.bert.embeddings.word_embeddings(next_token)
            input_seq = token_emb.unsqueeze(1)
            
        return torch.stack(all_logits, dim=1)

    def _preprocess_text(self, text):
        return self.tokenizer(
            text,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

class ADCDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.data = []
        self.tokenizer = tokenizer
        self.load_data(data_path)
    
    def load_data(self, data_path):
        json_files = glob.glob(os.path.join(data_path, "*.json"))
        
        for json_file in tqdm(json_files, desc=f"Loading from {data_path}"):
            with open(json_file, 'r', encoding="utf-8") as f:
                file_data = json.load(f)
                for key, sequence in file_data.items():
                    for step in sequence:
                        entry = DataEntry(**step)
                        self.data.append(entry)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        entry = self.data[idx]
        
        encoded = self.tokenizer(
            entry.prompt,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        if entry.results.type_action.strip():
            target_encoded = self.tokenizer(
                entry.results.type_action,
                max_length=32,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            target_tokens = target_encoded['input_ids'].squeeze(0)
        else:
            target_tokens = torch.zeros(32, dtype=torch.long)
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'action_label': torch.tensor(entry.results.action_type, dtype=torch.long),
            'query_labels': target_tokens,
            'touch_coords': torch.tensor(entry.results.touch_coords, dtype=torch.float32),
            'lift_coords': torch.tensor(entry.results.lift_coords, dtype=torch.float32),
            'bounding_boxes': entry.coordinates
        }

class ADCLoss(nn.Module):
    def __init__(self, action_weight=1.0, query_weight=1.0, coord_weight=2.0):
        super().__init__()
        self.action_loss_fn = nn.CrossEntropyLoss()
        self.query_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        self.coord_loss_fn = nn.HuberLoss()
        self.weights = {
            'action': action_weight,
            'query': query_weight,
            'coord': coord_weight
        }

    def forward(self, outputs, targets):
        losses = {}
        
        # Action loss
        losses['action'] = self.action_loss_fn(
            outputs['action_logits'], 
            targets['action_labels']
        )
        
        # Query loss (only for search actions)
        if outputs['query_logits'] is not None:
            query_logits = outputs['query_logits'].view(-1, outputs['query_logits'].size(-1))
            query_targets = targets['query_labels'].view(-1)
            losses['query'] = self.query_loss_fn(query_logits, query_targets)
        else:
            losses['query'] = torch.tensor(0.0)
        
        # Coordinate loss (only for gesture actions)
        if outputs['coordinates'] is not None:
            losses['coord'] = self.coord_loss_fn(
                outputs['coordinates'],
                torch.cat([targets['touch_coords'], targets['lift_coords']], dim=1)
            )
        else:
            losses['coord'] = torch.tensor(0.0)
        
        # Weighted total
        total_loss = sum(self.weights[k] * losses[k] for k in losses)
        
        return {
            'total_loss': total_loss,
            **losses
        }

class ADCTrainer:
    def __init__(self, config_path="data_parameters.txt"):
        self.config = self.load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = ADC_Agent(
            model_path=self.config['model_path'],
            lstm_hidden_size=self.config['lstm_hidden_size'],
            max_seq_length=self.config['max_seq_length'],
            freeze_bert=self.config['freeze_bert'],
            dropout=self.config['dropout']
        ).to(self.device)
        
        self.loss_fn = ADCLoss(
            action_weight=self.config['action_weight'],
            query_weight=self.config['query_weight'],
            coord_weight=self.config['coord_weight']
        )
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=3
        )
        
        # Warmup scheduler
        self.warmup_scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: min(1.0, epoch / 10)
        )
        
        # Datasets
        self.train_dataset = ADCDataset(self.config['train_path'], self.model.tokenizer)
        self.val_dataset = ADCDataset(self.config['val_path'], self.model.tokenizer)
        self.test_dataset = ADCDataset(self.config['test_path'], self.model.tokenizer)
        
        # Dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            collate_fn=collate_fn
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            collate_fn=collate_fn
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            collate_fn=collate_fn
        )
        
        # Tracking
        self.best_val_loss = float('inf')
        os.makedirs(self.config['save_path'], exist_ok=True)
        self.results_file = os.path.join(self.config['save_path'], 'results.csv')
        
        with open(self.results_file, 'w') as f:
            f.write("epoch,train_loss,val_loss,test_loss,lr\n")

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch in pbar:
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            self.optimizer.zero_grad()
            outputs = self.model.forward_training({
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask']
            })
            
            losses = self.loss_fn(outputs, {
                'action_labels': batch['action_label'],
                'query_labels': batch['query_labels'],
                'touch_coords': batch['touch_coords'],
                'lift_coords': batch['lift_coords']
            })
            
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += losses['total_loss'].item()
            pbar.set_postfix({'loss': f"{losses['total_loss'].item():.4f}"})
        
        return total_loss / len(self.train_loader)

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                
                outputs = self.model.forward_training({
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask']
                })
                
                losses = self.loss_fn(outputs, {
                    'action_labels': batch['action_label'],
                    'query_labels': batch['query_labels'],
                    'touch_coords': batch['touch_coords'],
                    'lift_coords': batch['lift_coords']
                })
                
                total_loss += losses['total_loss'].item()
        
        return total_loss / len(dataloader)

    def train(self):
        for epoch in range(self.config['epochs']):
            train_loss = self.train_epoch()
            val_loss = self.evaluate(self.val_loader)
            test_loss = self.evaluate(self.test_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            self.warmup_scheduler.step()
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.model.save_pretrained(os.path.join(
                    self.config['save_path'], 
                    f'best_model_epoch_{epoch}'
                ))
            
            # Log results
            with open(self.results_file, 'a') as f:
                f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f},{test_loss:.4f},{self.optimizer.param_groups[0]['lr']:.2e}\n")
            
            print(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}, Test={test_loss:.4f}")
