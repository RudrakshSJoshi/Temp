import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import os
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union
from transformers import DistilBertModel, DistilBertTokenizer, DistilBertConfig
from pydantic import BaseModel
import glob
from tqdm import tqdm
import datetime

def collate_fn(batch):
    # Handle tensors that can be batched
    batched_data = {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'action_label': torch.stack([item['action_label'] for item in batch]),
        'query_labels': torch.stack([item['query_labels'] for item in batch]),
        'touch_coords': torch.stack([item['touch_coords'] for item in batch]),
        'lift_coords': torch.stack([item['lift_coords'] for item in batch]),
        # Keep bounding boxes as a list of lists
        'bounding_boxes': [item['bounding_boxes'] for item in batch]
    }
    return batched_data

# Data models (unchanged)
class Results(BaseModel):
    action_type: int
    type_action: str
    touch_coords: List[float]
    lift_coords: List[float]

class DataEntry(BaseModel):
    prompt: str
    coordinates: List[List[float]]
    results: Results

# ADC Agent (unchanged)
class ADC_Agent(nn.Module):
    def __init__(self, 
                model_path: str = "models/distilbert-base-uncased",
                lstm_hidden_size: int = 256,
                max_seq_length: int = 512,
                freeze_bert: bool = False,
                dropout: float = 0.1):
        # ... (keep existing implementation unchanged)
        pass

# Custom Dataset (unchanged)
class ADCDataset(Dataset):
    def __init__(self, data_path: str, tokenizer):
        # ... (keep existing implementation unchanged)
        pass

# Updated Loss Function with separate backward passes
class ADCLoss(nn.Module):
    def __init__(self, 
                 action_weight: float = 1.0,
                 query_weight: float = 0.5,
                 coord_weight: float = 2.0,
                 tap_threshold: float = 0.04,
                 box_tolerance: float = 0.10):
        super().__init__()
        self.action_weight = action_weight
        self.query_weight = query_weight
        self.coord_weight = coord_weight
        self.tap_threshold = tap_threshold
        self.box_tolerance = box_tolerance
        
        # Separate loss functions for each head
        self.action_loss_fn = nn.CrossEntropyLoss()
        self.query_loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        self.coord_loss_fn = nn.MSELoss()  # Using MSE instead of Huber
    
    def is_tap(self, touch_coords, lift_coords):
        """Check if gesture is a tap based on Euclidean distance"""
        dist = torch.sqrt(torch.sum((touch_coords - lift_coords) ** 2, dim=-1))
        return dist <= self.tap_threshold
    
    def point_in_box(self, point, box):
        """Check if point (x, y) is inside box [x, y, w, h]"""
        px, py = point[0], point[1]
        bx, by, bw, bh = box[0], box[1], box[2], box[3]
        return (bx <= px <= bx + bw) and (by <= py <= by + bh)
    
    def get_swipe_direction(self, touch_coords, lift_coords):
        """Get swipe direction based on angle"""
        dx = lift_coords[0] - touch_coords[0]
        dy = lift_coords[1] - touch_coords[1]
        
        angle = math.atan2(dy, dx) * 180 / math.pi
        if angle < 0:
            angle += 360
            
        if 45 < angle < 135:
            return "up"
        elif 135 < angle < 225:
            return "left"
        elif 225 < angle < 315:
            return "down"
        else:
            return "right"
    
    def safe_loss(self, loss_fn, *args):
        """Compute loss with NaN handling"""
        loss = loss_fn(*args)
        if torch.isnan(loss):
            return torch.zeros_like(loss)
        return loss
    
    def compute_coordinate_loss(self, pred_coords, target_touch, target_lift, bounding_boxes):
        """Compute custom coordinate loss with MSE and NaN handling"""
        batch_size = pred_coords.size(0)
        total_loss = 0.0
        
        for i in range(batch_size):
            # Extract predicted coordinates [touch_x, touch_y, lift_x, lift_y]
            pred_touch = pred_coords[i, :2]
            pred_lift = pred_coords[i, 2:]
            
            actual_touch = target_touch[i]
            actual_lift = target_lift[i]

            boxes = bounding_boxes[i]
            
            # Check if actual gesture is tap or swipe
            actual_is_tap = self.is_tap(actual_touch, actual_lift)
            pred_is_tap = self.is_tap(pred_touch, pred_lift)
            
            # Heavy penalty if predicted and actual categories differ
            if actual_is_tap != pred_is_tap:
                loss = self.safe_loss(
                    self.coord_loss_fn, 
                    pred_coords[i], 
                    torch.cat([actual_touch, actual_lift])
                )
                total_loss += 10.0 * loss  # Heavy penalty
                continue
            
            # Same category - apply specific logic
            if actual_is_tap:
                # Tap logic
                tap_coord = (actual_touch + actual_lift) / 2  # Average for actual tap center
                
                # Check which bounding box the actual tap hits
                boxes = bounding_boxes[i]
                hit_box = None
                for box in boxes:
                    if self.point_in_box(tap_coord, box):
                        hit_box = box
                        break
                
                # Check if predicted tap hits the same box
                pred_tap_coord = (pred_touch + pred_lift) / 2
                pred_hits_same_box = False
                
                if hit_box is not None:
                    pred_hits_same_box = self.point_in_box(pred_tap_coord, hit_box)
                
                if pred_hits_same_box:
                    # No error if hitting same box
                    pass
                else:
                    # Check 10% tolerance
                    dist = torch.sqrt(torch.sum((pred_tap_coord - tap_coord) ** 2))
                    if dist <= self.box_tolerance:
                        # No error within tolerance
                        pass
                    else:
                        # Apply MSE loss
                        loss = self.safe_loss(
                            self.coord_loss_fn,
                            pred_coords[i], 
                            torch.cat([actual_touch, actual_lift])
                        )
                        total_loss += loss
            else:
                # Swipe logic - check direction
                actual_direction = self.get_swipe_direction(actual_touch, actual_lift)
                pred_direction = self.get_swipe_direction(pred_touch, pred_lift)
                
                if actual_direction != pred_direction:
                    # Wrong direction - apply MSE loss
                    loss = self.safe_loss(
                        self.coord_loss_fn,
                        pred_coords[i], 
                        torch.cat([actual_touch, actual_lift])
                    )
                    total_loss += loss
                # If direction matches, no additional loss
        
        return total_loss / batch_size if batch_size > 0 else torch.tensor(0.0)
    
    def forward(self, outputs, targets):
        losses = {}
        
        # Action loss with NaN handling
        losses['action_loss'] = self.safe_loss(
            self.action_loss_fn,
            outputs['action_logits'],
            targets['action_labels']
        )
        
        # Query loss (only for search actions) with NaN handling
        if outputs['query_logits'].numel() > 0:
            # Reshape for sequence loss
            query_logits = outputs['query_logits'].view(-1, outputs['query_logits'].size(-1))
            query_targets = targets['query_labels'].view(-1)
            
            losses['query_loss'] = self.safe_loss(
                self.query_loss_fn,
                query_logits,
                query_targets
            )
        else:
            losses['query_loss'] = torch.tensor(0.0, device=outputs['action_logits'].device)
        
        # Coordinate loss with NaN handling
        if outputs['coordinates'].numel() > 0:
            losses['coord_loss'] = self.compute_coordinate_loss(
                outputs['coordinates'],
                targets['touch_coords'],
                targets['lift_coords'],
                targets['bounding_boxes']
            )
        else:
            losses['coord_loss'] = torch.tensor(0.0, device=outputs['action_logits'].device)
        
        # Weighted total loss
        losses['total_loss'] = (
            self.action_weight * losses['action_loss'] +
            self.query_weight * losses['query_loss'] +
            self.coord_weight * losses['coord_loss']
        )
        
        return losses

# Updated Training Pipeline with separate backward passes
class ADCTrainer:
    def __init__(self, config_path: str = "data_parameters.txt"):
        self.config = self.load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = ADC_Agent(
            model_path=self.config['model_path'],
            lstm_hidden_size=self.config['lstm_hidden_size'],
            max_seq_length=self.config['max_seq_length'],
            freeze_bert=self.config['freeze_bert'],
            dropout=self.config['dropout']
        ).to(self.device)
        
        # Initialize loss function
        self.loss_fn = ADCLoss(
            action_weight=self.config['action_weight'],
            query_weight=self.config['query_weight'],
            coord_weight=self.config['coord_weight'],
            tap_threshold=self.config['tap_threshold'],
            box_tolerance=self.config['box_tolerance']
        )
        
        # Initialize optimizers for each head
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Initialize scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config['reduce_lr_factor'],
            patience=self.config['reduce_lr_patience']
        )
        
        # Load datasets
        self.train_dataset = ADCDataset(self.config['train_path'], self.model.tokenizer)
        self.val_dataset = ADCDataset(self.config['val_path'], self.model.tokenizer)
        self.test_dataset = ADCDataset(self.config['test_path'], self.model.tokenizer)
    
        # Create dataloaders
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
        
        # Tracking variables
        self.best_test_mse = float('inf')
        self.best_val_mse_at_best_test = float('inf')
        self.current_epoch = 0
        
        # Create results directory if it doesn't exist
        os.makedirs(self.config['save_path'], exist_ok=True)
        self.results_file = os.path.join(self.config['save_path'], 'training_results.txt')
        
        # Write header to results file
        with open(self.results_file, 'w') as f:
            f.write("epoch,train_total,train_action,train_query,train_coord,val_total,val_action,val_query,val_coord,test_total,test_action,test_query,test_coord,lr,timestamp\n")
    
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from file"""
        # ... (keep existing implementation unchanged)
        pass
    
    def train_epoch(self):
        """Train for one epoch with separate backward passes"""
        self.model.train()
        total_losses = {'total_loss': 0, 'action_loss': 0, 'query_loss': 0, 'coord_loss': 0}
        num_batches = 0
        
        with tqdm(self.train_loader, desc="ðŸ”¥ Training", 
                 bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}') as pbar:
            for batch in pbar:
                # Move batch to device
                for key in batch:
                    if torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model.forward_training(
                    {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
                )
                
                # Prepare targets
                targets = {
                    'action_labels': batch['action_label'],
                    'query_labels': batch['query_labels'],
                    'touch_coords': batch['touch_coords'],
                    'lift_coords': batch['lift_coords'],
                    'bounding_boxes': batch['bounding_boxes']
                }
                
                # Compute losses with NaN checks
                losses = self.loss_fn(outputs, targets)
                
                # Backward pass for each component separately
                if torch.isfinite(losses['action_loss']):
                    (self.config['action_weight'] * losses['action_loss']).backward(retain_graph=True)
                
                if torch.isfinite(losses['query_loss']):
                    (self.config['query_weight'] * losses['query_loss']).backward(retain_graph=True)
                
                if torch.isfinite(losses['coord_loss']):
                    (self.config['coord_weight'] * losses['coord_loss']).backward()
                
                # Clip gradients and update
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # Accumulate losses
                for key in total_losses:
                    if key in losses:
                        total_losses[key] += losses[key].item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f"{losses['total_loss'].item():.4f}",
                    'Action': f"{losses.get('action_loss', 0):.4f}",
                    'Query': f"{losses.get('query_loss', 0):.4f}",
                    'Coord': f"{losses.get('coord_loss', 0):.4f}"
                })
        
        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        return avg_losses
    
    def evaluate(self, dataloader, desc="Evaluating"):
        """Evaluate on given dataloader with NaN checks"""
        self.model.eval()
        total_losses = {'total_loss': 0, 'action_loss': 0, 'query_loss': 0, 'coord_loss': 0}
        num_batches = 0
        
        with torch.no_grad(), tqdm(dataloader, desc=desc, 
                                bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}') as pbar:
            for batch in pbar:
                # Move batch to device
                for key in batch:
                    if torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(self.device)
                
                # Forward pass
                outputs = self.model.forward_training(
                    {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
                )
                
                # Prepare targets
                targets = {
                    'action_labels': batch['action_label'],
                    'query_labels': batch['query_labels'],
                    'touch_coords': batch['touch_coords'],
                    'lift_coords': batch['lift_coords'],
                    'bounding_boxes': batch['bounding_boxes']
                }
                
                # Compute losses with NaN checks
                losses = self.loss_fn(outputs, targets)
                
                # Accumulate losses
                for key in total_losses:
                    if key in losses:
                        total_losses[key] += losses[key].item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f"{losses['total_loss'].item():.4f}",
                    'Action': f"{losses.get('action_loss', 0):.4f}",
                    'Query': f"{losses.get('query_loss', 0):.4f}",
                    'Coord': f"{losses.get('coord_loss', 0):.4f}"
                })
        
        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        return avg_losses
    
    def save_model(self, epoch, test_mse, val_mse):
        """Save model if criteria are met"""
        save_model = False
        
        if test_mse < self.best_test_mse:
            self.best_test_mse = test_mse
            self.best_val_mse_at_best_test = val_mse
            save_model = True
        elif test_mse == self.best_test_mse and val_mse < self.best_val_mse_at_best_test:
            self.best_val_mse_at_best_test = val_mse
            save_model = True
        
        if save_model:
            save_path = os.path.join(self.config['save_path'], f'best_model_epoch_{epoch}')
            self.model.save_pretrained(save_path)
            print(f"Model saved at epoch {epoch} with Test MSE: {test_mse:.6f}, Val MSE: {val_mse:.6f}")
    
    def log_results(self, epoch, train_losses, val_losses, test_losses, lr):
        """Log results to file"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.results_file, 'a') as f:
            f.write(
                f"{epoch},{train_losses['total_loss']:.6f},{train_losses['action_loss']:.6f},"
                f"{train_losses['query_loss']:.6f},{train_losses['coord_loss']:.6f},"
                f"{val_losses['total_loss']:.6f},{val_losses['action_loss']:.6f},"
                f"{val_losses['query_loss']:.6f},{val_losses['coord_loss']:.6f},"
                f"{test_losses['total_loss']:.6f},{test_losses['action_loss']:.6f},"
                f"{test_losses['query_loss']:.6f},{test_losses['coord_loss']:.6f},"
                f"{lr:.2e},{timestamp}\n"
            )
    
    def train(self):
        """Main training loop with separate backward passes"""
        print("Starting training...")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        print(f"Test samples: {len(self.test_dataset)}")
        
        for epoch in range(self.current_epoch, self.config['epochs']):
            self.current_epoch = epoch + 1
            print(f"\n{'='*60}")
            print(f"EPOCH {self.current_epoch}/{self.config['epochs']}")
            print(f"{'='*60}")
            
            # Train
            train_losses = self.train_epoch()
            print(f"TRAIN    Total: {train_losses['total_loss']:.6f}  Action: {train_losses['action_loss']:
