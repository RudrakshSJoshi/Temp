def train_epoch(self):
    """Train for one epoch with separate gradient flows for each head"""
    self.model.train()
    total_losses = {'total_loss': 0, 'action_loss': 0, 'query_loss': 0, 'coord_loss': 0}
    num_batches = 0
    
    with tqdm(self.train_loader, desc="Training", 
             bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}') as pbar:
        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
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
            
            # Compute losses
            losses = self.loss_fn(outputs, targets)
            
            # Create dummy zero tensor for gradient accumulation
            dummy_zero = torch.tensor(0.0, requires_grad=True, device=self.device)
            
            # Action loss backward (only affects action head)
            if torch.isfinite(losses['action_loss']):
                (dummy_zero + self.config['action_weight'] * losses['action_loss']).backward(retain_graph=True)
            
            # Query loss backward (only affects query head)
            if torch.isfinite(losses['query_loss']):
                (dummy_zero + self.config['query_weight'] * losses['query_loss']).backward(retain_graph=True)
            
            # Coordinate loss backward (only affects coordinate head)
            if torch.isfinite(losses['coord_loss']):
                (dummy_zero + self.config['coord_weight'] * losses['coord_loss']).backward()
            
            # Clip gradients and update
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track losses
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
    
    return {k: v / num_batches for k, v in total_losses.items()}
