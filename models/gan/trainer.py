"""
PyTorch Trainer for GANomaly 3D Medical Image Anomaly Detection

This trainer handles:
- Training loop with separate generator and discriminator optimization
- Validation every N epochs
- Best model checkpointing based on validation AUROC
- Standard metrics (AUROC, accuracy, precision, recall, F1)
- GPU/CPU automatic detection
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib.pyplot as plt


class GanomalyTrainer:
    """
    Trainer for GANomaly model on 3D medical images.
    
    Args:
        model: GanomalyModel instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        generator_loss: GeneratorLoss instance
        discriminator_loss: DiscriminatorLoss instance
        learning_rate: Learning rate for optimizers (default: 0.0002)
        beta1: Beta1 parameter for Adam (default: 0.5)
        beta2: Beta2 parameter for Adam (default: 0.999)
        checkpoint_dir: Directory to save checkpoints (default: './checkpoints')
        device: Device to use ('cuda' or 'cpu', auto-detected if None)
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        generator_loss,
        discriminator_loss,
        learning_rate=0.0002,
        beta1=0.5,
        beta2=0.999,
        checkpoint_dir='./checkpoints',
        device=None
    ):
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Model and data
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Loss functions
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss
        
        # Optimizers
        self.optimizer_d = torch.optim.Adam(
            self.model.discriminator.parameters(),
            lr=5e-5,
            betas=(beta1, beta2)
        )
        self.optimizer_g = torch.optim.Adam(
            self.model.generator.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2)
        )
        
        # Checkpoint setup
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.histogram_dir = os.path.join(checkpoint_dir, 'histograms')
        os.makedirs(self.histogram_dir, exist_ok=True)
        
        # Metrics tracking
        self.best_val_auroc = 0.0
        self.train_history = {
            'generator_loss': [],
            'discriminator_loss': [],
            'total_loss': []
        }
        self.val_history = {
            'auroc': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        # For normalization during validation/test
        self.min_score = float('inf')
        self.max_score = float('-inf')
    
    def train_epoch(self, epoch):
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with average losses
        """
        self.model.train()
        
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # Unpack batch: (idx, img_n, img_t, target, filename)
            idx, img_n, img_t, target, filename = batch
            images = img_t.to(self.device)
            
            # Forward pass through model
            padded, fake, latent_i, latent_o = self.model(images)
            
            # ==================== Generator Update ====================
            pred_real, _ = self.model.discriminator(padded)
            pred_fake, _ = self.model.discriminator(fake)
            
            g_loss = self.generator_loss(
                latent_i, latent_o, padded, fake, pred_real, pred_fake
            )
            
            self.optimizer_g.zero_grad()
            g_loss.backward(retain_graph=True)
            self.optimizer_g.step()
            
            # ==================== Discriminator Update ====================
            # Detach fake to avoid backprop through generator
            pred_fake_detached, _ = self.model.discriminator(fake.detach())
            
            d_loss = self.discriminator_loss(pred_real, pred_fake_detached)
            
            self.optimizer_d.zero_grad()
            d_loss.backward()
            self.optimizer_d.step()
            
            # Track losses
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'G_loss': f'{g_loss.item():.4f}',
                'D_loss': f'{d_loss.item():.4f}'
            })
        
        # Calculate average losses
        avg_g_loss = epoch_g_loss / len(self.train_loader)
        avg_d_loss = epoch_d_loss / len(self.train_loader)
        avg_total_loss = avg_g_loss + avg_d_loss
        
        return {
            'generator_loss': avg_g_loss,
            'discriminator_loss': avg_d_loss,
            'total_loss': avg_total_loss
        }

    def plot_histograms(self, epoch, scores, labels, thr, normalized_scores=None):
        """
        Plot and save histograms of anomaly scores for normal and anomalous samples.
        
        Args:
            epoch: Current epoch number
            scores: Raw anomaly scores
            labels: Ground truth labels (0=normal, 1=anomaly)
            normalized_scores: Normalized scores (optional)
        """
        # Separate scores by class
        normal_scores = scores[labels == 0]
        anomaly_scores = scores[labels == 1]
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 1, figsize=(14, 5))
        
        # Plot raw scores
        ax = axes
        if len(normal_scores) > 0:
            ax.hist(normal_scores, bins=50, alpha=0.6, label=f'Normal (n={len(normal_scores)})', 
                   color='green', edgecolor='black')
        if len(anomaly_scores) > 0:
            ax.hist(anomaly_scores, bins=50, alpha=0.6, label=f'Anomaly (n={len(anomaly_scores)})', 
                   color='red', edgecolor='black')

        ax.axvline(x=thr, color='blue', linestyle='--', linewidth=2, label=f'Threshold {thr}')

        ax.set_xlabel('Raw Anomaly Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Raw Anomaly Scores - Epoch {epoch}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'Normal: μ={normal_scores.mean():.4f}, σ={normal_scores.std():.4f}\n'
        if len(anomaly_scores) > 0:
            stats_text += f'Anomaly: μ={anomaly_scores.mean():.4f}, σ={anomaly_scores.std():.4f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=9, 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # # Plot normalized scores if provided
        # if normalized_scores is not None:
        #     ax = axes[1]
        #     normal_norm = normalized_scores[labels == 0]
        #     anomaly_norm = normalized_scores[labels == 1]
            
        #     if len(normal_norm) > 0:
        #         ax.hist(normal_norm, bins=50, alpha=0.6, label=f'Normal (n={len(normal_norm)})', 
        #                color='green', edgecolor='black')
        #     if len(anomaly_norm) > 0:
        #         ax.hist(anomaly_norm, bins=50, alpha=0.6, label=f'Anomaly (n={len(anomaly_norm)})', 
        #                color='red', edgecolor='black')
            
        #     # Add threshold line at 0.5
        #     ax.axvline(x=thr, color='blue', linestyle='--', linewidth=2, label=f'Threshold {thr}')
            
        #     ax.set_xlabel('Normalized Anomaly Score', fontsize=12)
        #     ax.set_ylabel('Frequency', fontsize=12)
        #     ax.set_title(f'Normalized Anomaly Scores - Epoch {epoch}', fontsize=14, fontweight='bold')
        #     ax.legend(fontsize=10)
        #     ax.grid(True, alpha=0.3)
            
        #     # Add statistics text
        #     stats_text = f'Normal: μ={normal_norm.mean():.4f}, σ={normal_norm.std():.4f}\n'
        #     if len(anomaly_norm) > 0:
        #         stats_text += f'Anomaly: μ={anomaly_norm.mean():.4f}, σ={anomaly_norm.std():.4f}'
        #     ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
        #            verticalalignment='top', fontsize=9,
        #            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(self.histogram_dir, f'epoch_{epoch:04d}_histograms.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def return_best_thr(self, y_true, y_score):
        precs, recs, thrs = precision_recall_curve(y_true, y_score)
        f1s = 2 * precs * recs / (precs + recs + 1e-7)
        f1s = f1s[:-1]
        thrs = thrs[~np.isnan(f1s)]
        f1s = f1s[~np.isnan(f1s)]
        best_thr = thrs[np.argmax(f1s)]
        return best_thr
    
    def validate(self, epoch):
        """
        Validate the model.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        all_scores = []
        all_labels = []
        
        # Reset min/max for this validation run
        self.min_score = float('inf')
        self.max_score = float('-inf')
        
        print("\nValidation - Computing anomaly scores...")
        
        with torch.no_grad():
            # First pass: compute scores and find min/max
            for batch in tqdm(self.val_loader, desc='Val Pass 1'):
                idx, img_n, img_t, target, filename = batch
                images = img_t.to(self.device)
                labels = target.cpu().numpy()
                
                # Get anomaly scores
                output = self.model(images)
                scores = output.pred_score.cpu().numpy()
                
                # Update min/max
                self.min_score = min(self.min_score, scores.min())
                self.max_score = max(self.max_score, scores.max())
                
                all_scores.extend(scores)
                all_labels.extend(labels)
        
        # Normalize scores
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        normalized_scores = (all_scores - self.min_score) / (self.max_score - self.min_score + 1e-8)

        thr = self.return_best_thr(all_labels, all_scores)
        self.plot_histograms(epoch, all_scores, all_labels, thr, normalized_scores)

        # Compute metrics
        predictions = (all_scores > thr).astype(int)
        
        # Calculate metrics
        auroc = roc_auc_score(all_labels, all_scores)
        accuracy = accuracy_score(all_labels, predictions)
        precision = precision_score(all_labels, predictions, zero_division=0)
        recall = recall_score(all_labels, predictions, zero_division=0)
        f1 = f1_score(all_labels, predictions, zero_division=0)
        
        metrics = {
            'auroc': auroc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        print(f"\nValidation Metrics (Epoch {epoch}):")
        print(f"  AUROC:     {auroc:.4f}")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        
        return metrics
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            metrics: Dictionary of validation metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'metrics': metrics,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_val_auroc': self.best_val_auroc
        }
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
        
        # Also save latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, 'latest_model.pth')
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        self.train_history = checkpoint['train_history']
        self.val_history = checkpoint['val_history']
        self.best_val_auroc = checkpoint['best_val_auroc']
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Best validation AUROC: {self.best_val_auroc:.4f}")
        
        return checkpoint['epoch']
    
    def train(self, num_epochs, val_every=5, resume_from=None):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            val_every: Run validation every N epochs
            resume_from: Path to checkpoint to resume from (optional)
        """
        start_epoch = 1
        
        if resume_from is not None:
            start_epoch = self.load_checkpoint(resume_from) + 1
        
        print(f"Starting training for {num_epochs} epochs")
        print(f"Validation every {val_every} epochs")
        print(f"Checkpoint directory: {self.checkpoint_dir}")
        
        for epoch in range(start_epoch, num_epochs + 1):
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Log training metrics
            self.train_history['generator_loss'].append(train_metrics['generator_loss'])
            self.train_history['discriminator_loss'].append(train_metrics['discriminator_loss'])
            self.train_history['total_loss'].append(train_metrics['total_loss'])
            
            print(f"\nEpoch {epoch}/{num_epochs} - Training Metrics:")
            print(f"  Generator Loss:     {train_metrics['generator_loss']:.4f}")
            print(f"  Discriminator Loss: {train_metrics['discriminator_loss']:.4f}")
            print(f"  Total Loss:         {train_metrics['total_loss']:.4f}")
            
            # Validation
            if epoch % val_every == 0 or epoch == num_epochs:
                val_metrics = self.validate(epoch)
                
                # Log validation metrics
                for key, value in val_metrics.items():
                    self.val_history[key].append(value)
                
                # Check if this is the best model
                is_best = val_metrics['auroc'] > self.best_val_auroc
                if is_best:
                    self.best_val_auroc = val_metrics['auroc']
                    print(f"\n*** New best model! AUROC: {self.best_val_auroc:.4f} ***")
                
                # Save checkpoint
                self.save_checkpoint(epoch, val_metrics, is_best=is_best)
            
            print(f"\n{'-'*60}\n")
        
        # Save final training history
        self.save_training_history()
        
        print(f"Training completed!")
        print(f"Best validation AUROC: {self.best_val_auroc:.4f}")
    
    def save_training_history(self):
        """Save training history to JSON file."""
        history = {
            'train': self.train_history,
            'val': self.val_history,
            'best_val_auroc': self.best_val_auroc
        }
        
        history_path = os.path.join(self.checkpoint_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        
        print(f"Saved training history to {history_path}")


def create_trainer(
    model,
    train_loader,
    val_loader,
    generator_loss,
    discriminator_loss,
    config
):
    """
    Factory function to create a trainer.
    
    Args:
        model: GanomalyModel instance
        train_loader: Training data loader
        val_loader: Validation data loader
        generator_loss: GeneratorLoss instance
        discriminator_loss: DiscriminatorLoss instance
        config: Dictionary with training configuration
        
    Returns:
        GanomalyTrainer instance
    """
    trainer = GanomalyTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        generator_loss=generator_loss,
        discriminator_loss=discriminator_loss,
        learning_rate=config.get('learning_rate', 0.0002),
        beta1=config.get('beta1', 0.5),
        beta2=config.get('beta2', 0.999),
        checkpoint_dir=config.get('checkpoint_dir', './checkpoints'),
        device=config.get('device', None)
    )
    
    return trainer