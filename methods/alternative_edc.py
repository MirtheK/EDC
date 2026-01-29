import contextlib
import os
from collections import Counter
from copy import deepcopy
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import *
from torch.cuda.amp import autocast, GradScaler
from train_utils import Bn_Controller
from torch.autograd import Function
import matplotlib.pyplot as plt
import cv2
import nibabel as nib


class EDC:
    def __init__(self, model, num_epochs, num_eval, amap_reduction='max', tb_log=None, logger=None):
        """
        Epoch-based training for anomaly detection.
        """
        super(EDC, self).__init__()

        self.loader = {}
        self.model = model

        self.tb_log = tb_log

        self.optimizer = None
        self.scheduler = None

        self.epochs = num_epochs
        self.num_eval = num_eval

        self.logger = logger
        self.print_fn = print if logger is None else logger.info
        self.amap_reduction = amap_reduction
        self.bn_controller = Bn_Controller()

    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        self.print_fn(f'[!] data loader keys: {self.loader_dict.keys()}')

    def set_dset(self, dset):
        self.ulb_dset = dset

    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, args, logger=None):
        """
        Train the model for args.num_epochs epochs.
        Each epoch processes the entire training dataset.
        """
        self.model.train()

        best_eval_auc = 0.0
        best_epoch = 0

        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext

        # eval for once to verify if the checkpoint is loaded correctly
        if args.resume == True:
            eval_dict = self.evaluate(args=args)
            print(eval_dict)

        train_log = []
        
        self.print_fn(f"\nStarting training for {self.epochs} epochs")
        self.print_fn(f"Training samples per epoch: {len(self.loader_dict['train'].dataset)}")
        self.print_fn(f"Batches per epoch: {len(self.loader_dict['train'])}\n")
        
        # EPOCH-BASED TRAINING LOOP
        for epoch in range(self.epochs):
            self.model.train()
            
            epoch_loss = 0.0
            epoch_e1_std = 0.0
            epoch_e2_std = 0.0
            epoch_e3_std = 0.0
            num_batches = 0
            
            self.print_fn(f"\n{'='*60}")
            self.print_fn(f"Epoch {epoch+1}/{self.epochs}")
            self.print_fn(f"{'='*60}")

            # Process all training data
            for batch_idx, (idx, x, _, y, filename) in enumerate(self.loader_dict['train']):
                x = x.cuda(args.gpu)

                with amp_cm():
                    result = self.model(x)
                    total_loss = result['loss'].mean()

                # parameter updates
                if args.amp:
                    scaler.scale(total_loss).backward()
                    if args.clip > 0:
                        scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    if args.clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                    self.optimizer.step()

                self.model.zero_grad()

                # Accumulate metrics
                epoch_loss += total_loss.detach().item()
                epoch_e1_std += result['e1_std'].detach().item()
                epoch_e2_std += result['e2_std'].detach().item()
                epoch_e3_std += result['e3_std'].detach().item()
                num_batches += 1

                # Periodic logging during epoch
                if batch_idx % 10 == 0:
                    progress = 100. * batch_idx / len(self.loader_dict['train'])
                    self.print_fn(
                        f"  [{batch_idx}/{len(self.loader_dict['train'])}] "
                        f"({progress:.0f}%) Loss: {total_loss.item():.4f}"
                    )

            # Step scheduler after each epoch
            if self.scheduler is not None:
                self.scheduler.step()

            # Calculate epoch averages
            avg_epoch_loss = epoch_loss / num_batches
            avg_e1_std = epoch_e1_std / num_batches
            avg_e2_std = epoch_e2_std / num_batches
            avg_e3_std = epoch_e3_std / num_batches

            # Evaluate every 5 epochs
            if (epoch + 1) % self.num_eval == 0:
                self.print_fn(f"\nEvaluating...")
                eval_dict = self.evaluate(args=args)
            else:
                eval_dict = {}  # Skip evaluation for this epoch

            # Prepare metrics dictionary
            tb_dict = {
                'epoch': epoch + 1,
                'train/loss': avg_epoch_loss,
                'train/e1_std': avg_e1_std,
                'train/e2_std': avg_e2_std,
                'train/e3_std': avg_e3_std,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            tb_dict.update(eval_dict)

            save_path = os.path.join(args.save_dir, args.save_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # Save best model (only when evaluation was performed)
            if eval_dict and tb_dict['eval/AUC'] > best_eval_auc:
                best_eval_auc = tb_dict['eval/AUC']
                best_epoch = epoch + 1
                self.save_model('model_best.pth', save_path)
                self.print_fn(f"\n*** New Best Model: AUC = {best_eval_auc:.4f} ***")

            # Print epoch summary
            self.print_fn(f"\nEpoch {epoch+1} Summary:")
            self.print_fn(f"  Train Loss:      {avg_epoch_loss:.4f}")
            if eval_dict:  # Only print eval metrics if evaluation was performed
                self.print_fn(f"  Eval Loss:       {tb_dict['eval/loss']:.4f}")
                self.print_fn(f"  Eval AUC:        {tb_dict['eval/AUC']:.4f}")
                self.print_fn(f"  Eval F1:         {tb_dict['eval/f1']:.4f}")
                self.print_fn(f"  Eval Recall:     {tb_dict['eval/recall']:.4f}")
                self.print_fn(f"  Eval Specificity:{tb_dict['eval/specificity']:.4f}")
            self.print_fn(f"  Learning Rate:   {tb_dict['lr']:.6f}")
            self.print_fn(f"  Best AUC so far: {best_eval_auc:.4f} (Epoch {best_epoch})")

            # Log to tensorboard
            if self.tb_log is not None:
                self.tb_log.update(tb_dict, epoch)

            train_log.append(tb_dict)

            # Save periodic checkpoints
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', save_path, epoch)

        # Save final model and training log
        self.save_model('model_final.pth', save_path)
        
        with open(os.path.join(save_path, 'train_log.pkl'), 'wb') as f:
            pickle.dump(train_log, f)

        # Final evaluation
        self.print_fn(f"\n{'='*60}")
        self.print_fn("Final Evaluation...")
        eval_dict = self.evaluate(args=args, save_visual=True)
        eval_dict.update({
            'eval/best_auc': best_eval_auc, 
            'eval/best_epoch': best_epoch
        })
        
        self.print_fn(f"\n{'='*60}")
        self.print_fn("Training Complete!")
        self.print_fn(f"Best AUC: {best_eval_auc:.4f} achieved at Epoch {best_epoch}")
        self.print_fn(f"{'='*60}\n")
        
        return eval_dict

    @torch.no_grad()
    def evaluate(self, eval_loader=None, args=None, save_visual=False):
        """
        Evaluate the model on the evaluation dataset.
        Normalizes anomaly maps across all images in the dataset.
        """
        self.model.eval()
        if eval_loader is None:
            eval_loader = self.loader_dict['eval']
            
        total_num = 0.0
        total_loss = 0.0
        y_true = []
        y_prob = []
        y1_prob = []
        y2_prob = []
        y3_prob = []
        
        # First pass: collect all anomaly maps for global normalization
        all_anomaly_maps = []
        all_images = []
        all_labels = []
        all_filenames = []
        all_results = []

        self.print_fn("  Collecting anomaly maps for normalization...")
        for _, x, xo, y, file_names in eval_loader:
            x, y = x.cuda(args.gpu), y.cuda(args.gpu).float()
            num_batch = x.shape[0]
            total_num += num_batch
            
            result = self.model(x)
            
            # Store for later processing
            all_anomaly_maps.append(result['p_all'].cpu())
            all_images.append(xo)
            all_labels.append(y.cpu())
            all_filenames.extend(file_names)
            all_results.append({
                'p1': result['p1'].cpu(),
                'p2': result['p2'].cpu(),
                'p3': result['p3'].cpu(),
                'loss': result['loss'].detach().item() * num_batch
            })
            
            total_loss += result['loss'].detach().item() * num_batch

        # Concatenate all anomaly maps
        all_anomaly_maps = torch.cat(all_anomaly_maps, dim=0)  # (N, C, D, H, W) or (N, C, H, W)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Global normalization: compute min/max across ALL samples
        global_min = all_anomaly_maps.min()
        global_max = all_anomaly_maps.max()
        self.print_fn(f"  Global anomaly map range: [{global_min:.4f}, {global_max:.4f}]")
        
        # Normalize all anomaly maps with global statistics
        all_anomaly_maps_norm = (all_anomaly_maps - global_min) / (global_max - global_min + 1e-8)
        
        # Second pass: compute scores with normalized maps
        self.print_fn("  Computing anomaly scores...")
        for i in range(len(all_results)):
            # Get normalized maps for this batch
            start_idx = sum(r['p1'].shape[0] for r in all_results[:i])
            end_idx = start_idx + all_results[i]['p1'].shape[0]
            
            p_all_norm = all_anomaly_maps_norm[start_idx:end_idx]
            
            # Also normalize individual scale maps
            p1 = all_results[i]['p1']
            p1_norm = (p1 - global_min) / (global_max - global_min + 1e-8)
            p2 = all_results[i]['p2']
            p2_norm = (p2 - global_min) / (global_max - global_min + 1e-8)
            p3 = all_results[i]['p3']
            p3_norm = (p3 - global_min) / (global_max - global_min + 1e-8)
            
            # Compute anomaly scores
            if self.amap_reduction == 'mean':
                p_img = p_all_norm.flatten(1).mean(1)
                p1_img = p1_norm.flatten(1).mean(1)
                p2_img = p2_norm.flatten(1).mean(1)
                p3_img = p3_norm.flatten(1).mean(1)
            elif isinstance(self.amap_reduction, float):
                # Mean of top X% most anomalous pixels
                anomaly_map = p_all_norm.flatten(1)
                p_img = torch.sort(anomaly_map, dim=1, descending=True)[0][:,
                        :int(anomaly_map.shape[1] * self.amap_reduction)].mean(dim=1)
                anomaly_map = p1_norm.flatten(1)
                p1_img = torch.sort(anomaly_map, dim=1, descending=True)[0][:,
                         :int(anomaly_map.shape[1] * self.amap_reduction)].mean(dim=1)
                anomaly_map = p2_norm.flatten(1)
                p2_img = torch.sort(anomaly_map, dim=1, descending=True)[0][:,
                         :int(anomaly_map.shape[1] * self.amap_reduction)].mean(dim=1)
                anomaly_map = p3_norm.flatten(1)
                p3_img = torch.sort(anomaly_map, dim=1, descending=True)[0][:,
                         :int(anomaly_map.shape[1] * self.amap_reduction)].mean(dim=1)
            else:  # max
                p_img = p_all_norm.flatten(1).max(1)[0]
                p1_img = p1_norm.flatten(1).max(1)[0]
                p2_img = p2_norm.flatten(1).max(1)[0]
                p3_img = p3_norm.flatten(1).max(1)[0]

            y_batch = all_labels[start_idx:end_idx]
            y_true.extend(y_batch.tolist())
            y_prob.extend(p_img.tolist())
            y1_prob.extend(p1_img.tolist())
            y2_prob.extend(p2_img.tolist())
            y3_prob.extend(p3_img.tolist())

        # Visualization with globally normalized maps
        if save_visual:
            save_path = os.path.join(args.save_dir, args.save_name, 'heatmap')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                
            self.print_fn("  Saving visualizations...")
            all_images_concat = torch.cat(all_images, dim=0)
            for i in range(len(all_filenames)):
                image = all_images_concat[i].numpy()
                anomaly_map = all_anomaly_maps_norm[i].numpy()
                file_name = all_filenames[i]
                self.save_anomaly_map(anomaly_map, image, save_path, file_name)

        # Calculate metrics with multiple threshold methods
        self.print_fn("  Computing metrics...")
        
        thresh_f1 = return_best_thr_f1(y_true, y_prob)
        thresh_youden = return_best_thr_youden(y_true, y_prob)
        thresh_gmean = return_best_thr_gmean(y_true, y_prob)
        thresh_median = np.median(y_prob)
        
        self.print_fn(f"  Threshold (F1):      {thresh_f1:.4f}")
        self.print_fn(f"  Threshold (Youden):  {thresh_youden:.4f}")
        self.print_fn(f"  Threshold (G-mean):  {thresh_gmean:.4f}")
        self.print_fn(f"  Threshold (Median):  {thresh_median:.4f}")
        
        thresh = thresh_f1
        
        acc = accuracy_score(y_true, y_prob >= thresh)
        f1 = f1_score(y_true, y_prob >= thresh)
        recall = recall_score(y_true, y_prob >= thresh)
        specificity = specificity_score(y_true, y_prob >= thresh)

        AUC = roc_auc_score(y_true, y_prob)
        AUC1 = roc_auc_score(y_true, y1_prob)
        AUC2 = roc_auc_score(y_true, y2_prob)
        AUC3 = roc_auc_score(y_true, y3_prob)
        
        # Additional metrics
        avg_prob_normal = np.mean([p for p, l in zip(y_prob, y_true) if l == 0])
        avg_prob_anomaly = np.mean([p for p, l in zip(y_prob, y_true) if l == 1])
        
        self.print_fn(f"  Avg score (normal):  {avg_prob_normal:.4f}")
        self.print_fn(f"  Avg score (anomaly): {avg_prob_anomaly:.4f}")

        self.model.train()
        return {
            'eval/loss': total_loss / total_num, 
            'eval/f1': f1, 
            'eval/recall': recall,
            'eval/specificity': specificity, 
            'eval/acc': acc,
            'eval/AUC': AUC, 
            'eval/AUC1': AUC1, 
            'eval/AUC2': AUC2, 
            'eval/AUC3': AUC3,
            'eval/threshold_f1': thresh_f1,
            'eval/threshold_youden': thresh_youden,
            'eval/threshold_gmean': thresh_gmean,
            'eval/avg_score_normal': avg_prob_normal,
            'eval/avg_score_anomaly': avg_prob_anomaly,
            'eval/y_prob_mean': np.mean(y_prob),
            'eval/y_prob_std': np.std(y_prob),
        }

    def save_model(self, save_name, save_path):
        """Save model state dict."""
        save_filename = os.path.join(save_path, save_name)
        torch.save({'model': self.model.state_dict()}, save_filename)
        self.print_fn(f"Model saved: {save_filename}")

    def save_checkpoint(self, save_name, save_path, epoch):
        """Save full checkpoint including optimizer and scheduler."""
        save_filename = os.path.join(save_path, save_name)
        checkpoint = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            checkpoint['scheduler'] = self.scheduler.state_dict()
        torch.save(checkpoint, save_filename)
        self.print_fn(f"Checkpoint saved: {save_filename}")

    def load_model(self, load_path):
        """Load model from checkpoint."""
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        if 'epoch' in checkpoint:
            self.epoch = checkpoint['epoch']
        self.print_fn(f'Model loaded from {load_path}')

    def save_anomaly_map(self, anomaly_map, image, save_path, file_name):
        """Save anomaly map visualization."""
        img = nib.load("/projects/prjs1633/anomaly_detection/SHOMRI/zero_mask.nii.gz")
        affine = img.affine
        nib.save(nib.Nifti1Image(anomaly_map, affine), os.path.join(save_path, file_name))
        nib.save(nib.Nifti1Image(image, affine), os.path.join(image_save_path, file_name))


def return_best_thr_f1(y_true, y_score):
    """Find best threshold based on F1 score."""
    precs, recs, thrs = precision_recall_curve(y_true, y_score)
    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    thrs = thrs[~np.isnan(f1s)]
    f1s = f1s[~np.isnan(f1s)]
    best_thr = thrs[np.argmax(f1s)]
    return best_thr


def return_best_thr_youden(y_true, y_score):
    """
    Find best threshold using Youden's J statistic.
    J = sensitivity + specificity - 1
    Maximizes the difference between true positive rate and false positive rate.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j_scores = tpr - fpr  # Youden's J statistic
    best_idx = np.argmax(j_scores)
    best_thr = thresholds[best_idx]
    return best_thr


def return_best_thr_gmean(y_true, y_score):
    """
    Find best threshold using G-mean (geometric mean).
    G-mean = sqrt(sensitivity * specificity)
    Good for imbalanced datasets.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    # Calculate specificity = 1 - fpr
    specificity = 1 - fpr
    # Calculate G-mean
    gmean = np.sqrt(tpr * specificity)
    best_idx = np.argmax(gmean)
    best_thr = thresholds[best_idx]
    return best_thr


def specificity_score(y_true, y_score):
    """Calculate specificity (true negative rate)."""
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    TN = (y_true[y_score == 0] == 0).sum()
    N = (y_true == 0).sum()
    return TN / N if N > 0 else 0.0


if __name__ == "__main__":
    pass