import contextlib
import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, f1_score, precision_recall_curve, confusion_matrix
from torch.cuda.amp import autocast, GradScaler
# from train_utils import Bn_Controller # Placeholder: assuming this is defined elsewhere
import nibabel as nib
from copy import deepcopy # Included from original imports
from collections import Counter # Included from original imports



def calculate_threshold_from_errors(normal_train_errors, percentile=95):
    """
    Calculates the anomaly detection threshold based on the p-th percentile 
    of the pre-calculated reconstruction errors from the normal training data.
    """
    errors_array = np.array(normal_train_errors)
    
    if errors_array.size == 0:
        print("[WARNING] Input list of errors is empty, returning threshold 0.0.")
        return 0.0
    
    threshold = np.percentile(errors_array, percentile)
    return threshold


def specificity_score(y_true, y_pred_class):
    """Calculates specificity (True Negative Rate)."""
    y_true = np.array(y_true)
    y_pred_class = np.array(y_pred_class).astype(int) # Ensure binary class labels

    cm = confusion_matrix(y_true, y_pred_class, labels=[0, 1])
    TN = cm[0, 0]
    FP = cm[0, 1]
    
    N = TN + FP 
    return TN / N if N > 0 else 0.0


def min_max_norm(image):
    """Performs min-max normalization on an image/map."""
    a_min, a_max = image.min(), image.max()
    if a_max - a_min == 0:
        return np.zeros_like(image)
    return (image - a_min) / (a_max - a_min)



class EDC:
    def __init__(self, model, num_epochs, it=0, num_eval=10, amap_reduction='max', tb_log=None, logger=None, percentile=99):
        super().__init__()

        self.loader_dict = {}
        self.model = model

        self.num_eval = num_eval
        self.tb_log = tb_log
        self.optimizer = None
        self.scheduler = None

        self.epochs = num_epochs
        self.it = it
        self.logger = logger
        self.print_fn = print if logger is None else logger.info
        self.amap_reduction = amap_reduction
        self.percentile = percentile 
        # self.bn_controller = Bn_Controller() # Assuming this is defined/imported

    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        self.print_fn(f'[!] data loader keys: {self.loader_dict.keys()}')

    def set_dset(self, dset):
        self.ulb_dset = dset

    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def save_model(self, save_name, save_path):
        os.makedirs(save_path, exist_ok=True)
        save_filename = os.path.join(save_path, save_name)
        self.model.train()
        torch.save({'model': self.model.state_dict()}, save_filename)
        self.print_fn(f"model saved: {save_filename}")

    def load_model(self, load_path):
        if not os.path.exists(load_path):
            self.print_fn(f"WARNING: Checkpoint not found at {load_path}")
            return

        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['model'])
        
        # Load optimizer/scheduler state if they exist
        if self.optimizer and 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler and 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        
        if 'it' in checkpoint:
             self.it = checkpoint['it']
             
        self.print_fn(f'model loaded from: {load_path}')
        
    def save_anomaly_map(self, anomaly_map, image, save_path, image_save_path, file_name):
        try:
             # NOTE: Hardcoded path for zero_mask.nii.gz
             img = nib.load("/projects/prjs1633/anomaly_detection/SHOMRI/zero_mask.nii.gz")
             affine = img.affine
             nib.save(nib.Nifti1Image(anomaly_map*255, affine), os.path.join(save_path, file_name))
             nib.save(nib.Nifti1Image(image, affine), os.path.join(image_save_path, file_name))
        except FileNotFoundError:
             self.print_fn("WARNING: Cannot save Nifti files, zero_mask.nii.gz not found.")


    @torch.no_grad()
    def get_normal_train_errors(self, args):
        """
        Calculates reconstruction errors (p_img) for the normal training set 
        to determine the percentile threshold.
        """
        self.model.eval()
        train_loader = self.loader_dict.get('train')

        normal_train_errors = []
        
        for _, x, _, _, _ in train_loader: 
            x = x.cuda(args.gpu) if 'gpu' in args and args.gpu is not None else x

            result = self.model(x)
            if self.amap_reduction == 'mean':
                p_img = result['p_all'].flatten(1).mean(1)
            elif isinstance(self.amap_reduction, float):
                anomaly_map = result['p_all'].flatten(1)
                num_pixels_to_keep = int(anomaly_map.shape[1] * self.amap_reduction)
                p_img = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :num_pixels_to_keep].mean(dim=1)
            else:  # 'max' (default)
                p_img = result['p_all'].flatten(1).max(1)[0]

            normal_train_errors.extend(p_img.cpu().tolist())
            
        self.model.train()
        return normal_train_errors


    def train(self, args, logger=None):
        self.model.train()
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)

        start_batch.record()
        best_eval_auc, best_epoch = 0.0, 0
        patience_counter = 0
        save_path = os.path.join(args.save_dir, args.save_name)
        best_model_name = 'best_model.pth'

        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext

        if args.resume == True:
            # Assuming load_model is called before train, or needs to be called here
            self.load_model(os.path.join(args.load_path, 'latest_model.pth')) 
            eval_dict = self.evaluate(args=args)
            print(eval_dict)

        train_log = []
        total_epochs = self.epochs

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0

            for idx, x, _, y, filename in self.loader_dict['train']:

                end_batch.record()
                torch.cuda.synchronize()
                start_run.record()

                x = x.cuda()

                with amp_cm():
                    result = self.model(x)
                    total_loss = result['loss'].mean()

                # parameter updates
                if args.amp:
                    scaler.scale(total_loss).backward()
                    if args.clip > 0:
                        scaler.unscale_(self.optimizer)
                        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    if args.clip > 0:
                        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                    self.optimizer.step()

                self.scheduler.step()
                self.model.zero_grad()
                epoch_loss += total_loss.item()

                end_run.record()
                torch.cuda.synchronize()

                # tensorboard_dict update
                tb_dict = {
                    'train/total_loss': total_loss.detach().item(),
                    'train/e1_std': result['e1_std'].detach().item(),
                    'train/e2_std': result['e2_std'].detach().item(),
                    'train/e3_std': result['e3_std'].detach().item(),
                    'lr': self.optimizer.param_groups[0]['lr'],
                    'train/prefecth_time': start_batch.elapsed_time(end_batch) / 1000.,
                    'train/run_time': start_run.elapsed_time(end_run) / 1000.,
                }
                start_batch.record() # Start timing for next batch prefetch


            # periodic evaluation
            if (epoch + 1) % self.num_eval == 0:
                # Periodic evaluation uses F1-optimized threshold for logging
                eval_dict = self.evaluate(args=args, fixed_threshold=None) 
                tb_dict.update(eval_dict)
                save_path = os.path.join(args.save_dir, args.save_name)

                save_amap = False
                if tb_dict['eval/AUC'] > best_eval_auc:
                    best_eval_auc = tb_dict['eval/AUC']
                    best_epoch = epoch
                    patience_counter = 0 
                    self.save_model(best_model_name, save_path)
                    save_amap = True

                # else:
                #     patience_counter += 1
                
                if save_amap:
                    self.evaluate(args=args, save_visual=True, fixed_threshold=None)

                # if patience_counter >= 5:
                #     self.print_fn(f"Early stopping at epoch {epoch+1}: AUC hasn't improved for 5 evaluations.")
                #     break  

                self.print_fn(
                    f"Epoch [{epoch+1}/{total_epochs}], "
                    f"{tb_dict}, BEST_EVAL_AUC: {best_eval_auc:.4f}, at epoch {best_epoch+1}"
                )

                if self.tb_log is not None:
                    self.tb_log.update(tb_dict, self.it)
                    tb_dict['epoch'] = epoch
                    train_log.append(tb_dict)

                self.it += 1
                del tb_dict
                start_batch.record()


        f_save = open(os.path.join(save_path, 'train_log.pkl'), 'wb')
        pickle.dump(train_log, f_save)
        f_save.close()

        self.load_model(os.path.join(save_path, best_model_name)) 
        
        normal_errors = self.get_normal_train_errors(args)
        
        T_percentile = calculate_threshold_from_errors(
            normal_errors, 
            percentile=self.percentile 
        )
        
        eval_dict = self.evaluate(args=args, save_visual=False, fixed_threshold=T_percentile)
        eval_dict.update({'eval/best_auc': best_eval_auc, 'eval/best_epoch': best_epoch})
        
        return eval_dict


    @torch.no_grad()
    def evaluate(self, eval_loader=None, args=None, save_visual=False, fixed_threshold=None):
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

        for _, x, xo, y, file_names in eval_loader:
            x, y = x.cuda(args.gpu), y.cuda(args.gpu).float()
            num_batch = x.shape[0]
            total_num += num_batch
            result = self.model(x)
            
            # Anomaly Map Reduction Logic
            if self.amap_reduction == 'mean':
                p_img = result['p_all'].flatten(1).mean(1)
                p1_img = result['p1'].flatten(1).mean(1)
                p2_img = result['p2'].flatten(1).mean(1)
                p3_img = result['p3'].flatten(1).mean(1)
            elif isinstance(self.amap_reduction, float): 
                anomaly_map = result['p_all'].flatten(1)
                num_pixels_to_keep = int(anomaly_map.shape[1] * self.amap_reduction)
                p_img = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :num_pixels_to_keep].mean(dim=1)
                
                anomaly_map = result['p1'].flatten(1)
                p1_img = torch.sort(anomaly_map, dim=1, descending=True)[0][:,:int(anomaly_map.shape[1] * self.amap_reduction)].mean(dim=1)
                anomaly_map = result['p2'].flatten(1)
                p2_img = torch.sort(anomaly_map, dim=1, descending=True)[0][:,:int(anomaly_map.shape[1] * self.amap_reduction)].mean(dim=1)
                anomaly_map = result['p3'].flatten(1)
                p3_img = torch.sort(anomaly_map, dim=1, descending=True)[0][:,:int(anomaly_map.shape[1] * self.amap_reduction)].mean(dim=1)
                
            else:  # max (default)
                p_img = result['p_all'].flatten(1).max(1)[0]
                p1_img = result['p1'].flatten(1).max(1)[0]
                p2_img = result['p2'].flatten(1).max(1)[0]
                p3_img = result['p3'].flatten(1).max(1)[0]

            y_true.extend(y.cpu().tolist())
            y_prob.extend(p_img.cpu().tolist())
            y1_prob.extend(p1_img.cpu().tolist())
            y2_prob.extend(p2_img.cpu().tolist())
            y3_prob.extend(p3_img.cpu().tolist())

            total_loss += result['loss'].detach().item() * num_batch

            if save_visual:
                save_path = os.path.join(args.save_dir, args.save_name, 'anomaly_map')
                image_save_path = os.path.join(args.save_dir, args.save_name, 'image')

                os.makedirs(save_path, exist_ok=True)
                os.makedirs(image_save_path, exist_ok=True)

                anomaly_maps = F.interpolate(result['p_all'], size=xo.shape[2:], mode='trilinear')
                global_min = torch.min(anomaly_maps).cpu().numpy()
                global_max = torch.max(anomaly_maps).cpu().numpy()

                for i in range(xo.shape[0]):
                    image = np.squeeze(xo[i].cpu().numpy()) # shape: (D, H, W)
                    anomaly_map = np.squeeze(anomaly_maps[i].cpu().numpy())
                    amap_norm = (anomaly_map - global_min) / (global_max - global_min + 1e-8)

                    file_name = file_names[i]
                    self.save_anomaly_map(amap_norm, image, save_path, image_save_path, file_name)

        if fixed_threshold is not None:
            thresh = fixed_threshold
            self.print_fn(f"[EVAL] Using fixed threshold: {thresh:.4f}")
        else:
            precs, recs, thrs = precision_recall_curve(y_true, y_prob)
            f1s = 2 * precs * recs / (precs + recs + 1e-7)
            f1s = f1s[:-1]
            thrs = thrs[~np.isnan(f1s)]
            f1s = f1s[~np.isnan(f1s)]
            thresh = thrs[np.argmax(f1s)]
            
        y_pred_class = np.array(y_prob) >= thresh
        acc = accuracy_score(y_true, y_pred_class)
        f1 = f1_score(y_true, y_pred_class)
        recall = recall_score(y_true, y_pred_class)
        specificity = specificity_score(y_true, y_pred_class)

        AUC = roc_auc_score(y_true, y_prob)
        AUC1 = roc_auc_score(y_true, y1_prob)
        AUC2 = roc_auc_score(y_true, y2_prob)
        AUC3 = roc_auc_score(y_true, y3_prob)

        self.model.train()
        return {'eval/loss': total_loss / total_num, 'eval/thr':thresh, 'eval/f1': f1, 'eval/recall': recall,
                'eval/specificity': specificity, 'eval/acc': acc,
                'eval/AUC': AUC, 'eval/AUC1': AUC1, 'eval/AUC2': AUC2, 'eval/AUC3': AUC3
                }




import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, recall_score, f1_score

from torch.cuda.amp import autocast, GradScaler


def compute_youden_threshold(y_true, y_score):
    """
    Stable thresholding using ROC curve + Youden J statistic (TPR-FPR).
    """
    fpr, tpr, thr = roc_curve(y_true, y_score)
    J = tpr - fpr
    return thr[np.argmax(J)]


def specificity(y_true, y_pred):
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    return float(tn) / float(tn + fp + 1e-8)


class TrainerStable:
    def __init__(self, model, loader_dict, optimizer, scheduler, epochs, num_eval=5):
        self.model = model
        self.loaders = loader_dict
        self.opt = optimizer
        self.sch = scheduler
        self.epochs = epochs
        self.num_eval = num_eval

    def train(self, args):
        scaler = GradScaler(enabled=args.amp)
        best_auc = -1
        best_epoch = 0

        for epoch in range(self.epochs):
            self.model.train()
            loss_epoch = 0

            for _, x, _, _, _ in self.loaders["train"]:
                x = x.cuda()

                with autocast(enabled=args.amp):
                    res = self.model(x)
                    loss = res["loss"]

                self.opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.opt)
                scaler.update()
                self.sch.step()

                loss_epoch += loss.item()

            if (epoch + 1) % self.num_eval == 0:
                eval_dict = self.evaluate(args)
                auc = eval_dict["AUC"]

                print(f"[Epoch {epoch+1}] Loss: {loss_epoch:.4f} | AUC {auc:.4f}")

                if auc > best_auc:
                    best_auc = auc
                    best_epoch = epoch
                    torch.save(self.model.state_dict(), args.save_dir + "/best.pth")

        print(f"Best AUC = {best_auc:.4f} at epoch {best_epoch}")
        return best_auc

    @torch.no_grad()
    def evaluate(self, args, fixed_threshold=None):
        self.model.eval()

        y_true = []
        y_score = []
        y1 = []
        y2 = []
        y3 = []

        for _, x, _, y, _ in self.loaders["eval"]:
            x = x.cuda()
            y = y.cuda().float()

            res = self.model(x)
            score = res["score"]
            y_true.extend(y.cpu().tolist())
            y_score.extend(score.cpu().tolist())
            y1.extend(res["score1"].cpu().tolist())
            y2.extend(res["score2"].cpu().tolist())
            y3.extend(res["score3"].cpu().tolist())

        y_true = np.array(y_true)
        y_score = np.array(y_score)

        if fixed_threshold is None:
            thr = compute_youden_threshold(y_true, y_score)
        else:
            thr = fixed_threshold

        y_pred = (y_score >= thr).astype(int)

        return {
            "thr": thr,
            "acc": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "specificity": specificity(y_true, y_pred),
            "AUC": roc_auc_score(y_true, y_score),
            "AUC1": roc_auc_score(y_true, y1),
            "AUC2": roc_auc_score(y_true, y2),
            "AUC3": roc_auc_score(y_true, y3),
        }
