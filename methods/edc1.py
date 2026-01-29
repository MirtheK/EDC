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
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler


class EDC:
    def __init__(self, model, num_epochs, it=0, num_eval=10, amap_reduction='mean', tb_log=None, logger=None):
        """
        """

        super(EDC, self).__init__()

        self.loader = {}
        self.model = model

        self.num_eval = num_eval
        self.tb_log = tb_log

        self.optimizer = None
        self.scheduler = None

        self.epochs = num_epochs
        self.it = 0
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
        self.model.train()

        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)

        start_batch.record()
        best_eval_auc, best_it = 0.0, 0
        patience_counter = 0
        save_path = os.path.join(args.save_dir, args.save_name)
        best_model_name = 'best_model.pth'

        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext

        # eval for once to verify if the checkpoint is loaded correctly
        if args.resume == True:
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


            # periodic evaluation
            if (epoch + 1) % self.num_eval == 0:
                eval_dict = self.evaluate(args=args)
                tb_dict.update(eval_dict)
                save_path = os.path.join(args.save_dir, args.save_name)

                if tb_dict['eval/AUC'] > best_eval_auc:
                    best_eval_auc = tb_dict['eval/AUC']
                    best_epoch = epoch
                    patience_counter = 0 
                    self.save_model(best_model_name, save_path)
                    save_amap = True

                # else:
                #     patience_counter += 1

                if save_amap:
                    self.evaluate(args=args, save_visual=True)

                # if patience_counter >= 5:
                #     self.print_fn(f"Early stopping at epoch {epoch+1}: AUC hasn't improved for 5 evaluations.")
                #     break  

                self.print_fn(
                    f"Epoch [{epoch+1}/{total_epochs}], "
                    f"{tb_dict}, BEST_EVAL_AUC: {best_eval_auc}, at it {best_epoch}"
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

        eval_dict = self.evaluate(args=args, save_visual=False)
        eval_dict.update({'eval/best_auc': best_eval_auc, 'eval/best_epoch': best_epoch})
        return eval_dict

    @torch.no_grad()
    def evaluate(self, eval_loader=None, args=None, save_visual=False):
        self.model.eval()
        if eval_loader is None:
            eval_loader = self.loader_dict['eval']

        total_num = 0.0
        total_loss = 0.0
        all_maps = []
        all_images = []
        all_filenames = []
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
            if self.amap_reduction == 'mean':
                p_img = result['p_all'].flatten(1).mean(1)
                p1_img = result['p1'].flatten(1).mean(1)
                p2_img = result['p2'].flatten(1).mean(1)
                p3_img = result['p3'].flatten(1).mean(1)
            elif isinstance(self.amap_reduction, float):  # the mean of max 'self.amap_reduction' percent
                anomaly_map = result['p_all'].flatten(1)
                p_img = torch.sort(anomaly_map, dim=1, descending=True)[0][:,
                        :int(anomaly_map.shape[1] * self.amap_reduction)].mean(dim=1)
                anomaly_map = result['p1'].flatten(1)
                p1_img = torch.sort(anomaly_map, dim=1, descending=True)[0][:,
                         :int(anomaly_map.shape[1] * self.amap_reduction)].mean(dim=1)
                anomaly_map = result['p2'].flatten(1)
                p2_img = torch.sort(anomaly_map, dim=1, descending=True)[0][:,
                         :int(anomaly_map.shape[1] * self.amap_reduction)].mean(dim=1)
                anomaly_map = result['p3'].flatten(1)
                p3_img = torch.sort(anomaly_map, dim=1, descending=True)[0][:,
                         :int(anomaly_map.shape[1] * self.amap_reduction)].mean(dim=1)
            else:  # max
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
                anomaly_maps = F.interpolate(result['p_all'], size=xo.shape[2:], mode='trilinear')
                all_maps.append(anomaly_maps.cpu())
                all_images.append(xo.cpu())
                all_filenames.extend(file_names)

        thresh = return_best_thr_youden(y_true, y_prob)
        acc = accuracy_score(y_true, y_prob >= thresh)
        f1 = f1_score(y_true, y_prob >= thresh)
        recall = recall_score(y_true, y_prob >= thresh)
        specificity = specificity_score(y_true, y_prob >= thresh)

        AUC = roc_auc_score(y_true, y_prob)
        AUC1 = roc_auc_score(y_true, y1_prob)
        AUC2 = roc_auc_score(y_true, y2_prob)
        AUC3 = roc_auc_score(y_true, y3_prob)

        if save_visual:
            save_path = os.path.join(args.save_dir, args.save_name, 'anomaly_map')
            image_save_path = os.path.join(args.save_dir, args.save_name, 'image')

            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            if not os.path.exists(image_save_path):
                os.makedirs(image_save_path, exist_ok=True)

            # Compute global min/max across ALL batches
            all_anomaly_maps = torch.cat(all_maps, dim=0)
            global_min = torch.min(all_anomaly_maps).numpy()
            global_max = torch.max(all_anomaly_maps).numpy()
            print(f"Global min, max across all batches: {global_min}, {global_max}")

            # Normalize and save all anomaly maps
            idx = 0
            for batch_maps, batch_images in zip(all_maps, all_images):
                for i in range(batch_maps.shape[0]):
                    image = np.squeeze(batch_images[i].numpy())
                    anomaly_map = np.squeeze(batch_maps[i].numpy())
                    amap_norm = (anomaly_map - global_min) / (global_max - global_min + 1e-8)

                    file_name = all_filenames[idx]
                    self.save_anomaly_map(amap_norm, image, save_path, image_save_path, file_name)
                    idx += 1


        self.model.train()
        return {'eval/loss': total_loss / total_num, 'eval/thr':thresh, 'eval/f1': f1, 'eval/recall': recall,
                'eval/specificity': specificity, 'eval/acc': acc,
                'eval/AUC': AUC, 'eval/AUC1': AUC1, 'eval/AUC2': AUC2, 'eval/AUC3': AUC3
                }

    def save_model(self, save_name, save_path):
        save_filename = os.path.join(save_path, save_name)
        self.model.train()
        torch.save({'model': self.model.state_dict()},
                   save_filename)
        self.print_fn(f"model saved: {save_filename}")

    def load_model(self, load_path):
        checkpoint = torch.load(load_path)

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.it = checkpoint['it']
        self.print_fn('model loaded')

    def save_anomaly_map(self, anomaly_map, image, save_path, image_save_path, file_name):
        img = nib.load("/projects/prjs1633/anomaly_detection/SHOMRI/zero_mask.nii.gz")
        affine = img.affine

        # anomaly_map_norm = min_max_norm(anomaly_map)
        # norm_anomaly_map = (anomaly_map_norm * 255)
        # norm_anomaly_map = np.flip(norm_anomaly_map, axis=2).copy()
        nib.save(nib.Nifti1Image(anomaly_map, affine), os.path.join(save_path, file_name))
        nib.save(nib.Nifti1Image(image, affine), os.path.join(image_save_path, file_name))




def return_best_thr(y_true, y_score):
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


def specificity_score(y_true, y_score):
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    TN = (y_true[y_score == 0] == 0).sum()
    N = (y_true == 0).sum()
    return TN / N


if __name__ == "__main__":
    pass