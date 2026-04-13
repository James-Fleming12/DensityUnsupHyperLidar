#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import datetime
import os
import time
import cv2
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from common.avgmeter import *
from torch.utils.tensorboard import SummaryWriter
from common.sync_batchnorm.batchnorm import convert_model
from modules.scheduler.warmupLR import *
from modules.ioueval import *
from modules.losses.Lovasz_Softmax import Lovasz_softmax
from modules.scheduler.cosine import CosineAnnealingWarmUpRestarts

from tqdm import tqdm

import torch.nn.functional as F

def save_to_log(logdir, logfile, message):
    f = open(logdir + '/' + logfile, "a")
    f.write(message + '\n')
    f.close()
    return

def save_checkpoint(to_save, logdir, suffix=""):
    # Save the weights
    torch.save(to_save, logdir +
               "/SENet" + suffix)

class DGLSSTrainer():
    def __init__(self, ARCH, DATA, datadir, logdir, path=None, dist_type="standard", depth=False):
        """
        dist_type can be 'standard' (L1/MSE) or 'angular' (Cosine/ArcFace-style)
        """
        self.dist_type = dist_type
        self.lam1_max = 0.5
        self.lam2_max = 0.5
        self.lam1 = 0.0
        self.lam2 = 0.0

        # parameters
        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = datadir
        self.log = logdir
        self.path = path

        self.batch_time_t = AverageMeter()
        self.data_time_t = AverageMeter()
        self.batch_time_e = AverageMeter()
        self.epoch = 0

        self.info = {"train_loss": 0,
                     "train_acc": 0,
                     "train_iou": 0,
                     "valid_loss": 0,
                     "valid_acc": 0,
                     "valid_iou": 0,
                     "best_train_iou": 0,
                     "best_val_iou": 0}

        # get the data
        from dataset.kitti.parser import Parser
        self.parser = Parser(root=self.datadir,
                                          train_sequences=self.DATA["split"]["train"], # self.DATA["split"]["valid"] + self.DATA["split"]["train"] if finetune with valid
                                          valid_sequences=self.DATA["split"]["valid"],
                                          test_sequences=None,
                                          labels=self.DATA["labels"],
                                          color_map=self.DATA["color_map"],
                                          learning_map=self.DATA["learning_map"],
                                          learning_map_inv=self.DATA["learning_map_inv"],
                                          sensor=self.ARCH["dataset"]["sensor"],
                                          max_points=self.ARCH["dataset"]["max_points"],
                                          batch_size=self.ARCH["train"]["batch_size"],
                                          workers=self.ARCH["train"]["workers"],
                                          gt=True,
                                          shuffle_train=True)

        # weights for loss (and bias)

        epsilon_w = self.ARCH["train"]["epsilon_w"]
        content = torch.zeros(self.parser.get_n_classes(), dtype=torch.float)
        for cl, freq in DATA["content"].items():
            x_cl = self.parser.to_xentropy(cl)  # map actual class to xentropy class
            content[x_cl] += freq
        self.loss_w = 1 / (content + epsilon_w)  # get weights

        for x_cl, w in enumerate(self.loss_w):  # ignore the ones necessary to ignore
            if DATA["learning_ignore"][x_cl]:
                # don't weigh
                self.loss_w[x_cl] = 0
        print("Loss weights from content: ", self.loss_w.data)

        with torch.no_grad():
            if self.ARCH["train"]["pipeline"] == "hardnet":
                from modules.network.HarDNet import HarDNet
                self.model = HarDNet(self.parser.get_n_classes(), self.ARCH["train"]["aux_loss"])

            if self.ARCH["train"]["pipeline"] == "res":
                from modules.network.ResNet import ResNet_34
                self.model = ResNet_34(self.parser.get_n_classes(), self.ARCH["train"]["aux_loss"], depth=depth)

                def convert_relu_to_softplus(model, act):
                    for child_name, child in model.named_children():
                        if isinstance(child, nn.LeakyReLU):
                            setattr(model, child_name, act)
                        else:
                            convert_relu_to_softplus(child, act)
                if self.ARCH["train"]["act"] == "Hardswish":
                    convert_relu_to_softplus(self.model, nn.Hardswish())
                elif self.ARCH["train"]["act"] == "SiLU":
                    convert_relu_to_softplus(self.model, nn.SiLU())

            if self.ARCH["train"]["pipeline"] == "fid":
                from modules.network.Fid import ResNet_34
                self.model = ResNet_34(self.parser.get_n_classes(), self.ARCH["train"]["aux_loss"])

                if self.ARCH["train"]["act"] == "Hardswish":
                    convert_relu_to_softplus(self.model, nn.Hardswish())
                elif self.ARCH["train"]["act"] == "SiLU":
                    convert_relu_to_softplus(self.model, nn.SiLU())

        # save_to_log(self.log, 'model.txt', str(self.model))
        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Number of parameters: ", pytorch_total_params/1000000, "M")
        self.tb_logger = SummaryWriter(log_dir=self.log, flush_secs=20)

        # GPU?
        self.gpu = False
        self.multi_gpu = False
        self.n_gpus = 0
        self.model_single = self.model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Training in device: ", self.device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            cudnn.benchmark = True
            cudnn.fastest = True
            self.gpu = True
            self.n_gpus = 1
            self.model.cuda()
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)  # spread in gpus
            self.model = convert_model(self.model).cuda()  # sync batchnorm
            self.model_single = self.model.module  # single model to get weight names
            self.multi_gpu = True
            self.n_gpus = torch.cuda.device_count()


        self.criterion = nn.NLLLoss(weight=self.loss_w).to(self.device)
        self.ls = Lovasz_softmax(ignore=0).to(self.device)
        from modules.losses.boundary_loss import BoundaryLoss
        self.bd = BoundaryLoss().to(self.device)
        # loss as dataparallel too (more images in batch)
        if self.n_gpus > 1:
            self.criterion = nn.DataParallel(self.criterion).cuda()  # spread in gpus
            self.ls = nn.DataParallel(self.ls).cuda()

        if self.ARCH["train"]["scheduler"] == "consine":
            length = self.parser.get_train_size()
            dict = self.ARCH["train"]["consine"]
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=dict["min_lr"],
                                       momentum=self.ARCH["train"]["momentum"],
                                       weight_decay=self.ARCH["train"]["w_decay"])
            self.scheduler = CosineAnnealingWarmUpRestarts(optimizer=self.optimizer,
                                                           T_0=dict["first_cycle"] * length, T_mult=dict["cycle"],
                                                           eta_max=dict["max_lr"],
                                                           T_up=dict["wup_epochs"]*length, gamma=dict["gamma"])

        else:
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=self.ARCH["train"]["decay"]["lr"],
                                       momentum=self.ARCH["train"]["momentum"],
                                       weight_decay=self.ARCH["train"]["w_decay"])
            steps_per_epoch = self.parser.get_train_size()
            up_steps = int(self.ARCH["train"]["decay"]["wup_epochs"] * steps_per_epoch)
            final_decay = self.ARCH["train"]["decay"]["lr_decay"] ** (1 / steps_per_epoch)
            self.scheduler = warmupLR(optimizer=self.optimizer,
                                      lr=self.ARCH["train"]["decay"]["lr"],
                                      warmup_steps=up_steps,
                                      momentum=self.ARCH["train"]["momentum"],
                                      decay=final_decay)

        if self.path is not None:
            torch.nn.Module.dump_patches = True
            w_dict = torch.load(path + "/SENet",
                                map_location=lambda storage, loc: storage)
            self.model.load_state_dict(w_dict['state_dict'], strict=True)
            print("dict epoch:", w_dict['epoch'])
            # self.info = w_dict['info']
            print("info", w_dict['info'])

    def beam_drop(self, in_vol, p_range=(0.0, 0.15)):
        bs, channels, h, w = in_vol.shape
        p = np.random.uniform(p_range[0], p_range[1])
        num_drop = int(h * p)
        
        indices = np.random.choice(h, num_drop, replace=False)
        # Simulate dropout by zeroing out the entire row (beam)
        in_vol[:, :, indices, :] = 0 
        return in_vol
    
    def compute_prototypes(self, features, labels):
        b, c, h, w = features.shape
        valid_classes = [i for i in range(self.parser.get_n_classes()) if i != 0]
        n_valid = len(valid_classes)
        prototypes = torch.zeros(b, n_valid, c, device=features.device)
        for bi in range(b):
            feat_i = features[bi].permute(1, 2, 0).reshape(-1, c)
            lbl_i  = labels[bi].reshape(-1)
            for ci, cls in enumerate(valid_classes):
                mask = (lbl_i == cls)
                if mask.sum() > 10:
                    prototypes[bi, ci] = feat_i[mask].mean(0)
        return prototypes

    def calculate_estimate(self, epoch, iter):
        estimate = int((self.data_time_t.avg + self.batch_time_t.avg) * \
                       (self.parser.get_train_size() * self.ARCH['train']['max_epochs'] - (
                               iter + 1 + epoch * self.parser.get_train_size()))) + \
                   int(self.batch_time_e.avg * self.parser.get_valid_size() * (
                           self.ARCH['train']['max_epochs'] - (epoch)))
        return str(datetime.timedelta(seconds=estimate))

    @staticmethod
    def get_mpl_colormap(cmap_name):
        cmap = plt.get_cmap(cmap_name)
        # Initialize the matplotlib color map
        sm = plt.cm.ScalarMappable(cmap=cmap)
        # Obtain linear color range
        color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]
        return color_range.reshape(256, 1, 3)

    @staticmethod
    def make_log_img(depth, mask, pred, gt, color_fn):
        # input should be [depth, pred, gt]
        # make range image (normalized to 0,1 for saving)
        depth = (cv2.normalize(depth, None, alpha=0, beta=1,
                               norm_type=cv2.NORM_MINMAX,
                               dtype=cv2.CV_32F) * 255.0).astype(np.uint8)
        out_img = cv2.applyColorMap(
            depth, Trainer.get_mpl_colormap('viridis')) * mask[..., None]
        # make label prediction
        pred_color = color_fn((pred * mask).astype(np.int32))
        out_img = np.concatenate([out_img, pred_color], axis=0)
        # make label gt
        gt_color = color_fn(gt)
        out_img = np.concatenate([out_img, gt_color], axis=0)
        return (out_img).astype(np.uint8)

    @staticmethod
    def save_to_log(logdir, logger, info, epoch, w_summary=False, model=None, img_summary=False, imgs=[]):
        # save scalars
        for tag, value in info.items():
            logger.add_scalar(tag, value, epoch)

        # save summaries of weights and biases
        if w_summary and model:
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
                if value.grad is not None:
                    logger.histo_summary(
                        tag + '/grad', value.grad.data.cpu().numpy(), epoch)

        if img_summary and len(imgs) > 0:
            directory = os.path.join(logdir, "predictions")
            if not os.path.isdir(directory):
                os.makedirs(directory)
            for i, img in enumerate(imgs):
                name = os.path.join(directory, str(i) + ".png")
                cv2.imwrite(name, img)

    def train(self, epochs=None):

        self.ignore_class = []
        for i, w in enumerate(self.loss_w):
            if w < 1e-10:
                self.ignore_class.append(i)
                print("Ignoring class ", i, " in IoU evaluation")
        self.evaluator = iouEval(self.parser.get_n_classes(),
                                 self.device, self.ignore_class)
        # save_to_log(self.log, 'log.txt', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        if self.path is not None:
            acc, iou, loss, rand_img = self.validate(val_loader=self.parser.get_valid_set(),
                                             model=self.model,
                                             criterion=self.criterion,
                                             evaluator=self.evaluator,
                                             class_func=self.parser.get_xentropy_class_string,
                                             color_fn=self.parser.to_color,
                                             save_scans=self.ARCH["train"]["save_scans"])

        # train for n epochs
        max_epochs = epochs if epochs is not None else self.ARCH["train"]["max_epochs"]
        warmup_start = int(0.1 * max_epochs)
        
        for epoch in range(self.epoch, max_epochs):
            if epoch < warmup_start:
                self.lam1 = 0.0
                self.lam2 = 0.0
            else:
                progress = (epoch - warmup_start) / max(1, max_epochs - warmup_start)
                self.lam1 = self.lam1_max * progress
                self.lam2 = self.lam2_max * progress
            
            # train for 1 epoch

            acc, iou, loss = self.train_epoch(train_loader=self.parser.get_train_set(),
                                                           model=self.model,
                                                           criterion=self.criterion,
                                                           optimizer=self.optimizer,
                                                           epoch=epoch,
                                                           evaluator=self.evaluator,
                                                           scheduler=self.scheduler,
                                                           color_fn=self.parser.to_color,
                                                           report=self.ARCH["train"]["report_batch"],
                                                           show_scans=self.ARCH["train"]["show_scans"])


            # update info
            self.info["train_loss"] = loss
            self.info["train_acc"] = acc
            self.info["train_iou"] = iou

            # remember best iou and save checkpoint
            state = {'epoch': epoch, 'state_dict': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     'info': self.info,
                     'scheduler': self.scheduler.state_dict()
                     }
            save_checkpoint(state, self.log, suffix="")
            # save_checkpoint(state, self.log, suffix=""+str(epoch))

            if self.info['train_iou'] > self.info['best_train_iou']:
                # save_to_log(self.log, 'log.txt', "Best mean iou in training set so far, save model!")
                print("Best mean iou in training set so far, save model!")
                self.info['best_train_iou'] = self.info['train_iou']
                state = {'epoch': epoch, 'state_dict': self.model.state_dict(),
                         'optimizer': self.optimizer.state_dict(),
                         'info': self.info,
                         'scheduler': self.scheduler.state_dict()
                         }
                save_checkpoint(state, self.log, suffix="_train_best")

            if epoch % self.ARCH["train"]["report_epoch"] == 0:
                # evaluate on validation set
                print("*" * 80)
                acc, iou, loss, rand_img = self.validate(val_loader=self.parser.get_valid_set(),
                                                         model=self.model,
                                                         criterion=self.criterion,
                                                         evaluator=self.evaluator,
                                                         class_func=self.parser.get_xentropy_class_string,
                                                         color_fn=self.parser.to_color,
                                                         save_scans=self.ARCH["train"]["save_scans"])

                # update info
                self.info["valid_loss"] = loss
                self.info["valid_acc"] = acc
                self.info["valid_iou"] = iou

            # remember best iou and save checkpoint
            if self.info['valid_iou'] > self.info['best_val_iou']:
                # save_to_log(self.log, 'log.txt', "Best mean iou in validation so far, save model!")
                print("Best mean iou in validation so far, save model!")
                print("*" * 80)
                self.info['best_val_iou'] = self.info['valid_iou']

                # save the weights!
                state = {'epoch': epoch, 'state_dict': self.model.state_dict(),
                         'optimizer': self.optimizer.state_dict(),
                         'info': self.info,
                         'scheduler': self.scheduler.state_dict()
                         }
                save_checkpoint(state, self.log, suffix="_valid_best")

            print("*" * 80)

        print('Finished Training')
        # save_to_log(self.log, 'log.txt', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        return

    def train_epoch(self, train_loader, model, criterion, optimizer, epoch, evaluator, scheduler, color_fn, report=10, show_scans=False):
        losses = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()
        train_time = []

        # Empty the cache to train now
        if self.gpu:
            torch.cuda.empty_cache()

        evaluator.reset()
        model.train()
        
        scaler = torch.amp.GradScaler('cuda')
        end = time.time()
        for i, (in_vol, proj_mask, proj_labels, _, path_seq, path_name, _, _, _, _, _, _, _, _, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
            # Measure data loading time
            self.data_time_t.update(time.time() - end)
            
            if self.gpu:
                in_vol, proj_labels = in_vol.cuda(), proj_labels.cuda().long()

            in_vol_aug = self.beam_drop(in_vol.clone())

            start = time.time()

            with torch.amp.autocast('cuda'):
                if self.ARCH["train"]["aux_loss"]:
                    output, aux_list, z8, enc_s = model(in_vol, return_enc=True)
                    output_aug, aux_list_aug, z8_aug, enc_a = model(in_vol_aug, return_enc=True)
                else:
                    output, z8, enc_s = model(in_vol, return_enc=True)
                    output_aug, z8_aug, enc_a = model(in_vol_aug, return_enc=True)

                loss_sem_orig = criterion(torch.log(output.clamp(min=1e-8)), proj_labels) + 1.5 * self.ls(output, proj_labels)
                loss_sem_aug = criterion(torch.log(output_aug.clamp(min=1e-8)), proj_labels) + 1.5 * self.ls(output_aug, proj_labels)

                loss_sem = (loss_sem_orig + loss_sem_aug) / 2

                # SIFC Loss (with masking)
                valid_mask = (proj_labels > 0) # Shape: [B, H, W]
                tau = 0.5
                beam_present = (in_vol_aug[:, 0, :, :] != 0)
                paired_mask   = valid_mask & beam_present
                unpaired_mask = valid_mask & ~beam_present

                loss_sifc = torch.tensor(0.0, device=enc_s.device)

                if paired_mask.any():
                    if self.dist_type == 'angular':
                        es_n = F.normalize(enc_s, p=2, dim=1)
                        ea_n = F.normalize(enc_a, p=2, dim=1)
                        cos_dist = 1.0 - (es_n * ea_n).sum(dim=1)
                        loss_sifc = loss_sifc + cos_dist[paired_mask].mean()
                    else:
                        pm = paired_mask.unsqueeze(1).expand_as(enc_s)
                        loss_sifc = loss_sifc + F.l1_loss(enc_s[pm], enc_a[pm])

                if unpaired_mask.any():
                    B, C, H, W = enc_s.shape
                    enc_s_flat = enc_s.permute(0, 2, 3, 1).reshape(B, H*W, C)
                    enc_a_flat = enc_a.permute(0, 2, 3, 1).reshape(B, H*W, C)
                    paired_flat   = paired_mask.reshape(B, H*W)
                    unpaired_flat = unpaired_mask.reshape(B, H*W)

                    agg_losses = []
                    for b in range(B):
                        p_idx = paired_flat[b].nonzero(as_tuple=True)[0]
                        u_idx = unpaired_flat[b].nonzero(as_tuple=True)[0]
                        if len(p_idx) == 0 or len(u_idx) == 0:
                            continue

                        f_s_p = enc_s_flat[b][p_idx]
                        f_a_p = enc_a_flat[b][p_idx]
                        f_s_u = enc_s_flat[b][u_idx]

                        f_s_u_n = F.normalize(f_s_u, p=2, dim=1)
                        f_s_p_n = F.normalize(f_s_p, p=2, dim=1)
                        affinity = f_s_u_n @ f_s_p_n.T

                        affinity = affinity * (affinity >= tau).float()

                        pu = torch.stack([p_idx % W, p_idx // W], dim=1).float()
                        uu = torch.stack([u_idx % W, u_idx // W], dim=1).float()
                        sp_dist = torch.cdist(uu, pu)
                        inv_dist = 1.0 / (sp_dist + 1e-6)
                        weights = affinity * inv_dist
                        weight_sum = weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
                        weights = weights / weight_sum

                        f_agg = weights @ f_a_p

                        valid_agg = weights.sum(dim=1) > 0
                        if valid_agg.any():
                            agg_losses.append(F.l1_loss(f_s_u[valid_agg], f_agg[valid_agg]))

                    if agg_losses:
                        loss_sifc = loss_sifc + torch.stack(agg_losses).mean()

                # SCC Loss
                prototypes_s = self.compute_prototypes(z8, proj_labels)
                prototypes_a = self.compute_prototypes(z8_aug, proj_labels)

                all_protos = torch.cat([prototypes_s, prototypes_a], dim=0)

                valid_mask_proto = all_protos.norm(dim=2) > 0
                shared_valid = valid_mask_proto.all(dim=0)

                loss_scc = torch.tensor(0.0, device=z8.device)

                if shared_valid.sum() >= 2:
                    p = all_protos[:, shared_valid, :]
                    if self.dist_type == 'angular':
                        p = F.normalize(p, p=2, dim=2)
                    corr = torch.bmm(p, p.transpose(1, 2))
                    corr_mean = corr.mean(dim=0, keepdim=True)
                    loss_scc = ((corr - corr_mean) ** 2).mean()

                loss_total = loss_sem + (self.lam1 * loss_sifc) + (self.lam2 * loss_scc)

            optimizer.zero_grad()
            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()

            # Evaluation and Logging
            with torch.no_grad():
                argmax = output.argmax(dim=1)
                evaluator.addBatch(argmax, proj_labels)
                accuracy = evaluator.getacc()
                jaccard, class_jaccard = evaluator.getIoU()

            # Update Meters
            losses.update(loss_total.item(), in_vol.size(0))
            acc.update(accuracy.item(), in_vol.size(0))
            iou.update(jaccard.item(), in_vol.size(0))

            # Measure elapsed time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            curr_train_time = time.time() - start
            train_time.append(curr_train_time)
            
            self.batch_time_t.update(time.time() - end)
            end = time.time()

            # Step scheduler per iteration
            scheduler.step()
            
            if i % report == 0:
                print(f'Epoch: [{epoch}][{i}/{len(train_loader)}] '
                      f'Loss {losses.val:.4f} ({losses.avg:.4f}) '
                      f'Acc {acc.val:.3f} ({acc.avg:.3f}) '
                      f'IoU {iou.val:.3f} ({iou.avg:.3f})')

        print("Mean CNN training time:{:.4f}\t std:{:.4f}".format(np.mean(train_time), np.std(train_time)))
        return acc.avg, iou.avg, losses.avg

    def validate(self, val_loader, model, criterion, evaluator, class_func, color_fn, save_scans):
        losses = AverageMeter()
        jaccs = AverageMeter()
        wces = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()
        rand_imgs = []
        validation_time = []

        # switch to evaluate mode
        model.eval()
        evaluator.reset()

        # empty the cache to infer in high res
        if self.gpu:
            torch.cuda.empty_cache()

        with torch.no_grad():
            end = time.time()
            for i, (in_vol, proj_mask, proj_labels, _, path_seq, path_name, _, _, _, _, _, _, _, _, _) in tqdm(enumerate(val_loader), total=len(val_loader)):
                if not self.multi_gpu and self.gpu:
                    in_vol = in_vol.cuda()
                    proj_mask = proj_mask.cuda()
                if self.gpu:
                    proj_labels = proj_labels.cuda(non_blocking=True).long()

                start = time.time()
                # compute output
                if self.ARCH["train"]["aux_loss"]:
                    output, aux_list, z8 = model(in_vol)
                else:
                    output, z8 = model(in_vol)
                # measure elapsed time
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                res = time.time() - start
                validation_time.append(res)
                start = time.time()

                log_out = torch.log(output.clamp(min=1e-8))
                jacc = self.ls(output, proj_labels)
                wce = criterion(log_out, proj_labels)
                loss = wce + jacc

                argmax = output.argmax(dim=1)
                evaluator.addBatch(argmax, proj_labels)
                losses.update(loss.mean().item(), in_vol.size(0))
                jaccs.update(jacc.mean().item(),in_vol.size(0))

                wces.update(wce.mean().item(),in_vol.size(0))

                self.batch_time_e.update(time.time() - end)
                end = time.time()

                accuracy = evaluator.getacc()
                jaccard, class_jaccard = evaluator.getIoU()
                acc.update(accuracy.item(), in_vol.size(0))
                iou.update(jaccard.item(), in_vol.size(0))

            for i, jacc in enumerate(class_jaccard):
                self.info["valid_classes/" + class_func(i)] = jacc

        return acc.avg, iou.avg, losses.avg, rand_imgs

class Trainer():
    def __init__(self, ARCH, DATA, datadir, logdir, path=None):
        # parameters
        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = datadir
        self.log = logdir
        self.path = path

        self.batch_time_t = AverageMeter()
        self.data_time_t = AverageMeter()
        self.batch_time_e = AverageMeter()
        self.epoch = 0

        # put logger where it belongs

        self.info = {"train_loss": 0,
                     "train_acc": 0,
                     "train_iou": 0,
                     "valid_loss": 0,
                     "valid_acc": 0,
                     "valid_iou": 0,
                     "best_train_iou": 0,
                     "best_val_iou": 0}

        # get the data
        from dataset.kitti.parser import Parser
        self.parser = Parser(root=self.datadir,
                                          train_sequences=self.DATA["split"]["train"], # self.DATA["split"]["valid"] + self.DATA["split"]["train"] if finetune with valid
                                          valid_sequences=self.DATA["split"]["valid"],
                                          test_sequences=None,
                                          labels=self.DATA["labels"],
                                          color_map=self.DATA["color_map"],
                                          learning_map=self.DATA["learning_map"],
                                          learning_map_inv=self.DATA["learning_map_inv"],
                                          sensor=self.ARCH["dataset"]["sensor"],
                                          max_points=self.ARCH["dataset"]["max_points"],
                                          batch_size=self.ARCH["train"]["batch_size"],
                                          workers=self.ARCH["train"]["workers"],
                                          gt=True,
                                          shuffle_train=True)

        # weights for loss (and bias)

        epsilon_w = self.ARCH["train"]["epsilon_w"]
        content = torch.zeros(self.parser.get_n_classes(), dtype=torch.float)
        for cl, freq in DATA["content"].items():
            x_cl = self.parser.to_xentropy(cl)  # map actual class to xentropy class
            content[x_cl] += freq
        self.loss_w = 1 / (content + epsilon_w)  # get weights

        # power_value = 0.25
        # self.loss_w = np.power(self.loss_w, power_value) * np.power(10, 1 - power_value)

        for x_cl, w in enumerate(self.loss_w):  # ignore the ones necessary to ignore
            if DATA["learning_ignore"][x_cl]:
                # don't weigh
                self.loss_w[x_cl] = 0
        print("Loss weights from content: ", self.loss_w.data)

        with torch.no_grad():
            if self.ARCH["train"]["pipeline"] == "hardnet":
                from modules.network.HarDNet import HarDNet
                self.model = HarDNet(self.parser.get_n_classes(), self.ARCH["train"]["aux_loss"])

            if self.ARCH["train"]["pipeline"] == "res":
                from modules.network.ResNet import ResNet_34
                self.model = ResNet_34(self.parser.get_n_classes(), self.ARCH["train"]["aux_loss"])

                def convert_relu_to_softplus(model, act):
                    for child_name, child in model.named_children():
                        if isinstance(child, nn.LeakyReLU):
                            setattr(model, child_name, act)
                        else:
                            convert_relu_to_softplus(child, act)
                if self.ARCH["train"]["act"] == "Hardswish":
                    convert_relu_to_softplus(self.model, nn.Hardswish())
                elif self.ARCH["train"]["act"] == "SiLU":
                    convert_relu_to_softplus(self.model, nn.SiLU())

            if self.ARCH["train"]["pipeline"] == "fid":
                from modules.network.Fid import ResNet_34
                self.model = ResNet_34(self.parser.get_n_classes(), self.ARCH["train"]["aux_loss"])

                if self.ARCH["train"]["act"] == "Hardswish":
                    convert_relu_to_softplus(self.model, nn.Hardswish())
                elif self.ARCH["train"]["act"] == "SiLU":
                    convert_relu_to_softplus(self.model, nn.SiLU())

        # save_to_log(self.log, 'model.txt', str(self.model))
        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Number of parameters: ", pytorch_total_params/1000000, "M")
        # save_to_log(self.log, 'model.txt', "Number of parameters: %.5f M" %(pytorch_total_params/1000000))
        self.tb_logger = SummaryWriter(log_dir=self.log, flush_secs=20)

        # GPU?
        self.gpu = False
        self.multi_gpu = False
        self.n_gpus = 0
        self.model_single = self.model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Training in device: ", self.device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            cudnn.benchmark = True
            cudnn.fastest = True
            self.gpu = True
            self.n_gpus = 1
            self.model.cuda()
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)  # spread in gpus
            self.model = convert_model(self.model).cuda()  # sync batchnorm
            self.model_single = self.model.module  # single model to get weight names
            self.multi_gpu = True
            self.n_gpus = torch.cuda.device_count()


        self.criterion = nn.NLLLoss(weight=self.loss_w).to(self.device)
        self.ls = Lovasz_softmax(ignore=0).to(self.device)
        from modules.losses.boundary_loss import BoundaryLoss
        self.bd = BoundaryLoss().to(self.device)
        # loss as dataparallel too (more images in batch)
        if self.n_gpus > 1:
            self.criterion = nn.DataParallel(self.criterion).cuda()  # spread in gpus
            self.ls = nn.DataParallel(self.ls).cuda()

        # self.optimizer = optim.AdamW(self.model.parameters(), lr=0.0001, weight_decay=0.0005)
        # from modules.adam_policy import MyLR
        # self.scheduler = MyLR(optimizer=self.optimizer, cycle=30)
        # print(self.optimizer)

        if self.ARCH["train"]["scheduler"] == "consine":
            length = self.parser.get_train_size()
            dict = self.ARCH["train"]["consine"]
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=dict["min_lr"],
                                       momentum=self.ARCH["train"]["momentum"],
                                       weight_decay=self.ARCH["train"]["w_decay"])
            self.scheduler = CosineAnnealingWarmUpRestarts(optimizer=self.optimizer,
                                                           T_0=dict["first_cycle"] * length, T_mult=dict["cycle"],
                                                           eta_max=dict["max_lr"],
                                                           T_up=dict["wup_epochs"]*length, gamma=dict["gamma"])

        else:
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=self.ARCH["train"]["decay"]["lr"],
                                       momentum=self.ARCH["train"]["momentum"],
                                       weight_decay=self.ARCH["train"]["w_decay"])
            steps_per_epoch = self.parser.get_train_size()
            up_steps = int(self.ARCH["train"]["decay"]["wup_epochs"] * steps_per_epoch)
            final_decay = self.ARCH["train"]["decay"]["lr_decay"] ** (1 / steps_per_epoch)
            self.scheduler = warmupLR(optimizer=self.optimizer,
                                      lr=self.ARCH["train"]["decay"]["lr"],
                                      warmup_steps=up_steps,
                                      momentum=self.ARCH["train"]["momentum"],
                                      decay=final_decay)

        if self.path is not None:
            torch.nn.Module.dump_patches = True
            w_dict = torch.load(path + "/SENet",
                                map_location=lambda storage, loc: storage)
            self.model.load_state_dict(w_dict['state_dict'], strict=True)
            # self.optimizer.load_state_dict(w_dict['optimizer'])
            # self.epoch = w_dict['epoch'] + 1
            # self.scheduler.load_state_dict(w_dict['scheduler'])
            print("dict epoch:", w_dict['epoch'])
            # self.info = w_dict['info']
            print("info", w_dict['info'])


    def calculate_estimate(self, epoch, iter):
        estimate = int((self.data_time_t.avg + self.batch_time_t.avg) * \
                       (self.parser.get_train_size() * self.ARCH['train']['max_epochs'] - (
                               iter + 1 + epoch * self.parser.get_train_size()))) + \
                   int(self.batch_time_e.avg * self.parser.get_valid_size() * (
                           self.ARCH['train']['max_epochs'] - (epoch)))
        return str(datetime.timedelta(seconds=estimate))

    @staticmethod
    def get_mpl_colormap(cmap_name):
        cmap = plt.get_cmap(cmap_name)
        # Initialize the matplotlib color map
        sm = plt.cm.ScalarMappable(cmap=cmap)
        # Obtain linear color range
        color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]
        return color_range.reshape(256, 1, 3)

    @staticmethod
    def make_log_img(depth, mask, pred, gt, color_fn):
        # input should be [depth, pred, gt]
        # make range image (normalized to 0,1 for saving)
        depth = (cv2.normalize(depth, None, alpha=0, beta=1,
                               norm_type=cv2.NORM_MINMAX,
                               dtype=cv2.CV_32F) * 255.0).astype(np.uint8)
        out_img = cv2.applyColorMap(
            depth, Trainer.get_mpl_colormap('viridis')) * mask[..., None]
        # make label prediction
        pred_color = color_fn((pred * mask).astype(np.int32))
        out_img = np.concatenate([out_img, pred_color], axis=0)
        # make label gt
        gt_color = color_fn(gt)
        out_img = np.concatenate([out_img, gt_color], axis=0)
        return (out_img).astype(np.uint8)

    @staticmethod
    def save_to_log(logdir, logger, info, epoch, w_summary=False, model=None, img_summary=False, imgs=[]):
        # save scalars
        for tag, value in info.items():
            logger.add_scalar(tag, value, epoch)

        # save summaries of weights and biases
        if w_summary and model:
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
                if value.grad is not None:
                    logger.histo_summary(
                        tag + '/grad', value.grad.data.cpu().numpy(), epoch)

        if img_summary and len(imgs) > 0:
            directory = os.path.join(logdir, "predictions")
            if not os.path.isdir(directory):
                os.makedirs(directory)
            for i, img in enumerate(imgs):
                name = os.path.join(directory, str(i) + ".png")
                cv2.imwrite(name, img)

    def train(self, epochs=None):

        self.ignore_class = []
        for i, w in enumerate(self.loss_w):
            if w < 1e-10:
                self.ignore_class.append(i)
                print("Ignoring class ", i, " in IoU evaluation")
        self.evaluator = iouEval(self.parser.get_n_classes(),
                                 self.device, self.ignore_class)
        # save_to_log(self.log, 'log.txt', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        if self.path is not None:
            acc, iou, loss, rand_img = self.validate(val_loader=self.parser.get_valid_set(),
                                             model=self.model,
                                             criterion=self.criterion,
                                             evaluator=self.evaluator,
                                             class_func=self.parser.get_xentropy_class_string,
                                             color_fn=self.parser.to_color,
                                             save_scans=self.ARCH["train"]["save_scans"])

        # train for n epochs
        max_epochs = epochs if epochs is not None else self.ARCH["train"]["max_epochs"]
        for epoch in range(self.epoch, max_epochs):
            # train for 1 epoch

            acc, iou, loss = self.train_epoch(train_loader=self.parser.get_train_set(),
                                                           model=self.model,
                                                           criterion=self.criterion,
                                                           optimizer=self.optimizer,
                                                           epoch=epoch,
                                                           evaluator=self.evaluator,
                                                           scheduler=self.scheduler,
                                                           color_fn=self.parser.to_color,
                                                           report=self.ARCH["train"]["report_batch"],
                                                           show_scans=self.ARCH["train"]["show_scans"])


            # update info
            self.info["train_loss"] = loss
            self.info["train_acc"] = acc
            self.info["train_iou"] = iou

            # remember best iou and save checkpoint
            state = {'epoch': epoch, 'state_dict': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     'info': self.info,
                     'scheduler': self.scheduler.state_dict()
                     }
            save_checkpoint(state, self.log, suffix="")
            # save_checkpoint(state, self.log, suffix=""+str(epoch))

            if self.info['train_iou'] > self.info['best_train_iou']:
                # save_to_log(self.log, 'log.txt', "Best mean iou in training set so far, save model!")
                print("Best mean iou in training set so far, save model!")
                self.info['best_train_iou'] = self.info['train_iou']
                state = {'epoch': epoch, 'state_dict': self.model.state_dict(),
                         'optimizer': self.optimizer.state_dict(),
                         'info': self.info,
                         'scheduler': self.scheduler.state_dict()
                         }
                save_checkpoint(state, self.log, suffix="_train_best")

            if epoch % self.ARCH["train"]["report_epoch"] == 0:
                # evaluate on validation set
                print("*" * 80)
                acc, iou, loss, rand_img = self.validate(val_loader=self.parser.get_valid_set(),
                                                         model=self.model,
                                                         criterion=self.criterion,
                                                         evaluator=self.evaluator,
                                                         class_func=self.parser.get_xentropy_class_string,
                                                         color_fn=self.parser.to_color,
                                                         save_scans=self.ARCH["train"]["save_scans"])

                # update info
                self.info["valid_loss"] = loss
                self.info["valid_acc"] = acc
                self.info["valid_iou"] = iou

            # remember best iou and save checkpoint
            if self.info['valid_iou'] > self.info['best_val_iou']:
                # save_to_log(self.log, 'log.txt', "Best mean iou in validation so far, save model!")
                print("Best mean iou in validation so far, save model!")
                print("*" * 80)
                self.info['best_val_iou'] = self.info['valid_iou']

                # save the weights!
                state = {'epoch': epoch, 'state_dict': self.model.state_dict(),
                         'optimizer': self.optimizer.state_dict(),
                         'info': self.info,
                         'scheduler': self.scheduler.state_dict()
                         }
                save_checkpoint(state, self.log, suffix="_valid_best")

            print("*" * 80)

            # save to log
            # Trainer.save_to_log(logdir=self.log,
            #                     logger=self.tb_logger,
            #                     info=self.info,
            #                     epoch=epoch,
            #                     w_summary=self.ARCH["train"]["save_summary"],
            #                     model=self.model_single,
            #                     img_summary=self.ARCH["train"]["save_scans"],
            #                     imgs=rand_img)
            # save_to_log(self.log, 'log.txt', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print('Finished Training')
        # save_to_log(self.log, 'log.txt', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        return

    def train_epoch(self, train_loader, model, criterion, optimizer, epoch, evaluator, scheduler, color_fn, report=10,
                    show_scans=False):
        losses = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()
        update_ratio_meter = AverageMeter()
        bd = AverageMeter()
        train_time = []

        # empty the cache to train now
        if self.gpu:
            torch.cuda.empty_cache()

        scaler = torch.amp.GradScaler('cuda')

        # switch to train mode
        model.train()

        end = time.time()
        for i, (in_vol, proj_mask, proj_labels, _, path_seq, path_name, _, _, _, _, _, _, _, _, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
            # measure data loading time
            self.data_time_t.update(time.time() - end)
            if not self.multi_gpu and self.gpu:
                in_vol = in_vol.cuda()
            if self.gpu:
                proj_labels = proj_labels.cuda().long()

                # proj_labels = proj_labels.unsqueeze(1).type(torch.FloatTensor)
                # from torch.nn import functional as F
                # [n, c, h, w] = proj_labels.size()
                # proj_labels_8 = F.interpolate(proj_labels, size=(h//8, w//8), mode='nearest').squeeze(1).cuda().long()
                # proj_labels_4 = F.interpolate(proj_labels, size=(h//4, w//4), mode='nearest').squeeze(1).cuda().long()
                # proj_labels_2 = F.interpolate(proj_labels, size=(h//2, w//2), mode='nearest').squeeze(1).cuda().long()
                # proj_labels = proj_labels.squeeze(1).cuda().long()

            start = time.time()
            with torch.amp.autocast('cuda'):
                model_output = model(in_vol)

                if self.ARCH["train"]["aux_loss"]:

                    output, aux_list, _ = model_output
                    z2, z4, z8 = aux_list
                    
                    lamda = self.ARCH["train"]["lamda"]

                    bdlosss = (self.bd(output, proj_labels.long()) + 
                            lamda * self.bd(z2, proj_labels.long()) + 
                            lamda * self.bd(z4, proj_labels.long()) + 
                            lamda * self.bd(z8, proj_labels.long()))

                    loss_m0 = criterion(torch.log(output.clamp(min=1e-8)), proj_labels) + 1.5 * self.ls(output, proj_labels.long())
                    loss_m2 = criterion(torch.log(z2.clamp(min=1e-8)), proj_labels) + 1.5 * self.ls(z2, proj_labels.long())
                    loss_m4 = criterion(torch.log(z4.clamp(min=1e-8)), proj_labels) + 1.5 * self.ls(z4, proj_labels.long())
                    loss_m8 = criterion(torch.log(z8.clamp(min=1e-8)), proj_labels) + 1.5 * self.ls(z8, proj_labels.long())
                    
                    loss_m = loss_m0 + lamda*loss_m2 + lamda*loss_m4 + lamda*loss_m8 + bdlosss
                else:
                    output, _ = model_output
                    bdlosss = self.bd(output, proj_labels.long())
                    loss_m = criterion(torch.log(output.clamp(min=1e-8)), proj_labels) + 1.5 * self.ls(output, proj_labels.long()) + bdlosss

            optimizer.zero_grad()

            # if self.n_gpus > 1:
            #     idx = torch.ones(self.n_gpus).cuda()
            #     loss_m.backward(idx)
            # else:
            #     loss_m.backward()
            # optimizer.step()

            '''

            scaler.scale(loss_m).backward()
            scaler.step(optimizer)
            scaler.update()

            # measure accuracy and record loss
            loss = loss_m.mean()
            '''
            loss = loss_m.mean()  # 🔄 先转成 scalar

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()




            with torch.no_grad():
                evaluator.reset()
                argmax = output.argmax(dim=1)
                evaluator.addBatch(argmax, proj_labels)
                accuracy = evaluator.getacc()
                jaccard, class_jaccard = evaluator.getIoU()

            losses.update(loss.item(), in_vol.size(0))
            acc.update(accuracy.item(), in_vol.size(0))
            iou.update(jaccard.item(), in_vol.size(0))
            bd.update(bdlosss.item(), in_vol.size(0))

            # measure elapsed time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            res = time.time() - start
            train_time.append(res)
            start = time.time()
            self.batch_time_t.update(time.time() - end)
            end = time.time()

            # get gradient updates and weights, so I can print the relationship of
            # their norms
            update_ratios = []
            for g in self.optimizer.param_groups:
                lr = g["lr"]

            # if show_scans:
            #     if i % self.ARCH["train"]["save_batch"] == 0:
            #         # get the first scan in batch and project points
            #         mask_np = proj_mask[0].cpu().numpy()
            #         depth_np = in_vol[0][0].cpu().numpy()
            #         pred_np = argmax[0].cpu().numpy()
            #         gt_np = proj_labels[0].cpu().numpy()
            #         out = Trainer.make_log_img(depth_np, mask_np, pred_np, gt_np, color_fn)

            #         directory = os.path.join(self.log, "train-predictions")
            #         if not os.path.isdir(directory):
            #             os.makedirs(directory)
            #         name = os.path.join(directory, str(i) + ".png")
            #         cv2.imwrite(name, out)


            # if i % self.ARCH["train"]["report_batch"] == 0:
            #     print('Lr: {lr:.3e} | '
            #           'Epoch: [{0}][{1}/{2}] | '
            #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
            #           'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
            #           'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
            #           'Bd {bd.val:.4f} ({bd.avg:.4f}) | '
            #           'acc {acc.val:.3f} ({acc.avg:.3f}) | '
            #           'IoU {iou.val:.3f} ({iou.avg:.3f}) | [{estim}]'.format(
            #         epoch, i, len(train_loader), batch_time=self.batch_time_t,
            #         data_time=self.data_time_t, loss=losses, bd=bd, acc=acc, iou=iou, lr=lr,
            #         estim=self.calculate_estimate(epoch, i)))

            #     save_to_log(self.log, 'log.txt', 'Lr: {lr:.3e} | '
            #                                      'Epoch: [{0}][{1}/{2}] | '
            #                                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
            #                                      'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
            #                                      'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
            #                                      'Bd {bd.val:.4f} ({bd.avg:.4f}) | '
            #                                      'acc {acc.val:.3f} ({acc.avg:.3f}) | '
            #                                      'IoU {iou.val:.3f} ({iou.avg:.3f}) | [{estim}]'.format(
            #         epoch, i, len(train_loader), batch_time=self.batch_time_t,
            #         data_time=self.data_time_t, loss=losses, bd=bd, acc=acc, iou=iou, lr=lr,
            #         estim=self.calculate_estimate(epoch, i)))
            # step scheduler
            scheduler.step()
        # scheduler.step()
        print("Mean CNN training time:{}\t std:{}".format(np.mean(train_time), np.std(train_time)))
        return acc.avg, iou.avg, losses.avg

    def validate(self, val_loader, model, criterion, evaluator, class_func, color_fn, save_scans):
        losses = AverageMeter()
        jaccs = AverageMeter()
        wces = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()
        rand_imgs = []
        validation_time = []

        model.eval()
        evaluator.reset()

        if self.gpu:
            torch.cuda.empty_cache()

        with torch.no_grad():
            end = time.time()
            for i, (in_vol, proj_mask, proj_labels, _, path_seq, path_name, _, _, _, _, _, _, _, _, _) in tqdm(enumerate(val_loader), total=len(val_loader)):
                if not self.multi_gpu and self.gpu:
                    in_vol = in_vol.cuda()
                    proj_mask = proj_mask.cuda()
                if self.gpu:
                    proj_labels = proj_labels.cuda(non_blocking=True).long()

                start = time.time()

                if self.ARCH["train"]["aux_loss"]:
                    output, aux_list, _ = model(in_vol)
                    z2, z4, z8 = aux_list
                else:
                    output, _ = model(in_vol)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                validation_time.append(time.time() - start)

                log_out = torch.log(output.clamp(min=1e-8))
                jacc = self.ls(output, proj_labels)
                wce = criterion(log_out, proj_labels)
                loss = wce + jacc

                argmax = output.argmax(dim=1)
                evaluator.addBatch(argmax, proj_labels)
                losses.update(loss.mean().item(), in_vol.size(0))
                jaccs.update(jacc.mean().item(), in_vol.size(0))
                wces.update(wce.mean().item(), in_vol.size(0))

                accuracy = evaluator.getacc()
                jaccard, class_jaccard = evaluator.getIoU()
                acc.update(accuracy.item(), in_vol.size(0))
                iou.update(jaccard.item(), in_vol.size(0))

                self.batch_time_e.update(time.time() - end)
                end = time.time()

            for i, jacc in enumerate(class_jaccard):
                self.info["valid_classes/" + class_func(i)] = jacc

        return acc.avg, iou.avg, losses.avg, rand_imgs