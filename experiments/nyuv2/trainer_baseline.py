import argparse
from collections import defaultdict
import logging
import copy

import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from tqdm import trange
from torchsummary import summary

from experiments.nyuv2.data import nyu_dataloaders
from experiments.nyuv2.metrics import compute_iou, compute_miou
from experiments.nyuv2.model import SegNetSplit
from experiments.utils import (get_device, set_logger, set_seed)
from experiments.weight_methods import WeightMethods


parser = argparse.ArgumentParser(description='NYU - Baselines')
parser.add_argument(
    '--method',
    default='equal',
    type=str,
    choices=['equal', 'uncert', 'dwa', 'cosine', 'stl', 'gradnorm'],
    help='multi-task weighting or stl: equal, uncert, dwa, cosine, gradnorm'
)
parser.add_argument('--dataroot', default='/nyuv2', type=str, help='dataset root')
parser.add_argument('--temp', default=2.0, type=float, help='temperature for DWA (must be positive)')
parser.add_argument('--grad-norm-alpha', default=1.5, type=float, help='alpha for gradNorm')
parser.add_argument('--weights-lr', default=None, type=float, help='learning rate for weights in gradNorm and Uncert')
parser.add_argument('--eval-every', type=int, default=1, help='num. epochs between test set eval')
parser.add_argument('--seed', type=int, default=45, help='random seed')
args = parser.parse_args()

# set seed - for reproducibility
set_seed(args.seed)
# logger config
set_logger()

# ======
# params
# ======
num_epochs = 200
batch_size = 8
val_batch_size = 2

# =========
# load data
# =========
nyuv2_train_loader, nyuv2_val_loader, nyuv2_test_loader = nyu_dataloaders(
    datapath=args.dataroot,
    validation_indices='./hpo_validation_indices.json',
    aux_set=False,
    batch_size=batch_size,
    val_batch_size=val_batch_size
)

# ================
# Weighting Method
# ================
main_task = 0  # segmentation
n_train_batch = len(nyuv2_train_loader)
T = args.temp
device = get_device()
weights_lr = args.weights_lr if args.weights_lr is not None else 1e-4

if args.method == 'gradnorm':
    weights_lr = 0.025
    logging.info("For GradNorm the default lr for weights is 0.025, like in the GradNorm paper")

weighting_method = WeightMethods(
    method=args.method,
    n_tasks=3,
    alpha=args.grad_norm_alpha,
    temp=T,
    n_train_batch=n_train_batch,
    n_epochs=num_epochs,
    main_task=main_task,
    device=device
)


# ====
# loss
# ====
def calc_loss(seg_pred, seg, depth_pred, depth, normal_pred, normal):
    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(depth, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).to(depth.device)

    # semantic loss: depth-wise cross entropy
    seg_loss = F.nll_loss(seg_pred, seg, ignore_index=-1)

    n_nonzeros = torch.nonzero(binary_mask).size(0)
    # depth loss: l1 norm
    depth_loss = torch.sum(torch.abs(depth_pred - depth) * binary_mask) / n_nonzeros

    # normal loss: dot product
    normal_loss = 1 - torch.sum((normal_pred * normal) * binary_mask) / n_nonzeros

    return [seg_loss, depth_loss, normal_loss]


# =====
# model
# =====
SegNet_SPLIT = SegNetSplit(logsigma=args.method == 'uncert')
summary(SegNet_SPLIT, input_size=(3, 288, 384), device='cpu')
SegNet_SPLIT = SegNet_SPLIT.to(device)


# ========
# evaluate
# ========
def evaluate(dataloader, model=None):
    model = model if model is not None else SegNet_SPLIT
    model.eval()
    total = 0
    eval_dict = defaultdict(float)

    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            batch = (t.to(device) for t in batch)
            eval_data, eval_label, eval_depth, eval_normal = batch
            eval_label = eval_label.type(torch.LongTensor).to(device)

            eval_pred = model(eval_data)
            # loss
            eval_loss = calc_loss(
                eval_pred[0],
                eval_label,
                eval_pred[1],
                eval_depth,
                eval_pred[2],
                eval_normal
            )

            curr_batch_size = eval_data.shape[0]
            total += curr_batch_size
            curr_eval_dict = dict(
                seg_loss=eval_loss[0].item() * curr_batch_size,
                seg_miou=compute_miou(eval_pred[0], eval_label).item() * curr_batch_size,
                seg_pixacc=compute_iou(eval_pred[0], eval_label).item() * curr_batch_size
            )

            for k, v in curr_eval_dict.items():
                eval_dict[k] += v

    for k, v in eval_dict.items():
        eval_dict[k] = v / total

    model.train()

    return eval_dict


# last shared layer for GradNorm
def get_last_shared_layer():
    last_shared_layer = SegNet_SPLIT.conv_block_dec[-5][0].parameters()
    return list(last_shared_layer)


# ==========
# optimizers
# ==========
if args.method == 'gradnorm':
    # add weights to optimizer
    param_groups = [
        {'params': SegNet_SPLIT.parameters()},
        {'params': weighting_method.method.weights, 'lr': weights_lr}
    ]
    optimizer = optim.Adam(param_groups, lr=1e-4)

elif args.method == 'uncert':
    # add weights to optimizer
    non_logsizgma_params = [p for n, p in SegNet_SPLIT.named_parameters() if n != 'logsigma']
    param_groups = [
        {'params': non_logsizgma_params},
        {'params': SegNet_SPLIT.logsigma, 'lr': weights_lr}
    ]

    optimizer = optim.Adam(param_groups, lr=1e-4)

else:
    optimizer = optim.Adam(SegNet_SPLIT.parameters(), lr=1e-4)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)


# ==========
# train loop
# ==========
best_metric = np.NINF
best_model_epoch = 0
best_miou, best_pixacc = 0, 0
step = 0
logsigma = None

# logging
logging.info(f"Weighting scheme: {args.method}")
epoch_iter = trange(num_epochs)
for epoch in epoch_iter:
    # iteration for all batches
    SegNet_SPLIT.train()
    for k, batch in enumerate(nyuv2_train_loader):
        step += 1

        batch = (t.to(device) for t in batch)
        train_data, train_label, train_depth, train_normal = batch
        train_label = train_label.type(torch.LongTensor).to(device)

        if args.method == 'uncert':
            train_pred, logsigma = SegNet_SPLIT(train_data)
        else:
            train_pred = SegNet_SPLIT(train_data)

        optimizer.zero_grad()
        train_loss = calc_loss(
            train_pred[0],
            train_label,
            train_pred[1],
            train_depth,
            train_pred[2],
            train_normal
        )

        # weights for Cosine and GradNorm
        shared_parameters = [
            p for n, p in SegNet_SPLIT.named_parameters() if 'task' not in n
        ] if args.method == 'cosine' else None
        last_shared_layer = get_last_shared_layer() if args.method == 'gradnorm' else None

        # weight losses and backward
        loss = weighting_method.backwards(
            train_loss,
            epoch=epoch,
            logsigmas=logsigma,
            shared_parameters=shared_parameters,
            last_shared_params=last_shared_layer,
            returns=True
        )

        epoch_iter.set_description(f'[{epoch} {k}] Training loss {loss.data.cpu().numpy().item():.5f}')

        # update parameters
        optimizer.step()

    scheduler.step(epoch=epoch)

    if (epoch + 1) % args.eval_every == 0:
        val_metrics = evaluate(nyuv2_val_loader)
        test_metrics = evaluate(nyuv2_test_loader)

        logging.info(
            f"Epoch: {epoch + 1}, Test mIoU = {test_metrics['seg_miou']:.4f}, "
            f"Test PixAcc = {test_metrics['seg_pixacc']:.4f}"
        )

        if val_metrics["seg_miou"] >= best_metric:
            logging.info(f"Saving model, epoch {epoch + 1}")
            best_model_epoch = epoch + 1
            best_metric = val_metrics["seg_miou"]
            best_model = copy.deepcopy(SegNet_SPLIT)

# final evaluation
logging.info(f"End of training, best model from epoch {best_model_epoch}")

test_metrics = evaluate(nyuv2_test_loader, model=best_model)
logging.info(
    f"Epoch: {epoch + 1}, Test mIoU = {test_metrics['seg_miou']:.4f}, "
    f"Test PixAcc = {test_metrics['seg_pixacc']:.4f}"
)
