import argparse
import logging
from io import BytesIO
import copy

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import classification_report
import torch.nn.functional as F
from torchsummary import summary
from tqdm import trange
from torch.utils.data import DataLoader
from experiments.oxford_pet.data import Oxford_Pet
from experiments.oxford_pet.models import ResNet18, AuxiliaryNet
from experiments.utils import (get_device, set_logger, set_seed, str2bool, topk)
from auxilearn.optim import MetaOptimizer
from auxilearn.hypernet import (MonoHyperNet, MonoLinearHyperNet, MonoNonlinearHyperNet)
torch.set_printoptions(profile="full")


parser = argparse.ArgumentParser(description='Oxford-Pet - trainer')
parser.add_argument('--script-name', default='Oxford-Pet')
parser.add_argument('--exp_name', type=str, default='', metavar='N',
                    help='experiment name suffix')
parser.add_argument('--num-epochs', type=int, default=150, help='Number of epochs')
parser.add_argument('--dataroot', type=str, default='./dataset', help='dataset root')
parser.add_argument('--scheduler', default=True, type=str2bool, help='learning rate scheduler')
parser.add_argument('--aux-scale', default=.25, type=float, help='auxiliary task scale')
parser.add_argument('--optimizer', default="sgd", choices=['adam', 'sgd'], type=str)
parser.add_argument('--aux-optim', type=str, default='sgd', choices=['adam', 'sgd'],  help="auxilearn optimizer")
parser.add_argument('--aux_net', type=str, choices=['linear', 'nonlinear'], default='linear')
parser.add_argument('--nonlinear-init-upper', default=.2, type=float, help='upper value for nonlinear net')
parser.add_argument('--hidden-dim', type=lambda s: [item.strip() for item in s.split(',')], default='10,10',
                    help="List of hidden dims for nonlinear")
parser.add_argument('--linear-activation', type=str2bool, nargs='?', const=True, default=False, help="for deep linear net")
parser.add_argument('--skip', type=str2bool, default=True, help='skip connection on linear network')
parser.add_argument('--weightnorm', type=str2bool, default=False, help='apply weight norm on layers')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--wd', default=5e-3, type=float, help='weight decay')
parser.add_argument('--auxgrad-every', type=int, default=30, help="number of opt. steps between aux params update")
parser.add_argument('--num-aux-classes', type=int, default=5, help='number of classes for aux task')
parser.add_argument('--aux-lr', type=float, default=1e-3, help='Auxiliary learning rate')
parser.add_argument('--aux-momentum', type=float, default=.9, help='Auxiliary momentum')
parser.add_argument('--aux-wd', type=float, default=5e-3, help='Auxiliary weght decay')
parser.add_argument('--samples-per-class', default=30, type=int, help='number of samples per class in train with label')
parser.add_argument('--aux-set-size', default=0.0084, type=float, help='pct of samples to allocate for aux. set')
parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
parser.add_argument('--test-batch-size', type=int, default=128, help='Batch size')
parser.add_argument('--eta-min', type=float, default=0., help='min eta')
parser.add_argument('--t-max', type=int, default=None, help='t max')
parser.add_argument('--out-dir', type=str, default='./outputs', help='Output dir')
parser.add_argument('--seed', default=42, type=int, help='random seed')
parser.add_argument('--gpus', default='0', type=str, help='default gpu')
parser.add_argument('--auxiliary-pretrained', type=str2bool, nargs='?', const=True, default=False,
                    help="label generator from pretrained")
parser.add_argument('--primary-pretrained', type=str2bool, nargs='?', const=True, default=False,
                    help="pretrained primary network")

args = parser.parse_args()

# set seed
set_seed(args.seed)

# logger config
set_logger()

exp_name = f'pet_learn_aux_seed_{args.seed}_lr_{args.lr}_wd_{args.wd}_aux_lr_{args.aux_lr}' \
        f'_aux_wd_{args.aux_wd}_samples_{args.samples_per_class}_aux_size_{args.aux_set_size}'
exp_name += '_' + args.exp_name

logging.info(str(args))

num_classes = 37
main_task = 0

# =========
# load data
# =========
batch_size = args.batch_size
test_batch_size = args.test_batch_size

datasets = Oxford_Pet(root=args.dataroot)
# data splits
trainset, valset, testset, auxset = datasets.get_datasets(train_shot=args.samples_per_class,
                                                              aux_set_size=args.aux_set_size)

train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
val_loader = DataLoader(valset, batch_size=test_batch_size, num_workers=4)
test_loader = DataLoader(testset, batch_size=test_batch_size, num_workers=4)
auxiliary_loader = DataLoader(auxset, batch_size=batch_size, num_workers=4, shuffle=True)

n_train_batch = len(train_loader)
device = get_device(no_cuda=args.gpus == "-1", gpus=args.gpus)


# ====
# loss
# ====
def calc_loss(x_pred, x_output, num_output, pri=True):
    """Focal loss
    :param x_pred:  prediction of primary network (either main or auxiliary)
    :param x_output: label
    :param pri: is primary task output?
    :param num_output: number of classes
    :return: loss per sample
    """
    if not pri:
        # generated auxiliary label is a soft-assignment vector (no need to change into one-hot vector)
        x_output_onehot = x_output
    else:
        # convert a single label into a one-hot vector
        x_output_onehot = torch.zeros((len(x_output), num_output)).to(x_pred.device)
        x_output_onehot.scatter_(1, x_output.unsqueeze(1), 1)
        x_pred = F.softmax(x_pred, dim=1)

    loss = torch.sum(- x_output_onehot * (1 - x_pred) ** 2 * torch.log(x_pred + 1e-12), dim=1)

    return loss

# =====
# model
# =====
assert args.num_aux_classes > 1
psi = [args.num_aux_classes] * num_classes
param_net = ResNet18(psi=psi, pretrained=args.primary_pretrained, num_classes=num_classes)

# print summary
summary(param_net, input_size=(3, 224, 224), device='cpu')
param_net = param_net.to(device)

# ================
# hyperparam model
# ================
auxiliary_generate_net = AuxiliaryNet(psi=psi, pretrained=args.auxiliary_pretrained)
auxiliary_generate_net = auxiliary_generate_net.to(device)

# ============================
# AuxiLearn Combine losses
# ============================
hypernet_mapping = dict(
    linear=MonoLinearHyperNet,
    nonlinear=MonoNonlinearHyperNet,
)

auxnet_mapping = dict(
    linear=MonoLinearHyperNet,
    nonlinear=MonoNonlinearHyperNet,
)

auxnet_config = dict(input_dim=2, main_task=0, weight_normalization=args.weightnorm)

if args.aux_net == 'nonlinear':
    auxnet_config['hidden_sizes'] = [int(l) for l in args.hidden_dim]
    auxnet_config['init_upper'] = args.nonlinear_init_upper
else:
    auxnet_config['skip_connection'] = args.skip

auxiliary_combine_net = auxnet_mapping[args.aux_net](**auxnet_config)
auxiliary_combine_net = auxiliary_combine_net.to(device)

# ==========
# optimizers
# ==========
optimizer = optim.SGD(param_net.parameters(), lr=args.lr, weight_decay=args.wd, momentum=.9) \
            if args.optimizer == 'sgd' \
            else optim.Adam(param_net.parameters(), lr=args.lr, weight_decay=args.wd)

if args.scheduler:
    t_max = args.num_epochs if args.t_max is None else args.t_max
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=args.eta_min)

auxiliary_params = list(auxiliary_generate_net.parameters())
auxiliary_params += list(auxiliary_combine_net.parameters())


meta_opt = optim.SGD(auxiliary_params, lr=args.aux_lr, momentum=args.aux_momentum, weight_decay=args.aux_wd) \
           if args.aux_optim == 'sgd' \
           else optim.Adam(auxiliary_params, lr=args.aux_lr, weight_decay=args.aux_wd)
    
meta_optimizer = MetaOptimizer(
    meta_optimizer=meta_opt, hpo_lr=args.lr, truncate_iter=3, max_grad_norm=50
)


# ================
# helper functions
# ================
# evaluating test/val data
def model_evalution(loader, print_clf_report=False):
    param_net.eval()

    targets = []
    preds = []

    acc_loss = 0.
    num_samples = 0
    with torch.no_grad():  # operations inside don't track history
        for k, batch in enumerate(loader):
            batch = (t.to(device) for t in batch)
            data, clf_labels = batch

            pred, _ = param_net(data)

            targets.append(clf_labels)
            preds.append(pred)

            loss = calc_loss(pred, clf_labels, pri=True, num_output=num_classes).sum()

            num_samples += clf_labels.shape[0]
            acc_loss += loss

        target = torch.cat(targets, dim=0).detach().cpu().numpy()
        pred = torch.cat(preds, dim=0)[:, :num_classes].detach().cpu().numpy()
        max_logit = pred.argmax(1)
        pred_top3 = topk(target, pred, 3)
        pred_top5 = topk(target, pred, 5)

    clf_report = classification_report(target, max_logit, output_dict=True)
    if print_clf_report:
        logging.info('\n' + classification_report(target, max_logit))

    param_net.train()
    return acc_loss/num_samples, clf_report, [pred_top3, pred_top5]


def model_save(model, file=None):
    if file is None:
        file = BytesIO()
    torch.save(model.state_dict(), file)
    return file


# ==============
# hypergrad step
# ==============
def hyperstep():
    meta_val_loss = .0
    for n_val_step, val_batch in enumerate(auxiliary_loader):
        val_batch = (t.to(device) for t in val_batch)
        data, clf_labels = val_batch

        val_pred, val_labels = param_net(data)

        val_loss = calc_loss(val_pred, clf_labels, pri=True, num_output=num_classes).mean()
        meta_val_loss += val_loss
        break

    inner_loop_end_train_loss = 0.
    for n_train_step, train_batch in enumerate(train_loader):
        # to device and take only first val_batch_size
        train_batch = (t.to(device)[:batch_size, ] for t in train_batch)

        train_data, train_target = train_batch
        train_main_pred, train_aux_pred = param_net(train_data)
        train_aux_target = auxiliary_generate_net(train_data, train_target)

        inner_train_loss_main = calc_loss(train_main_pred, train_target, pri=True,
                                          num_output=num_classes)
        inner_train_loss_aux = calc_loss(train_aux_pred, train_aux_target, pri=False,
                                         num_output=num_classes * args.num_aux_classes) * args.aux_scale

        inner_loop_end_train_loss += auxiliary_combine_net(
            torch.stack((inner_train_loss_main, inner_train_loss_aux)).t()
        )
        break

    phi = list(auxiliary_generate_net.parameters())
    phi += list(auxiliary_combine_net.parameters())
    W = [p for n, p in param_net.named_parameters() if 'classifier2' not in n]

    curr_hypergrads = meta_optimizer.step(
        val_loss=meta_val_loss,
        train_loss=inner_loop_end_train_loss,
        aux_params=phi,
        parameters=W,
        return_grads=True
    )

    return curr_hypergrads


# ==========
# train loop
# ==========
best_main_val_loss = np.inf
step = hp_index = 0
best_val_acc = best_epoch = 0
last_eval_index = -1
list_train_losses = []
epoch_iter = trange(args.num_epochs)

# logging
logging.info(
    f"Train examples: {len(train_loader.dataset)}, val examples: "
    f"{len(val_loader.dataset)}, test examples {len(test_loader.dataset)}"
)

# logging
logging.info(f"Number of meta steps {int((args.num_epochs * n_train_batch) / args.auxgrad_every)}")


for epoch in epoch_iter:
    index = epoch
    targets = []
    preds = []
    curr_loss = 0.
    curr_main_loss = 0.
    num_samples = 0

    # iteration for all batches
    param_net.train()

    for k, batch in enumerate(train_loader):
        step += 1
        optimizer.zero_grad()

        batch = (t.to(device) for t in batch)
        train_data, train_label = batch

        main_pred, aux_pred = param_net(train_data)
        aux_label = auxiliary_generate_net(train_data, train_label)

        train_loss_main = calc_loss(main_pred, train_label, pri=True, num_output=num_classes)
        train_loss_aux = calc_loss(aux_pred, aux_label, pri=False,
                                   num_output=num_classes * args.num_aux_classes) * args.aux_scale

        targets.append(train_label.cpu().numpy())
        preds.append(main_pred.detach().cpu().numpy())

        # mean over batch and sum over task losses
        loss = auxiliary_combine_net(torch.stack((train_loss_main, train_loss_aux)).t())

        main_loss_data = train_loss_main.mean().item()
        aux_loss_data = train_loss_aux.mean().item()

        # logging
        epoch_iter.set_description(f'[{epoch} {k}] Training loss {loss.data.cpu().numpy().item():.5f}')
        loss.backward()
        optimizer.step()

        num_samples += train_label.shape[0]
        curr_loss += loss.item() * train_label.shape[0]
        curr_main_loss += main_loss_data * train_label.shape[0]

        # hyperparams step
        if step % args.auxgrad_every == 0:
            curr_hypergrads = hyperstep()
            if isinstance(auxiliary_combine_net, MonoHyperNet):
                # monotonic network
                auxiliary_combine_net.clamp()

    if args.scheduler:
        scheduler.step(epoch=epoch)
        lrs = scheduler.get_lr()
        logging.info(f"learning rate is {lrs[0]:.8f}")

    target = np.concatenate(targets, axis=0)
    full_pred = np.concatenate(preds, axis=0)[:, :num_classes].argmax(1)
    train_clf_report = classification_report(target, full_pred, output_dict=True)

    aux_set_loss, aux_clf_report, _ = model_evalution(auxiliary_loader)
    eval_loss, clf_report, preds_top = model_evalution(val_loader)
    if clf_report['accuracy'] >= best_val_acc:
        logging.info(f"Saving model, epoch {epoch + 1}")
        best_val_clf_report = copy.deepcopy(clf_report)
        best_model = copy.deepcopy(param_net)
        best_epoch = epoch + 1
        best_main_val_loss = eval_loss
        best_val_acc = clf_report['accuracy']

    logging.info(f"Epoch: {epoch}, "
                 f"Train weighted loss: {curr_loss / num_samples:.5f}, "
                 f"Train main loss: {curr_main_loss / num_samples:.5f}, "
                 f"Train Accuracy: {train_clf_report['accuracy']:.5f}")

    logging.info(f"Epoch: {epoch}, "
                 f"Meta Val total loss: {aux_set_loss:.5f}, "
                 f"Meta Val Accuracy: {aux_clf_report['accuracy']:.5f}")

    logging.info(f"Epoch: {epoch}, "
                 f"Val total loss: {eval_loss:.5f}, "
                 f"Val Accuracy: {clf_report['accuracy']:.5f}")


param_net = best_model
test_loss, clf_report, _ = model_evalution(test_loader, print_clf_report=True)
logging.info(f"Best epoch: {best_epoch}, Test total loss: {test_loss:.5f}")
logging.info(f"Best epoch: {best_epoch}, Test acc: {clf_report['accuracy']:.5f}, "
             f"Val acc: {best_val_clf_report['accuracy']:.5f}")
