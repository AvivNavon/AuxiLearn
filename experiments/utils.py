import logging
import random
import numpy as np
import torch
import argparse

def set_logger():
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )


def set_seed(seed):
    """for reproducibility

    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(no_cuda=False, gpus='0'):
    return torch.device("cuda:" + gpus if torch.cuda.is_available() and not no_cuda else "cpu")


def detach_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def topk(true, pred, k):
    max_pred = np.argsort(pred, axis=1)[:, -k:]  # take top k
    two_d_true = np.expand_dims(true, 1)  # 1d -> 2d
    two_d_true = np.repeat(two_d_true, k, axis=1)  # repeat along second axis
    return (two_d_true == max_pred).sum()/true.shape[0]