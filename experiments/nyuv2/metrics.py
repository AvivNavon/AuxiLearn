import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, multilabel_confusion_matrix)
from sklearn.preprocessing import MultiLabelBinarizer

"""Source: https://github.com/lorenmt/mtan"""


def main_task_accuracy(pred, target, main_task):
    target = (target == main_task) * 1
    pred = pred.argmax(1)
    pred = (pred == main_task) * 1
    acc = accuracy_score(target, pred)

    return acc


def clf_report(pred, target, main_task=None, stl=False):
    if stl:
        assert main_task is not None, "please supply main task position"
        target = (target == main_task) * 1
        pred = pred.argmax(1)  # there are 2 outputs and we use cross-entropy,
        # so this is the same as taking the one with prob > .5
        return classification_report(target, pred, output_dict=True)

    else:  # mtl
        target = [[t] for t in target]
        one_hot_target = MultiLabelBinarizer().fit_transform(target)
        preds = (pred > 0) * 1  # this is equivalent to prob > .5
        return classification_report(one_hot_target, preds, output_dict=True)


def compute_miou(x_pred, x_output, class_nb=13):
    _, x_pred_label = torch.max(x_pred, dim=1)
    x_output_label = x_output
    batch_size = x_pred.size(0)
    for i in range(batch_size):
        true_class = 0
        first_switch = True
        for j in range(class_nb):
            pred_mask = torch.eq(
                x_pred_label[i], j * torch.ones(x_pred_label[i].shape).type(torch.LongTensor).to(x_pred.device)
            )
            true_mask = torch.eq(
                x_output_label[i], j * torch.ones(x_output_label[i].shape).type(torch.LongTensor).to(x_pred.device)
            )
            mask_comb = pred_mask.type(torch.FloatTensor) + true_mask.type(torch.FloatTensor)
            union = torch.sum((mask_comb > 0).type(torch.FloatTensor))
            intsec = torch.sum((mask_comb > 1).type(torch.FloatTensor))
            if union == 0:
                continue
            if first_switch:
                class_prob = intsec / union
                first_switch = False
            else:
                class_prob = intsec / union + class_prob
            true_class += 1
        if i == 0:
            batch_avg = class_prob / true_class
        else:
            batch_avg = class_prob / true_class + batch_avg
    return batch_avg / batch_size


def compute_iou(x_pred, x_output):
    """Pixel accuracy

    :param x_pred:
    :param x_output:
    :return:
    """
    _, x_pred_label = torch.max(x_pred, dim=1)
    x_output_label = x_output
    batch_size = x_pred.size(0)
    for i in range(batch_size):
        if i == 0:
            pixel_acc = torch.div(
                torch.sum(torch.eq(x_pred_label[i], x_output_label[i]).type(torch.FloatTensor)),
                torch.sum((x_output_label[i] >= 0).type(torch.FloatTensor))
            )
        else:
            pixel_acc = pixel_acc + torch.div(
                torch.sum(
                    torch.eq(x_pred_label[i], x_output_label[i]).type(torch.FloatTensor)
                ),
                torch.sum((x_output_label[i] >= 0).type(torch.FloatTensor))
            )
    return pixel_acc / batch_size


def depth_error(x_pred, x_output):
    binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).to(x_pred.device)
    x_pred_true = x_pred.masked_select(binary_mask)
    x_output_true = x_output.masked_select(binary_mask)
    abs_err = torch.abs(x_pred_true - x_output_true)
    rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
    return torch.sum(abs_err) / torch.nonzero(binary_mask).size(0), \
        torch.sum(rel_err) / torch.nonzero(binary_mask).size(0)


def normal_error(x_pred, x_output):
    binary_mask = (torch.sum(x_output, dim=1) != 0)
    error = torch.acos(
        torch.clamp(torch.sum(x_pred * x_output, 1).masked_select(binary_mask), -1, 1)
    ).detach().cpu().numpy()
    error = np.degrees(error)
    return np.mean(error), np.median(error), np.mean(error < 11.25), np.mean(error < 22.5), np.mean(error < 30)


def confusion_mat(pred, target, main_task=None, stl=False):
    if stl:
        assert main_task is not None, "please supply main task position"
        target = (target == main_task) * 1
        pred = pred.argmax(1)  # there are 2 outputs and we use cross-entropy,
        # so this is the same as taking the one with prob > .5
        return confusion_matrix(target, pred)

    else:  # mtl
        target = [[t] for t in target]
        one_hot_target = MultiLabelBinarizer().fit_transform(target)
        preds = (pred > 0) * 1  # this is equivalent to prob > .5
        return multilabel_confusion_matrix(one_hot_target, preds)
