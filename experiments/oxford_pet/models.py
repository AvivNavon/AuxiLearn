import torch
import numpy as np
from torch import nn
import torchvision.models as models


class ResNet18(nn.Module):
    def __init__(self, psi, pretrained=False, progress=True, num_classes=37, **kwargs):
        super().__init__()
        self.model = models.resnet18(pretrained=pretrained, progress=progress, **kwargs)

        # main task classifier
        self.classifier1 = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

        # auxiliary task classifier
        self.classifier2 = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, int(np.sum(psi))),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.model(x)
        p1, p2 = self.classifier1(x), self.classifier2(x)
        return p1, p2


class AuxiliaryNet(nn.Module):
    def __init__(self, psi, pretrained=False, progress=True, **kwargs):
        super().__init__()
        self.model = models.resnet18(pretrained=pretrained, progress=progress, **kwargs)

        self.class_nb = psi
        # generate label head
        self.classifier1 = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, int(np.sum(self.class_nb))),
        )

    def mask_softmax(self, x, mask, dim=1):
        z = x.max(dim=dim)[0]
        x = x - z.reshape(-1, 1)
        logits = torch.exp(x) * mask / (torch.sum(torch.exp(x) * mask, dim=dim, keepdim=True) + 1e-7)
        return logits

    def forward(self, x, y):
        device = x.device
        x = self.model(x)

        # build a binary mask by psi, we add epsilon=1e-8 to avoid nans
        index = torch.zeros([len(self.class_nb), np.sum(self.class_nb)]) + 1e-8
        for i in range(len(self.class_nb)):
            index[i, int(np.sum(self.class_nb[:i])):np.sum(self.class_nb[:i + 1])] = 1
        mask = index[y].to(device)

        predict = self.classifier1(x.view(x.size(0), -1))
        label_pred = self.mask_softmax(predict, mask, dim=1)
        return label_pred
