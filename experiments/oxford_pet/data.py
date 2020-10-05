import pandas as pd
import numpy as np
import os
import shutil
from pathlib import Path
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
from torchvision import models, transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.random import sample_without_replacement


def split_sets_to_folders(root='./dataset'):
    """
    Input:
        root - root directory
        orig_path - original file path
        dest_path - destination file path
    Output:
        Write to file a new data split: 0 - train, 1 - validation, 2 - test
    """
    def create_folder(split='train'):
        dest_folder_path = os.path.join(root, 'images_processed', split)
        # make new data path dir
        if not os.path.exists(dest_folder_path):
            os.makedirs(dest_folder_path)

    val_pct = 0.3  # between 30-50 images per class with an average of 42

    create_folder(split='train')
    create_folder(split='validation')
    create_folder(split='test')

    # get rel files
    meta_data_trn = os.path.join(root, 'annotations', 'trainval.txt')
    meta_data_trn = pd.read_csv(meta_data_trn, header=None, sep=' ')
    meta_data_tst = os.path.join(root, 'annotations', 'test.txt')
    meta_data_tst = pd.read_csv(meta_data_tst, header=None, sep=' ')

    X_train, X_val, y_train, y_val = train_test_split(meta_data_trn.iloc[:, 0], meta_data_trn.iloc[:, 1],
                                                      test_size=val_pct, random_state=42,
                                                      stratify=meta_data_trn.iloc[:, 1])
    X_test, y_test = meta_data_tst.iloc[:, 0], meta_data_tst.iloc[:, 1]

    X_train = X_train.values
    X_val = X_val.values
    X_test = X_test.values
    y_train = y_train.values
    y_val = y_val.values
    y_test = y_test.values

    src_images_path = os.path.join(root, 'images')
    dest_folder_path = os.path.join(root, 'images_processed')
    for idx, image in enumerate(os.listdir(src_images_path)):
        if image.split('.')[1] != 'jpg':
            continue
        image_name = image.split('.')[0]
        if image_name in X_train:
            img_idx = np.where(X_train == image_name)[0].item()
            lbl = str(y_train[img_idx] - 1)
            split = 'train'
        elif image_name in X_val:
            img_idx = np.where(X_val == image_name)[0].item()
            lbl = str(y_val[img_idx] - 1)
            split = 'validation'
        elif image_name in X_test:
            img_idx = np.where(X_test == image_name)[0].item()
            lbl = str(y_test[img_idx] - 1)
            split = 'test'
        else:
            print(image_name + " wasn't found")
            continue

        class_folder_path = os.path.join(dest_folder_path, split, lbl)
        # make new data path dir
        if not os.path.exists(class_folder_path):
            os.makedirs(class_folder_path)

        # move file to new path
        old_class_path = os.path.join(src_images_path, image)
        shutil.move(old_class_path, class_folder_path)


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


class Oxford_Pet:

    def __init__(self, root='./dataset'):
        self.data_dir = Path(root)

    def get_datasets(self, train_shot, aux_set_size=0.):
        # train_shot: number of samples per class
        #

        assert type(train_shot) == int

        traindir = self.data_dir / 'images_processed' / 'train'
        validatedir = self.data_dir / 'images_processed' / 'validation'
        testdir = self.data_dir / 'images_processed' / 'test'

        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),  # Crop to 224x224 a random patch
                transforms.RandomHorizontalFlip(),  # data augmentation
                transforms.ToTensor(),  # to tensor and scale to 0-1
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # imagenet statistics
            ]),
            'test': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),  # at test time we want deterministic cropping
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # imagenet statistics
            ]),
        }

        trainset = ImageFolder(root=traindir, transform=data_transforms['train'])
        path_to_labels = trainset.samples  # list of tuples: (path, label_id)
        labels = [l for p, l in path_to_labels]
        num_classes = np.unique(labels).shape[0]

        if train_shot * num_classes < len(trainset):
            train_indices, _ = train_test_split(range(len(trainset)), train_size=train_shot * num_classes,
                                                random_state=42, stratify=labels)
        else:
            train_indices = list(range(len(trainset)))

        trainset = Subset(trainset, train_indices)  # take only relevant samples
        validateset = ImageFolder(root=validatedir, transform=data_transforms['test'])
        testset = ImageFolder(root=testdir, transform=data_transforms['test'])

        # list of samples labels for stratify in case meta val is larger than #classes
        labels_list = [trainset.dataset[trainset.indices[l]][1] for l in range(len(trainset))]
        test_size = aux_set_size if aux_set_size * len(trainset) >= num_classes else num_classes

        train_indices, aux_indices = train_test_split(
            range(len(trainset)), test_size=test_size, random_state=42,
            stratify=labels_list
        )

        # if less than number of classes sample only 1 per class
        if test_size > aux_set_size * len(trainset):
            sampled_meta_val_indices = sample_without_replacement(test_size,
                                                                  aux_set_size * len(trainset),
                                                                  random_state=42)
            aux_indices_tmp = [aux_indices[i] for i in sampled_meta_val_indices]
            train_additonal_indices = set(aux_indices) - set(aux_indices_tmp)
            train_indices += list(train_additonal_indices)  # add back to train
            aux_indices = aux_indices_tmp

        metavalset = Subset(trainset, aux_indices)
        trainset = Subset(trainset, train_indices)

        return trainset, validateset, testset, metavalset


if __name__ == '__main__':
    split_sets_to_folders(root='./dataset')