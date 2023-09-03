import torch
import os
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
from torchvision.models import vit_b_16, ViT_B_16_Weights
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from scipy.special import softmax
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix
# import xDNN_optimization as xdnn
import time
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import argparse
import torchvision
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import copy
from torchvision import models



import os.path
from typing import Any, Callable, List, Optional, Tuple, Union

from PIL import Image

from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset


class Caltech101(VisionDataset):
    """`Caltech 101 <https://data.caltech.edu/records/20086>`_ Dataset.

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Args:
        root (string): Root directory of dataset where directory
            ``caltech101`` exists or will be saved to if download is set to True.
        target_type (string or list, optional): Type of target to use, ``category`` or
            ``annotation``. Can also be a list to output a tuple with all specified
            target types.  ``category`` represents the target class, and
            ``annotation`` is a list of points from a hand-generated outline.
            Defaults to ``category``.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(
        self,
        root: str,
        target_type: Union[List[str], str] = "category",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(os.path.join(root, "caltech101"), transform=transform, target_transform=target_transform)
        os.makedirs(self.root, exist_ok=True)
        if isinstance(target_type, str):
            target_type = [target_type]
        self.target_type = [verify_str_arg(t, "target_type", ("category", "annotation")) for t in target_type]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        self.categories = sorted(os.listdir(os.path.join(self.root, "101_ObjectCategories")))
        self.categories.remove("BACKGROUND_Google")  # this is not a real class

        # For some reason, the category names in "101_ObjectCategories" and
        # "Annotations" do not always match. This is a manual map between the
        # two. Defaults to using same name, since most names are fine.
        name_map = {
            "Faces": "Faces_2",
            "Faces_easy": "Faces_3",
            "Motorbikes": "Motorbikes_16",
            "airplanes": "Airplanes_Side_2",
        }
        self.annotation_categories = list(map(lambda x: name_map[x] if x in name_map else x, self.categories))

        self.index: List[int] = []
        self.y = []
        for (i, c) in enumerate(self.categories):
            n = len(os.listdir(os.path.join(self.root, "101_ObjectCategories", c)))
            self.index.extend(range(1, n + 1))
            self.y.extend(n * [i])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where the type of target specified by target_type.
        """
        import scipy.io

        img = Image.open(
            os.path.join(
                self.root,
                "101_ObjectCategories",
                self.categories[self.y[index]],
                f"image_{self.index[index]:04d}.jpg",
            )
        ).convert("RGB")

        target: Any = []
        for t in self.target_type:
            if t == "category":
                target.append(self.y[index])
            elif t == "annotation":
                data = scipy.io.loadmat(
                    os.path.join(
                        self.root,
                        "Annotations",
                        self.annotation_categories[self.y[index]],
                        f"annotation_{self.index[index]:04d}.mat",
                    )
                )
                target.append(data["obj_contour"])
        target = tuple(target) if len(target) > 1 else target[0]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def _check_integrity(self) -> bool:
        # can be more robust and check hash of files
        return os.path.exists(os.path.join(self.root, "101_ObjectCategories"))

    def __len__(self) -> int:
        return len(self.index)

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_and_extract_archive(
            "https://drive.google.com/file/d/137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp",
            self.root,
            filename="101_ObjectCategories.tar.gz",
            md5="b224c7392d521a49829488ab0f1120d9",
        )
        download_and_extract_archive(
            "https://drive.google.com/file/d/175kQy3UsZ0wUEHZjqkUDdNVssr7bgh_m",
            self.root,
            filename="Annotations.tar",
            md5="6f83eeb1f24d99cab4eb377263132c91",
        )

    def extra_repr(self) -> str:
        return "Target type: {target_type}".format(**self.__dict__)



def train_model(model, dataloaders, criterion, optimizer, num_epochs=100):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    print('Saving..')
    state = {
        'model': best_model_wts,
        'acc': best_acc,
        'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.pth')
    
    return model, val_acc_history
    
# test the model on test set
def test_model(model, dataloaders, criterion):
    
    since = time.time()
    
    running_accuracy = 0.0
    
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            total += labels.size(0)
            running_accuracy += torch.sum(preds == labels.data)
    
    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            
    test_accuracy = running_accuracy.double() / total * 100
    
    print('Test Accuracy: {:.4f}'.format(test_accuracy))
    
    return test_accuracy
    

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, weights, use_pretrained=True):
    
    model_ft = None
    input_size = 0
    
    if model_name == "vit_b_16":
        model_ft = vit_b_16(weights=weights)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.heads.head.in_features
        model_ft.heads.head = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == "resnet50":
        model_ft = models.resnet50(weights=weights)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == "resnet101":
        model_ft = models.resnet101(weights=weights)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == "vgg16":
        """ VGG16_bn
        """
        model_ft = models.vgg16(weights=weights)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        
    else:
        print("Invalid model name, exiting...")
        exit()
        
    return model_ft, input_size
    


if __name__ == '__main__':
  
    

    
    
    model_name = "resnet101"
    
    weights = models.ResNet101_Weights.DEFAULT

    num_classes = 10
    
    # batch size for training
    
    batch_size = 256
    
    # number of epochs to train for
    
    num_epochs = 200
    
    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    
    feature_extract  = True
    
    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, weights, use_pretrained=True)

    # Print the model we just instantiated
    print(model_ft)
    
    

    print("Initializing Datasets and Dataloaders...")
    
   
    
    image_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=weights.transforms())
    
    trainset, valset = torch.utils.data.random_split(image_dataset, [40000, 10000])
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=weights.transforms())
    
    
    
    dataloaders_dict = {'train': torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4),
                       'val': torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=4),
                       'test': torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)}
    
    
    # Detect if we have a GPU available
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    # Send the model to GPU
    
    model_ft = model_ft.to(device)
    
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)
        
    pytorch_total_params = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
    print("Total number of trainable parameters: ", pytorch_total_params)
   # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    
    
    # Setup the loss fxn
    
    criterion = nn.CrossEntropyLoss()
    
    # Train and evaluate
    
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)
    
    
    # Test the model on test set
    
    test_model(model_ft, dataloaders_dict, criterion)
    
    
    
    


        
        


