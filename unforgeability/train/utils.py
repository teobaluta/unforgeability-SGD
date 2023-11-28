from enum import Enum
import torch
import torchvision
import numpy as np
import os
from train.globals import MyGlobals
import train.models as models
import torch.nn as nn
from typing import Tuple


class Architecture(Enum):
    LeNet5 = 'lenet5'
    ResNet18 = 'resnet18'
    VGG11 = 'vgg11'
    FCN3 = 'fcn3'
    ResNet18_BN = 'resnet18_bn'
    ResNet_mini = 'resnet_mini'
    VGG_mini = 'vgg_mini'

    def __str__(self):
        return self.value


class Dataset(Enum):
    MNIST = 'mnist'
    CIFAR10 = 'cifar10'
    CIFAR100 = 'cifar100'

    def __str__(self):
        return self.value


def get_num_classes(dataset: Dataset) -> int:
    if dataset == Dataset.CIFAR100:
        return 100
    return 10


def factory_model(arch: Architecture, input_channels: Tuple[int, int], input_features,
                  num_classes: int) -> nn.Module:
    if arch == Architecture.LeNet5:
        model = models.LeNet5(num_classes, input_channels)
    elif arch == Architecture.ResNet18:
        model = models.ResNet18(num_classes)
    elif arch == Architecture.ResNet18_BN:
        model = torchvision.models.resnet18()
        model.fc.out_features = num_classes
    elif arch == Architecture.VGG11:
        model = torchvision.models.vgg11()
    elif arch == Architecture.FCN3:
        model = models.FCN3(num_classes, input_features)
    elif arch == Architecture.ResNet_mini:
        model = models.Resnet(input_channels)
    elif arch == Architecture.VGG_mini:
        model = models.VGG_mini()

    return model


def compute_grad(model, loss_fn, sample, target):
    sample = sample.unsqueeze(0)  # prepend batch dimension for processing
    target = target.unsqueeze(0)

    prediction = model(sample)
    loss = loss_fn(prediction, target)

    return torch.autograd.grad(loss, list(model.parameters()))


def compute_sample_grads(model, loss_fn, data, targets, batch_size):
    """Process each sample with per sample gradient."""
    sample_grads = [compute_grad(model, loss_fn, data[i], targets[i])
                    for i in range(batch_size)]
    sample_grads = zip(*sample_grads)
    sample_grads = [torch.stack(shards) for shards in sample_grads]
    return sample_grads


def save_grads(per_sample_grads, arch, dataset, epoch, iteration):
    for layer in range(len(per_sample_grads)):
        per_sample_grads[layer] = per_sample_grads[layer].to(
            torch.device('cpu')).numpy()
    path = os.path.join(MyGlobals.RESULTDIR, f'{dataset}/{arch}-train_grads/')
    os.makedirs(path, exist_ok=True)
    result_file = os.path.join(
        path, f'{dataset}_{arch}-train_grads-epoch-{epoch}-ts_{iteration}.pt')
    torch.save(per_sample_grads, result_file)


def save_params(model_state, optimizer_state, loss, arch, dataset, epoch, iteration):
    path = os.path.join(MyGlobals.RESULTDIR, f'{dataset}/{arch}-batch_size_64/')
    os.makedirs(path, exist_ok=True)
    result_file = os.path.join(path, f'{dataset}_{arch}-ckpt-epoch_{epoch}-ts_{iteration}.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer_state,
        'loss': loss,
    }, result_file)


def flatten_gradient_ids(model):
    state_dict = model.state_dict()
    gradient_ids = []
    total_params = 0
    for layer in state_dict:
        t = torch.tensor(model.state_dict()[layer].shape)
        params = torch.prod(t)
        total_params += int(params.cpu())
        gradient_ids.append(int(params.cpu()))

    return total_params, gradient_ids
