import torch
import torch.nn as nn
import numpy as np
from train.data import LoadData
import train.utils as utils
from train.globals import MyGlobals
import os
import argparse
import configparser

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.backends.cudnn.deterministic = True
torch.set_default_dtype(torch.float64)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
g = torch.Generator()
g.manual_seed(0)

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--device', type=str, help='Select the gpu to run.')
parser.add_argument(
    '-a', '--arch', type=utils.Architecture, help='The model architecture.'
)
parser.add_argument(
    '-ds', '--dataset', type=utils.Dataset, help='The dataset to train on.'
)
parser.add_argument(
    '--batch_size', type=int, help='batch_size to train with'
)
parser.add_argument(
    '--num_epochs', type=int, help='number of epochs to train for'
)
parser.add_argument(
    '--lr', type=float, help='learning rate to train with'
)
parser.add_argument(
    '--save_ckpts', type=int, help='set 1 to save model checkpoints, 0 otherwise'
)
# TODO num classes should be evaluated by the LoadData class itself
parser.add_argument(
    '-n',
    '--num_classes',
    type=int,
    help='The input number of classes that your dataset has',
)

args = parser.parse_args()

batch_size = args.batch_size
num_classes = args.num_classes
lr = args.lr
num_epochs = args.num_epochs
save_ckpts = args.save_ckpts

device = args.device

data_loader = LoadData(args.dataset, True, batch_size, g)

model = utils.factory_model(
    args.arch, data_loader.input_channels, data_loader.input_features, num_classes
).to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=lr)

total_step = len(data_loader.train_loader)

if save_ckpts:
    utils.save_params(
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        loss=0,
        arch=args.arch,
        dataset=args.dataset.value,
        epoch='init',
        iteration='init',
    )

batch_ind = np.empty(shape=(num_epochs, len(data_loader.train_loader), batch_size))


for epoch in range(num_epochs):
    for i, (images, labels, indices) in enumerate(data_loader.train_loader):
        batch_ind[epoch][i] = indices.numpy()
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if save_ckpts:
            utils.save_params(
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                loss=loss,
                arch=args.arch,
                dataset=args.dataset.value,
                epoch=epoch,
                iteration=i,
            )

        if (i + 1) % 40 == 0:
            print(
                'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, num_epochs, i + 1, total_step, loss.item()
                )
            )

path = os.path.join(MyGlobals.RESULTDIR,
                    f'{args.dataset.value}/{args.arch}-batch_indices-batch_size_64/')
os.makedirs(path, exist_ok=True)
np.save(os.path.join(path,
                     f'{args.dataset.value}-{args.arch}-num_epochs_{num_epochs}-batch_indices.npy'),
        batch_ind)

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels, _ in data_loader.train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Train accuracy: {} %'.format(100 * correct / total))

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels, _ in data_loader.test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test accuracy: {} %'.format(100 * correct / total))
