import torch
import torchvision
import torchvision.transforms as transforms

import os

from train.utils import Dataset
from train.globals import MyGlobals

# torch.backends.cudnn.deterministic = True
# torch.set_default_dtype(torch.float64)
# torch.manual_seed(42)


def dataset_with_indices(cls):
    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })


class LoadData:
    def __init__(
        self,
        dataset: Dataset,
        randomize=True,
        batch_size=64,
        generator=torch.Generator(),
        num_workers=4,
        drop_last=True
    ):
        self.dataset = dataset
        self.randomize = randomize
        self.load_dir = os.path.join(MyGlobals.DATADIR, dataset.value)
        self.input_features = None
        self.input_channels = None
        self.train_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.test_loader = None
        self.batch_size = batch_size
        self.generator = generator
        self.num_workers = num_workers
        self.drop_last = drop_last
        self._load_data()

    def _load_data(self):
        print(
            'Load dir {}; dataset name {}'.format(
                self.load_dir, self.dataset.name)
        )
        if self.dataset == Dataset.MNIST:
            MNIST = dataset_with_indices(torchvision.datasets.MNIST)
            self.train_dataset = MNIST(root=self.load_dir,
                                       train=True,
                                       transform=transforms.Compose([
                                           transforms.Resize((32, 32)),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=(0.1307,), std=(0.3081,))]),
                                       download=True)

            self.test_dataset = MNIST(root=self.load_dir,
                                      train=False,
                                      transform=transforms.Compose([
                                          transforms.Resize((32, 32)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.1325,), std=(0.3105,))]),
                                      download=True)

            self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=self.randomize,
                                                            drop_last=self.drop_last,
                                                            generator=self.generator,
                                                            num_workers=self.num_workers,
                                                            pin_memory=True)

            self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                           batch_size=self.batch_size,
                                                           shuffle=self.randomize,
                                                           drop_last=self.drop_last,
                                                           generator=self.generator,
                                                           num_workers=self.num_workers,
                                                           pin_memory=True)

            self.input_channels = self.train_dataset[0][0].shape[0]
            self.input_features = torch.prod(
                torch.tensor(self.train_dataset[0][0].shape))

        elif self.dataset == Dataset.CIFAR10:
            CIFAR10 = dataset_with_indices(torchvision.datasets.CIFAR10)
            self.train_dataset = CIFAR10(root=self.load_dir,
                                         train=True,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                                         download=True)

            self.test_dataset = CIFAR10(root=self.load_dir,
                                        train=False,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                                        download=True)

            self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=self.randomize,
                                                            drop_last=self.drop_last,
                                                            generator=self.generator,
                                                            num_workers=self.num_workers,
                                                            pin_memory=True)

            self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                           batch_size=self.batch_size,
                                                           shuffle=self.randomize,
                                                           drop_last=self.drop_last,
                                                           generator=self.generator,
                                                           num_workers=self.num_workers,
                                                           pin_memory=True)
            self.input_channels = self.train_dataset[0][0].shape[0]
            self.input_features = torch.prod(
                torch.tensor(self.train_dataset[0][0].shape))

        elif self.dataset == Dataset.CIFAR100:
            CIFAR100 = dataset_with_indices(torchvision.datasets.CIFAR100)
            self.train_dataset = CIFAR100(
                root=self.load_dir,
                train=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        MyGlobals.CIFAR100_TRAIN_MEAN, MyGlobals.CIFAR100_TRAIN_STD)
                ]),
                download=True
            )
            self.test_dataset = CIFAR100(
                root=self.load_dir,
                train=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        MyGlobals.CIFAR100_TEST_MEAN, MyGlobals.CIFAR100_TEST_STD)
                ]),
                download=True
            )
            self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=self.randomize,
                                                            drop_last=self.drop_last,
                                                            generator=self.generator,
                                                            num_workers=self.num_workers,
                                                            pin_memory=True)

            self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                           batch_size=self.batch_size,
                                                           shuffle=self.randomize,
                                                           drop_last=self.drop_last,
                                                           generator=self.generator,
                                                           num_worker=self.num_workers,
                                                           pin_memory=True)

            self.input_channels = self.train_dataset[0][0].shape[0]
            self.input_features = torch.prod(
                torch.tensor(self.train_dataset[0][0].shape))
