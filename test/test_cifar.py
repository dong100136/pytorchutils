
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim

from pytorchutils.models.ResNet import Cifar10_ResNet44
from pytorchutils.trainer import Trainer



train_transform = transforms.Compose(
    [transforms.RandomSizedCrop(32),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='~/.torch/datasets', train=True,
                                        download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=10, pin_memory=True)

testset = torchvision.datasets.CIFAR10(root='~/.torch/datasets', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=10, pin_memory=True)

model = Cifar10_ResNet44()

trainer = Trainer("base_v2", model,
                  resume=True)

critern = nn.CrossEntropyLoss(reduce=True)


@trainer.loss
def loss_fn(outputs, labels):
    return critern(outputs, labels)


@trainer.metric
def metric_fn(outputs, labels):
    preds = torch.argmax(outputs, dim=-1)
    right = labels.eq(preds).sum()
    return right


trainer.run(trainloader, testloader, epochs=100)
