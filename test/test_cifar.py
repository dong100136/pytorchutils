import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from pytorchutils.models.ResNet import Cifar10_ResNet44
from pytorchutils.trainer import Trainer



parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, help='model_name', required=True)

flag_parser = parser.add_mutually_exclusive_group(required=True)
flag_parser.add_argument('--resume', dest='resume', action='store_true')
flag_parser.add_argument('--no-resume', dest='resume', action='store_false')
parser.set_defaults(resume=True)

parser.add_argument("--gpu", type=str, default='0')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

train_transform = transforms.Compose(
    [
     transforms.RandomSizedCrop(32),
    #  transforms.RandomHorizontalFlip(),
    #  transforms.RandomVerticalFlip(),
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
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-7)
trainer = Trainer(args.name, model,lr=0.01, lr_decay=0.95, optimizer=optimizer,
                  resume=args.resume)

critern = nn.CrossEntropyLoss(reduce=True)


@trainer.loss
def loss_fn(outputs, labels):
    return critern(outputs, labels)


@trainer.metric
def metric_fn(outputs, labels):
    preds = torch.argmax(outputs, dim=-1)
    right = labels.eq(preds).sum()
    return right, len(preds)


@trainer.lr
def update_lr(config):
    if config['global_epoch'] < 10:
        return 0.01
    elif config['global_epoch'] < 20:
        return 0.001
    else:
        return 0.0001


trainer.run(trainloader, testloader, epochs=100)
