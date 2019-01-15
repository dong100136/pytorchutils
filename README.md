# Pytorchutils

a library for using pytorch easier

## features

- trainer, easier to train and resume

- default models

- history log

- plot and compare hitstory easily

- csv dataloader

## quick start

```sh
$ pip3 uninstall pytorchutils
$ pip3 install --user  git+https://github.com/dong100136/pytorchutils.git
```

## simple usage

```python
from pytorchutils.trainer import Trainer
from pytorchutils.models import Cifar10_ResNet44

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

trainer = Trainer("base_v2", model,resume=True)

trainer.run(trainloader, testloader, epochs=100)
```
