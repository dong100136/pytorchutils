import torch
import time
import os
from tqdm import tqdm
import numpy as np
from torchsummary import summary
from History import History


class Trainer():
    def __init__(self, model_name, model,
                 optimizer_fn=torch.optim.SGD, loss_fn=torch.nn.CrossEntropyLoss(), metric_fn=None,
                 resume=True, lr=0.01,
                 model_save_base_path="./saved_model", use_gpu=True):
        self.config = {}
        self.config['model_name'] = model_name
        self.config['model_save_path'] = os.path.join(
            model_save_base_path, model_name)
        self.config['global_step'] = 0
        self.config['global_epoch'] = 0
        self.config['init_lr'] = lr
        self.config['lr'] = lr
        self.config['use_gpu'] = use_gpu
        self.history = History(self.config['model_save_path'])

        self.model = model
        self.optim = optimizer_fn(model.parameters(), lr=0.01)

        self.optimizer_fn = optimizer_fn
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn

        self.before_batch_fns = []
        self.after_batch_fns = []

        if resume == False:
            self.history.clear()
        else:
            self.resume_stat()

        self.model = model.cuda() if use_gpu else model

    def resume_stat(self):
        self.model, self.optim, self.config = self.history.load_model(
            self.model, self.optim)

    def run(self, train_dataset, valid_dataset, epochs=10):
        begin_epoch = self.config['global_epoch']
        end_epoch = self.config['global_epoch']+epochs
        for epoch in range(begin_epoch, end_epoch):
            print("=====================\nepoch %d/%d\n=====================" %
                  (epoch+1, end_epoch))
            train_loss, train_acc = self.train(train_dataset, 'train')
            valid_loss, valid_acc = self.train(valid_dataset, 'valid')

            self.config['global_epoch'] += 1
            self.check_and_save()

    def check_and_save(self):
        self.history.save_model(self.model, self.optim, self.config)

    def train(self, dataset, mode='train'):
        if mode == 'train':
            self.model.train(True)
        else:
            self.model.eval()

        since = time.time()

        mean_loss = []
        mean_right = []
        mean_count = []

        tbar = tqdm(dataset, total=len(dataset),
                    leave=False)

        for inputs, labels in tbar:
            epoch_since = time.time()

            inputs, labels = self.to_gpu(inputs, labels)

            for func in self.before_batch_fns:
                func(inputs, labels)

            self.optim.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            right = self.metric_fn(outputs, labels)

            mean_loss.append(float(loss))
            mean_right.append(int(right))
            mean_count.append(len(labels))
            acc = float(right)/len(labels)

            if mode == 'train':
                loss.backward()
                self.optim.step()

            for func in self.after_batch_fns:
                func(inputs, outputs, labels)

            tbar.set_description("[Loss] %.4f [Acc] %.4f" % (float(loss), acc))

            self.config['global_step'] += 1

        del inputs, labels, loss, acc

        loss = float(np.mean(mean_loss))
        acc = np.sum(mean_right)/np.sum(mean_count)
        lr = self.get_lr()

        time_elapsed = time.time()-since
        print('%s [Loss]: %.4f [Acc]: %.4f [LR]: %.4f [Time]: %.0f m %.0fs' % (
            mode, loss, acc, lr,
            time_elapsed // 60, time_elapsed % 60))

        self.history.save(mode, self.config['global_step'], loss, acc, lr)
        return loss, acc

    def get_lr(self):
        return self.optim.param_groups[0]['lr']

    def before_batch(self, func):
        self.before_batch_fns.append(func)

    def after_batch(self, func):
        self.after_batch_fns.append(func)

    def loss(self, loss_fn):
        self.loss_fn = loss_fn

    def metric(self, metric_fn):
        self.metric_fn = metric_fn

    def check(self):
        pass

    def to_gpu(self, inputs, labels):
        if self.config['use_gpu']:
            inputs = inputs.cuda()
            labels = labels.cuda()
        return inputs, labels

    def summary(self, input_size):
        summary(self.model, input_size)


if __name__ == '__main__':
    from ResNet import Cifar10_ResNet44
    import torch
    from torch import nn, optim
    import torchvision
    import torchvision.transforms as transforms
    import os
    from torchsummary import summary
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
