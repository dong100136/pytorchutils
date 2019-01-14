import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import skimage
import torch
import torchvision
import torchvision.transforms as transforms
from pytorchutils.models.unet import UNet
from pytorchutils.trainer import Trainer
from torch import nn, optim
import numpy as np
from pytorchutils.models.unet import eval_and_save_rs

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, help='model_name', required=True)

flag_parser = parser.add_mutually_exclusive_group(required=True)
flag_parser.add_argument('--resume', dest='resume', action='store_true')
flag_parser.add_argument('--no-resume', dest='resume', action='store_false')
parser.set_defaults(resume=True)

parser.add_argument("--epochs",type=int,default=0)
parser.add_argument("--gpu", type=str, default='0')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


class MyDataloader(torch.utils.data.Dataset):
    def __init__(self, path):
        self.img_path = os.path.join(path, 'img')
        self.mask_path = os.path.join(path, 'mask')
        self.img_list = [x[:-4] for x in os.listdir(self.img_path)]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.img_path, img_name+'.jpg')
        mask_path = os.path.join(self.mask_path, img_name+"_mask.png")

        img = skimage.io.imread(img_path)
        mask = skimage.io.imread(mask_path)

        img = np.transpose(img,(2,0,1))
        img = img/255.0
        mask = np.sum(mask,axis=-1, keepdims=True)
        mask[mask>0] = 1
        mask = np.transpose(mask,(2,0,1))
        mask = np.int32(mask)
        # mask = mask.reshape(np.shape(mask)[:-1])

        img = torch.FloatTensor(img)
        mask = torch.FloatTensor(mask)
        return img, mask


train_dataloader = MyDataloader("/home/yejiandong/server-96/meike/stone-data/train")
# train_dataloader = MyDataloader("/Users/stone/server/meike/stone-data")
train_dataloader = torch.utils.data.DataLoader(train_dataloader, batch_size=2,
                                               shuffle=True,
                                               pin_memory=True)


valid_dataloader = MyDataloader("/home/yejiandong/server-96/meike/stone-data/valid")
# train_dataloader = MyDataloader("/Users/stone/server/meike/stone-data")
valid_dataloader = torch.utils.data.DataLoader(valid_dataloader, batch_size=2,
                                               shuffle=False,
                                               pin_memory=True)

model = UNet(n_channels=3, n_classes=1)

trainer = Trainer(args.name,model,resume=args.resume,lr=0.001,lr_decay=0.9)

@trainer.loss
def loss(outputs,mask):
    return torch.nn.functional.binary_cross_entropy(outputs,mask)

@trainer.metric
def metric(outputs,mask):
    preds = torch.zeros_like(mask)
    # print(outputs)
    preds[outputs>0.5] = 1
    a,b,c,d= mask.size()
    n = a*b*c*d
    return torch.sum(mask.eq(preds)),n

trainer.run(train_dataloader,valid_dataloader,epochs=args.epochs)

eval_and_save_rs(trainer.model,valid_dataloader,save_path="/home/yejiandong/server-96/meike/stone-data/predict")