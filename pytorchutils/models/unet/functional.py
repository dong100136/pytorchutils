import torch
import os
import skimage
import matplotlib.pyplot as plt


def eval_and_save_rs(model,dataloader,save_path,threshold=0.5,use_gpu=True):
    i = -1
    os.system("rm %s/*"%(save_path))
    for inputs,labels in dataloader:
        if use_gpu:
            model = model.cuda()
            inputs = inputs.cuda()
        else:
            model = model.cpu()

        model.eval()

        output = model(inputs).detach().cpu()
        preds = torch.zeros_like(output)
        preds[output>threshold] = 1
        preds = preds.permute((0,2,3,1))
        preds = preds.reshape(preds.size()[:-1])
        labels = labels.permute((0,2,3,1))
        labels = labels.reshape(labels.size()[:-1])

        n = len(output)

        for j  in range(n):
            i+=1
            img_name = "predidct_%d.png"%i
            img_path = os.path.join(save_path,img_name)

            img = torch.cat([preds[j],labels[j]],dim=1)

            skimage.io.imsave(img_path,img)

