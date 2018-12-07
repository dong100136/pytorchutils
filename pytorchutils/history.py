from pytorchutils.utils import check_and_create_dir
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import json
sns.set()


class History():
    def __init__(self, model_save_path):
        self.model_save_path = model_save_path

    def save(self,  mode, step, loss, acc, lr):

        path = os.path.join(self.model_save_path, "history.csv")

        with open(path, 'a+') as f:
            f.write("%s,%d,%.5f,%.5f,%.8f\n" %
                    (mode, step, loss, acc, lr))

    def save_model(self, model, optim, config):
        print("save model into %s" % (self.model_save_path))
        model_path = os.path.join(self.model_save_path, 'model.pkl')
        optim_path = os.path.join(self.model_save_path, 'optim.pkl')
        torch.save(model.state_dict(), model_path)
        torch.save(optim.state_dict(), optim_path)

        config_path = os.path.join(self.model_save_path, 'config.json')
        with open(config_path, 'w') as f:
            f.write(json.dumps(config))

    def load_model(self, model, optim,config):
        print("load model from %s" % (self.model_save_path))
        model_path = os.path.join(self.model_save_path, 'model.pkl')
        optim_path = os.path.join(self.model_save_path, 'optim.pkl')
        config_path = os.path.join(self.model_save_path, 'config.json')

        if os.path.exists(model_path) and os.path.exists(optim_path) and os.path.exists(config_path):
            model.load_state_dict(torch.load(model_path))
            optim.load_state_dict(torch.load(optim_path))
           
            with open(config_path, 'r') as f:
                lines = f.readlines()
                config = json.loads(''.join(lines))
        else:
            print("can't find checkpoint in %s"%self.model_save_path)
            check_and_create_dir(self.model_save_path)
        return model, optim, config

    def clear(self):
        print("remove all history in %s" % self.model_save_path)

        if os.path.exists(self.model_save_path):
            confirm = input("%s is exists, comfirm to delete it?[y/n]"%self.model_save_path)
            while (confirm!='y' and confirm!='n'):
                confirm= input('please input [y/n]')
            if confirm=='y':
                os.system("rm -rf %s" % self.model_save_path)
                check_and_create_dir(self.model_save_path)
            else:
                exit()


def plot(*args, **kwargs):
    if kwargs and 'show' in kwargs:
        show_curve = kwargs['show']
    else:
        show_curve = ['train', 'valid']

    data = {}
    for i, base_path in enumerate(args):
        p = os.path.join(base_path, 'history.csv')
        with open(p, 'r') as f:
            for line in f:
                mode, step, loss, acc, lr = line.strip().split(',')
                if not mode in data:
                    data[mode] = []
                if len(data[mode]) <= i:
                    data[mode].append({
                        'name': os.path.basename(base_path),
                        'step': [],
                        'loss': [],
                        'acc': [],
                        'lr': []})
                data[mode][i]['step'].append(int(step))
                data[mode][i]['loss'].append(float(loss))
                data[mode][i]['acc'].append(float(acc))
                data[mode][i]['lr'].append(float(lr))

    plt_args1 = []
    plt_names = []
    plt_args2 = []
    for mode in show_curve:
        for d in data[mode]:
            plt_args1.extend([d['step'], d['loss'], '-'])
            plt_args2.extend([d['step'], d['acc'], '-'])
            plt_names.append("%s-%s" % (d['name'], mode))

    print(*plt_names)

    plt.figure(figsize=(15, 6), dpi=100)

    plt.subplot(1, 2, 1)
    plt.plot(*plt_args1)
    plt.legend(plt_names)
    plt.title("loss")

    plt.subplot(1, 2, 2)
    plt.plot(*plt_args2)
    plt.legend(plt_names)
    plt.title("acc")
