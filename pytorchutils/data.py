from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, ToTensor
from skimage.io import imread, imsave
import random
import os


class CsvDataSet(Dataset):
    def __init__(self, csv, prefix="", suffix="", sample=None, mode='train', delimeter=',', transformer=None, seed=12345):
        super(CsvDataSet, self).__init__()
        self.csv_path = csv
        self.prefix = prefix
        self.delimeter = delimeter
        self.transformer = transformer
        self.toPIL = ToPILImage()
        self.toTensor = ToTensor()
        self.mode = mode
        self.sample = sample
        self.seed = seed
        self.suffix = suffix
        random.setstate(seed)

        self.__parse_csv__()

    def __parse_csv__(self):
        self.data = []
        self.clazz_num = {}
        with open(self.csv_path, 'r') as f:
            if self.mode == 'eval':
                self.data = [line.strip() for line in f]
            else:
                for line in f:
                    img_path, label = line.strip().split(self.delimeter)
                    self.data.append(
                        [img_path, int(label)])

                    if label not in self.clazz_num:
                        self.clazz_num[label] = 0
                    self.clazz_num[label] += 1

        random.shuffle(self.data)
        print("found %d images" % (len(self.data)))

        if self.mode != 'eval':
            for clazz in self.clazz_num:
                print("%s  %d" % (clazz, self.clazz_num[clazz]))

        if self.sample:
            print("sample %d data from all" % (len(self.data) * self.sample))
            self.data = self.data[:int(len(self.data)*self.sample)]

        self.size = len(self.data)

    def __getitem__(self, index):
        if self.mode == 'eval':
            img_name = self.data[index]
        else:
            img_name, label = self.data[index]

        img_path = os.path.join(self.prefix, img_name, self.suffix)
        img = imread(img_path)

        img = self.toPIL(img)

        if self.transformer:
            img = self.transformer(img)
        else:
            img = self.toTensor(img)

        if self.mode == 'eval':
            return img_name, img
        else:
            return img, label

    def __len__(self):
        return self.size
