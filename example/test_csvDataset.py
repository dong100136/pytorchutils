
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
##


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from pytorchutils.data import CsvDataSet

    dataset = CsvDataSet(csv="./list.txt", prefix="./data", delimeter=' ')
    dataset = DataLoader(dataset, batch_size=2)

    imgs, labels = next(iter(dataset))

    print(imgs[0].shape)
    print(labels)
