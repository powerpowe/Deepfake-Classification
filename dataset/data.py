import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms

class DataSet(Dataset):
    """
    quality: [high, low]
    dataset_name: [f2f, nt]
    type: [train, val, test]
    """

    def __init__(self, quality, dataset_name, type):
        d_quality = {'high': 'High Quality', 'low': 'Low Quality'}
        d_name = {'f2f': 'f2f_data', 'nt': 'nt_data'}
        self.path = f'./dataset/{d_quality[quality]}/{d_name[dataset_name]}/{type}'

        self.get_num_class()

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                         std = [0.229, 0.224, 0.225])
            ])


    def __len__(self):
        return len(self.data_name_list)

    def get_num_class(self):
        self.data_name_list = os.listdir(self.path)


    def __getitem__(self, idx):
        data_name = self.data_name_list[idx]
        img = read_image(self.path + '/' + data_name)
        img = img / 255  # [0, 1]

        if data_name[:4] == 'fake':
            label = 1
        elif data_name[:4] == 'real':
            label = 0
        return self.transform(img), label