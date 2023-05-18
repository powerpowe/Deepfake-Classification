import torch.cuda

from module import utils, training_testing
from dataset import data

setting = {"model_name": "xception",  # xception, efficientnet_b0
           "quality": 'low',  # high, low
           'dataset_name': 'nt'}  # f2f, nt

hyperparameter = {'batch_size': 32,
                  'lr': 1e-4,
                  'epoch': 100}

if __name__ == "__main__":
    model = utils.download_model(setting["model_name"], 2, True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dset = data.DataSet(setting['quality'], setting['dataset_name'], 'train')
    valid_dset = data.DataSet(setting['quality'], setting['dataset_name'], 'val')

    model = training_testing.training(model, train_dset, valid_dset, device, True, hyperparameter)

    test_dset = data.DataSet(setting['quality'], setting['dataset_name'], 'test')
    training_testing.testing(model, test_dset, device)

