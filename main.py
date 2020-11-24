import wandb
from models import mlp
import torch
import torch.nn as nn
import numpy as np
from data_loader import data_loader
from sklearn.utils import class_weight
from skorch import NeuralNetClassifier
from skorch.callbacks import *
from skorch.dataset import Dataset
from skorch.helper import predefined_split


def load_data(filepath, cfg, feature_list, start_feature=5, mask_value=0):
    X_train_data, y_train_data = data_loader.load_data(
        datafile=filepath + 'normalized_training.csv',
        flare_label=cfg.flare_label, series_len=cfg.seq_len,
        start_feature=start_feature, n_features=cfg.n_features,
        mask_value=mask_value, feature_list=feature_list)
    X_train_data = np.reshape(X_train_data,
                              (len(X_train_data), cfg.n_features))
    X_train_data = X_train_data.astype(np.float32)
    y_train_data = data_loader.label_transform(y_train_data)

    X_valid_data, y_valid_data = data_loader.load_data(
        datafile=filepath + 'normalized_validation.csv',
        flare_label=cfg.flare_label, series_len=cfg.seq_len,
        start_feature=start_feature, n_features=cfg.n_features,
        mask_value=mask_value, feature_list=feature_list)
    X_valid_data = np.reshape(X_valid_data,
                              (len(X_valid_data), cfg.n_features))
    X_valid_data = X_valid_data.astype(np.float32)
    y_valid_data = data_loader.label_transform(y_valid_data)

    X_test_data, y_test_data = data_loader.load_data(
        datafile=filepath + 'normalized_testing.csv',
        flare_label=cfg.flare_label, series_len=cfg.seq_len,
        start_feature=start_feature, n_features=cfg.n_features,
        mask_value=mask_value, feature_list=feature_list)
    X_test_data = np.reshape(X_test_data,
                             (len(X_test_data), cfg.n_features))
    X_test_data = X_test_data.astype(np.float32)
    y_test_data = data_loader.label_transform(y_test_data)

    return X_train_data, X_valid_data, X_test_data, y_train_data, y_valid_data, y_test_data


def init_callbacks():
    print('Metric defined')


def get_device(cfg):
    # GPU check
    use_cuda = cfg.cuda and torch.cuda.is_available()
    if cfg.cuda and torch.cuda.is_available():
        print("Cuda enabled and available")
    elif cfg.cuda and not torch.cuda.is_available():
        print("Cuda enabled not not available, CPU used.")
    elif not cfg.cuda:
        print("Cuda disabled")

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    return device


def get_feature_list():
    sharps = ['USFLUX', 'SAVNCPP', 'TOTPOT', 'ABSNJZH', 'SHRGT45', 'AREA_ACR',
              'R_VALUE', 'TOTUSJH', 'TOTUSJZ', 'MEANJZH', 'MEANJZD', 'MEANPOT',
              'MEANSHR', 'MEANALP', 'MEANGAM', 'MEANGBZ', 'MEANGBT',
              'MEANGBH']  # 18
    lorentz = ['TOTBSQ', 'TOTFX', 'TOTFY', 'TOTFZ', 'EPSX', 'EPSY',
               'EPSZ']  # 7
    history_features = ['Bdec', 'Cdec', 'Mdec', 'Xdec', 'Edec', 'logEdec',
                        'Bhis', 'Chis', 'Mhis', 'Xhis', 'Bhis1d', 'Chis1d',
                        'Mhis1d', 'Xhis1d', 'Xmax1d']  # 15
    listofuncorrfeatures = ['TOTUSJH', 'SAVNCPP', 'ABSNJZH', 'TOTPOT',
                            'AREA_ACR', 'Cdec', 'Chis', 'Edec', 'Mhis',
                            'Xmax1d', 'Mdec', 'MEANPOT', 'R_VALUE', 'Mhis1d',
                            'MEANGAM', 'TOTFX', 'MEANJZH', 'MEANGBZ', 'TOTFZ',
                            'TOTFY', 'logEdec', 'EPSZ', 'MEANGBH', 'MEANJZD',
                            'Xhis1d', 'Xdec', 'Xhis', 'EPSX', 'EPSY', 'Bhis',
                            'Bdec', 'Bhis1d']  # 32
    all_f = sharps + lorentz + history_features
    return all_f


def main():
    run = wandb.init()
    cfg = wandb.config
    filepath = './data/' + cfg.dataset

    device = get_device(cfg)
    feature_list = get_feature_list()

    X_train, X_valid, X_test, y_train, y_valid, y_test = load_data(filepath, cfg, feature_list)
    valid_ds = Dataset(X_valid, y_valid)

    model = mlp.MLPModule(input_units=cfg.n_features,
                          hidden_units=cfg.hidden_units,
                          num_hidden=cfg.layers,
                          dropout=cfg.dropout).to(device)

    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(y_train),
                                                      y_train)
    net = NeuralNetClassifier(model, max_epochs=cfg.epochs,
                              batch_size=cfg.batch_size,
                              criterion=nn.CrossEntropyLoss,
                              criterion__weight=torch.FloatTensor(
                                  class_weights).to(device),
                              optimizer=torch.optim.SGD,
                              optimizer__lr=cfg.learning_rate,
                              optimizer__weight_decay=cfg.weight_decay,
                              device=device,
                              train_split=predefined_split(valid_ds),
                              callbacks=[],
                              iterator_train__shuffle=True if cfg.shuffle else False,
                              warm_start=False)

    net.initialize()
    net.fit(X_train, y_train)

    y_pred = net.predict(X_test)
    # tss_test_score = skorch_utils.get_tss(y_test, y_pred)


if __name__ == '__main__':
    main()
