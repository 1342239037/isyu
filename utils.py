import torch
from torch.utils.data import Dataset, DataLoader
import dgl
import sklearn.metrics as met
from scipy.stats import pearsonr, spearmanr
from math import sqrt
import time
import os
from tqdm import tqdm


class metrics():
    # correlation coefficient
    def pearson(y_true, y_pred):
        if len(y_true) < 2:
            return [1, 1]
        return pearsonr(y_true, y_pred)

    def spearman(y_true, y_pred):
        return spearmanr(y_true, y_pred)

    # Regression metrics
    def mse(y_true, y_pred):
        return met.mean_squared_error(y_true, y_pred)

    def mae(y_true, y_pred):
        return met.mean_absolute_error(y_true, y_pred)

    def rmse(y_true, y_pred):
        return sqrt(metrics.mse(y_true, y_pred))

    def r2(y_true, y_pred):
        return met.r2_score(y_true, y_pred)

    # Classification metrics
    def acc(yt, yp):
        return met.accuracy_score(yt, yp)

    def kappa(yt, yp):
        return met.cohen_kappa_score(yt, yp)

    def f1(yt, yp):
        return met.f1_score(yt, yp)

    def recall(yt, yp):
        # TPR
        return met.recall_score(yt, yp)

    def bacc(yt, yp):
        # balanced accuracy
        return met.balanced_accuracy_score(yt, yp)

    def roc_auc(yt, yp):
        return met.roc_auc_score(yt, yp)

    def prec(yt, yp):
        return met.precision_score(yt, yp)

    def pr_auc(yt, yp):
        pass


def collate_merg(data, device):
    gAB = []
    gBA = []
    c = []
    y = []
    for item in data:
        gAB.append(item[0][0])
        gAB.append(item[0][1])
        gBA.append(item[0][1])
        gBA.append(item[0][0])
        c.append(torch.Tensor(item[0][2]))
        y.append(item[1])
    outgAB = dgl.batch(gAB).to(device)
    outgBA = dgl.batch(gBA).to(device)
    outc = torch.stack(c).to(device)
    Y = torch.Tensor(y).to(device)
    return (outgAB, outgBA, outc, Y)


class dataset(Dataset):
    def __init__(self, x, y, device, transform=None, target_transform=None):
        super(dataset, self).__init__()
        self.ddc = x
        self.labels = y
        self.transform = transform
        self.target_transform = target_transform
        self.device = device

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        ddc = self.ddc[index]
        label = self.labels[index]
        if self.transform:
            ddc = self.transform(ddc, self.device)
        if self.target_transform:
            label = self.target_transform(label, self.device)
        return ddc, label


def log(file, modelName, trainloss, validloss, validr2, \
        validpearson, addepoch=None):
    if not os.path.exists(f'./run/{modelName}'):
        os.makedirs(f'./run/{modelName}', exist_ok=True)
    filename = f"./run/{modelName}/{file}.log"
    content = f"{modelName}--{time.strftime('%m/%d-%H:%M:%S', time.localtime())}"
    if addepoch is not None:
        content = content + addepoch
    content = content \
              + ",%11.4f" % (trainloss) \
              + ",%11.4f" % (validloss) \
              + ",%11.6f" % (validr2) \
              + ",%11.6f" % (validpearson[0])
    with open(filename, "a") as fl:
        fl.write(content + "\n")


def evaluate(model, ds, device=torch.device("cpu")):
    data = dataset([i[0] for i in ds], [i[1] for i in ds], device)
    dataL = DataLoader(data, batch_size=256, shuffle=False, collate_fn=lambda x: collate_merg(x, device))

    model.eval()
    y_pred = []
    y_label = []

    with torch.no_grad():
        with tqdm(total=len(dataL)) as tepoch:
            for (dAB, dBA, c, y) in dataL:
                pred1,_ = model((dAB, c))
                pred2,_ = model((dBA, c))
                y_pred.append((pred1 + pred2) / 2)
                y_label.append(y)
                tepoch.update(1)
    if len(y_label) > 1:
        offset = y_pred[-2].shape[0] - y_pred[-1].shape[0]
        y_pred[-1].resize_(y_pred[-2].shape)
        y_label[-1].resize_(y_label[-2].shape)
        y_pred = torch.stack(y_pred, 0).view(-1).cpu()
        y_label = torch.stack(y_label, 0).view(-1).cpu()
        if offset > 0:
            y_pred = y_pred[:-offset]
            y_label = y_label[:-offset]
    else:
        y_label = y_label[0].reshape(-1).cpu()
        y_pred = y_pred[0].reshape(-1).cpu()

    mse = metrics.mse(y_label, y_pred)
    mae = metrics.mae(y_label, y_pred)
    rmse = metrics.rmse(y_label, y_pred)
    r2 = metrics.r2(y_label, y_pred)
    pearson = metrics.pearson(y_label, y_pred)
    spearman = metrics.spearman(y_label, y_pred)

    print("MSE: " + str(mse), end=" ")
    print("MAE: " + str(mae), end=" ")
    print("RMSE: " + str(rmse), end=" ")
    print("r2: " + str(r2), end=" ")
    print("pearson: " + str(pearson), end=" ")
    print("spearman: " + str(spearman))

    return (y_label, y_pred), (mse, mae, rmse, r2, pearson, spearman)
