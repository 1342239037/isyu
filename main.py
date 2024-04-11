import numpy as np
import random
import time
import torch
from torch.optim.lr_scheduler import ExponentialLR
import dgl
import pickle
from tqdm import tqdm
from sklearn.model_selection import KFold
from module.GTextSyn import GTextSyn
from utils import metrics, dataset, DataLoader, log, collate_merg


P = {
    "SEED": 5,
    "EPOCHES": 100,
    "BATCH_SIZE": 64,
    "TEST_BATCH": 256,
    "dropout": 0.1,
    "lr": 0.0003,
    "lr_gamma": 0.95,
}

dgl.random.seed(P["SEED"])
random.seed(P["SEED"])
torch.manual_seed(P["SEED"])
torch.cuda.manual_seed(P["SEED"])
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GTextSyn(P)
model.to(device)

MODEL_NAME = f"model-{time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())}"

optimizer = torch.optim.Adam(model.parameters(), lr=P['lr'])
criterion = torch.nn.MSELoss()
scheduler = ExponentialLR(optimizer, gamma=P["lr_gamma"])

with open(f"data/ONEIL_testfan.pkl", 'rb') as fp:
    ds_test = pickle.load(fp)
test_data = dataset([i[0] for i in ds_test], [i[1] for i in ds_test], device)
testDL = DataLoader(test_data, batch_size=P["TEST_BATCH"], shuffle=True, collate_fn=lambda x: collate_merg(x, device))
def fwd_pass(batch_x, batch_y=None, train=False):

    loss = None
    out ,incoderloss= model(batch_x)

    if train:
        y = batch_y.view(-1, 1)
        optimizer.zero_grad()
        loss = criterion(out, y)
        lossall = loss + incoderloss*0.0001
        lossall.backward()
        optimizer.step()

    return loss, out

def test(dataL, la="NONE"):
    model.eval()
    y_pred = []
    y_label = []

    print(f"---{la}ing:---")

    with torch.no_grad():
        with tqdm(total=len(dataL)) as tepoch:
            for (dAB, dBA, c, y) in dataL:
                _, pred1 = fwd_pass((dAB, c))
                _, pred2 = fwd_pass((dBA, c))
                y_pred.append((pred1 + pred2) / 2)
                y_label.append(y)
                tepoch.update(1)
        if len(y_label) > 1:
            offset = y_pred[-2].shape[0] - y_pred[-1].shape[0]
            y_pred[-1].resize_(y_pred[-2].shape)
            y_label[-1].resize_(y_label[-2].shape)
            y_pred = torch.stack(y_pred, 0).view(-1).cpu()
            y_label = torch.stack(y_label, 0).view(-1).cpu()
            if offset > 0 :
                y_pred = y_pred[:-offset]
                y_label = y_label[:-offset]
        else:
            y_label = y_label[0].reshape(-1).cpu()
            y_pred = y_pred[0].reshape(-1).cpu()
        loss = criterion(y_pred, y_label)
    mse = metrics.mse(y_label, y_pred)
    mae = metrics.mae(y_label, y_pred)
    rmse = metrics.rmse(y_label, y_pred)
    try:
        r2 = metrics.r2(y_label, y_pred)
        pearson = metrics.pearson(y_label, y_pred)
    except Exception as e:
        print(e)
    print("MSE: " + str(mse), end=" ")
    print("MAE: " + str(mae), end=" ")
    print("RMSE: " + str(rmse), end=" ")
    print("r2: " + str(r2), end=" ")
    print("pearson: " + str(pearson))
    return loss, mae, rmse, r2, pearson

def train(traindata):
    X = traindata[0]
    Y = traindata[1]
    KF = KFold(n_splits=5, shuffle=True, random_state=5)
    k = 0
    for epoch in range(P["EPOCHES"]):
        losses = []

        for i, (train_ind, valid_ind) in enumerate(KF.split(X)):
            model.train()
            split_trainx, split_trainy = [X[i] for i in train_ind], [Y[i] for i in train_ind]
            split_validx, split_validy = [X[i] for i in valid_ind], [Y[i] for i in valid_ind]

            _d = dataset(split_trainx, split_trainy, device)
            trainDL = DataLoader(_d, batch_size=P["BATCH_SIZE"], shuffle=True,
                                 collate_fn=lambda x: collate_merg(x, device))
            _d = dataset(split_validx, split_validy, device)
            validDL = DataLoader(_d, batch_size=P["TEST_BATCH"], shuffle=True,
                                 collate_fn=lambda x: collate_merg(x, device))

            print("=============================================\n---training:---")

            with tqdm(total=len(trainDL)) as tepoch:
                for j, (dAB, dBA, c, y) in enumerate(trainDL):
                    loss1, _ = fwd_pass((dAB, c), y, train=True)
                    loss2, _ = fwd_pass((dBA, c), y, train=True)
                    losses.append((loss1.item() + loss2.item()) / 2)
                    loss_mean = np.array(losses).mean()
                    tepoch.set_description(f"Epoch:{epoch + 1}/{P['EPOCHES']}--Fold:{i + 1}/5")
                    tepoch.set_postfix(loss=loss_mean)
                    tepoch.update(1)
            print(f"Average Loss: {np.array(losses).mean()}")

            valid_loss, _, _, r2, pearson = test(validDL, "valid")

            log("model", MODEL_NAME, loss_mean, valid_loss.item(), r2, pearson, f"--{epoch + 1}--{i}")

        scheduler.step()
        if k % 10 == 0:
            mse, mae, rmse, r2, pearson = test(testDL, "test")
            log("model", MODEL_NAME,mse.item(),rmse, r2, pearson, "---test---")

            print("test_mse: ", mse.item(), "\ntest_mae: ", mae, "\ntest_rmse: ", rmse, "\ntest_r2: ", r2,
                  "\ntest_pearson: ", pearson)
        k += 1


def main(study):

    global MODEL_NAME
    MODEL_NAME = f"{study}_" + MODEL_NAME

    with open(f"./data/{study}_trainfan.pkl", 'rb') as fp:
        ds = pickle.load(fp)
    X = [i[0] for i in ds]
    Y = [i[1] for i in ds]

    train((X, Y))
    torch.save(model, f'./run/{MODEL_NAME}/model.pth')
    # torch.save(model, f'./auxiliaryExp/ONEIL.pth')
    mse, mae, rmse, r2, pearson = test(testDL, "test")
    log("model", MODEL_NAME, 0, mse.item(), r2, pearson, "---test---")

    print("test_mse: ", mse.item(), "\ntest_mae: ", mae, "\ntest_rmse: ", rmse, "\ntest_r2: ", r2, "\ntest_pearson: ", pearson)



if __name__ == "__main__":
    main("ONEIL")
