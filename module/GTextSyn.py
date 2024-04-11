import torch as t
import torch.nn as nn
from .layers import MultiHeadAttention, DrugGraphConv
import torch.nn.functional as F
device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')

class SimCLR(nn.Module):
    def __init__(self, temperature=0.1):
        super(SimCLR, self).__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        # 数据归一化处理
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        z1_large = z1
        z2_large = z2
        step_batch_size = z1.shape[0]
        labels = t.eye(step_batch_size * 2)[t.arange(step_batch_size)].to(device)
        masks = t.eye(step_batch_size)[t.arange(step_batch_size)].to(device)
        logits_aa = (t.matmul(z1, z1_large.t()) / self.temperature)
        logits_aa = (logits_aa - masks * logits_aa)
        logits_bb = (t.matmul(z2, z2_large.t()) / self.temperature)
        logits_bb = (logits_bb - masks * logits_bb)
        logits_ab = (t.matmul(z1, z2_large.t()) / self.temperature)
        logits_ba = (t.matmul(z2, z1_large.t()) / self.temperature)
        criterion = nn.CrossEntropyLoss()
        loss_a = criterion(labels, t.cat([logits_ab, logits_aa], 1))
        loss_b = criterion(labels, t.cat([logits_ba, logits_bb], 1))
        loss = (loss_a + loss_b)
        return loss
class encoder1(nn.Module):
    def __init__(self):
        super(encoder1, self).__init__()
        self.encoder_MLP = nn.Sequential(
            nn.Linear(222, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 222),
            nn.ReLU()
        )

    def forward(self, data):
        encoderchu = self.encoder_MLP(data)
        return encoderchu




class GTextSyn(nn.Module):
    def __init__(self, params):
        super(GTextSyn, self).__init__()

        self.emb_dim = 37
        self.dropout = params["dropout"]

        self.drug_graph_conv = DrugGraphConv(74, self.emb_dim)

        self.cell_MLP = nn.Sequential(
            nn.Linear(956, 3072),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(3072, 1024),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(1024, self.emb_dim),
            nn.ReLU()
        )

        self.bilstm = nn.LSTM(37, 37, num_layers=2, batch_first=True, bidirectional=True, dropout=self.dropout)
        self.rnn = nn.RNN(37, 37, num_layers=2, batch_first=True, bidirectional=True, dropout=self.dropout)

        self.attention = MultiHeadAttention(74, 74, 2)
        self.fc_in = nn.Linear(444, 256)
        self.fc_out = nn.Linear(256, 1)
        self.encoder = encoder1()
        self.sir = SimCLR()

    def forward(self, data):
        drug = data[0]
        cell = data[1]
        cell = F.normalize(cell, dim=1)
        cell_emb = self.cell_MLP(cell).view(-1, 1, 37)

        drug_emb = self.drug_graph_conv(drug)

        sequence = t.cat((drug_emb, cell_emb), dim=1)

        output, _ = self.bilstm(sequence)
        output1, _ = self.rnn(sequence)
        out, outw = self.attention(output, return_attention=True)
        out1, outw2 = self.attention(output1, return_attention=True)
        #
        out2 = out.view(out.shape[0], -1)
        #
        out1 = out1.view(out1.shape[0], -1)
        outencoder = self.encoder(out2)
        outencoder2 = self.encoder(out1)
        outloss = self.sir(outencoder, outencoder2)
        out = t.cat((out2, out1), dim=1)
        out2 = F.relu(self.fc_in(out))
        out2 = self.fc_out(out2)

        return out2, outloss