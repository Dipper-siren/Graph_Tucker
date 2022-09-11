import numpy as np
import torch
from torch.nn.init import xavier_normal_


class TuckER(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(TuckER, self).__init__()

        self.E = torch.nn.Embedding(len(d.entities), d1, max_norm=2, norm_type=3)
        self.graph_E = torch.nn.Embedding(len(d.entities), d1)
        # self.graph_E = torch.zeros((len(d.entities), d1))
        self.R_low = torch.nn.Embedding(len(d.relations), d2)
        # self.R_high = torch.nn.Embedding(len(d.relations), d2)
        self.W_low = torch.nn.Parameter(torch.tensor(np.random.uniform(0, 0.5, (d2, d1, d1)),
                                                     dtype=torch.float, device="cuda", requires_grad=True))
        self.W_high = torch.nn.Parameter(torch.tensor(np.random.uniform(0.5, 1, (d2, d1, d1)),
                                                     dtype=torch.float, device="cuda", requires_grad=True))
        self.bias = kwargs["bias"]  # 作为超参 参与训练
        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        self.loss = torch.nn.BCELoss()
        self.d = d
        self.bn0 = torch.nn.BatchNorm1d(d1)
        self.bn1 = torch.nn.BatchNorm1d(d1)
        

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R_low.weight.data)
        # xavier_normal_(self.R_high.weight.data)

    def forward(self, e1_idx, r_idx):
        # self.E.weight.data = self.E.weight.data + self.E2.weight.data
        # self.W = self.W_low + self.W_high

        e1 = self.E(e1_idx)
        # print(e1)
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        r_low = self.R_low(r_idx)
        # print(r)
        W_mat = torch.mm(r_low, self.W_low.view(r_low.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat) 
        x = x.view(-1, e1.size(1))      
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.E.weight.transpose(1, 0))
        pred_1 = torch.sigmoid(x)

        r_high = self.R_low(r_idx)

        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))
        W_mat = torch.mm(r_high, self.W_high.view(r_high.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)
        x = torch.bmm(x, W_mat)
        x = x.view(-1, e1.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.E.weight.transpose(1, 0))
        pred_2 = torch.sigmoid(x)

        # 计算margin_loss
        W_temp = self.W_low - self.W_high
        W_temp += self.bias
        W_temp = torch.clamp(W_temp, -0.)
        W_temp = torch.mean(W_temp)
        pre = (pred_1 + pred_2) / 2

        return pre, W_temp  # , DURA_loss

    def forward_2(self, e1_idx, r_idx):    # 使用上下文的嵌入结果做预测。
        # self.E.weight.data = self.E.weight.data + self.E2.weight.data
        # self.W = self.W_low + self.W_high
        temp1 = self.E(e1_idx)
        temp2 = self.graph_E(e1_idx)
        graph_E = (temp1 + temp2) / 2
        e1 = graph_E
        # e1 = self.E(e1_idx)
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        r_low = self.R_low(r_idx)
        # print(r)
        W_mat = torch.mm(r_low, self.W_low.view(r_low.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat)
        x = x.view(-1, e1.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        # x = torch.mm(x, (self.E.weight.transpose(1, 0)+self.graph_E.weight.transpose(1, 0)) / 2)
        x = torch.mm(x, self.E.weight.transpose(1, 0))
        pred_1 = torch.sigmoid(x)
        r_high = self.R_low(r_idx)

        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))
        W_mat = torch.mm(r_high, self.W_high.view(r_high.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)
        x = torch.bmm(x, W_mat)
        x = x.view(-1, e1.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        # x = torch.mm(x, (self.E.weight.transpose(1, 0)+self.graph_E.weight.transpose(1, 0)) / 2)
        x = torch.mm(x, self.E.weight.transpose(1, 0))
        pred_2 = torch.sigmoid(x)

        # 计算margin_loss
        W_temp = self.W_low - self.W_high
        W_temp += self.bias
        W_temp = torch.clamp(W_temp, -0.)
        W_temp = torch.mean(W_temp)
        pre = (pred_1 + pred_2) / 2

        return pre, W_temp  # , DURA_loss

    def forward_graph_entity(self, e1_idx, r_idx, i):    # 使用上下文的嵌入结果做预测。
        e1 = self.E(e1_idx)
        x = e1
        x = x.view(-1, 1, e1.size(1))
        r_low = self.R_low(r_idx)
        W_mat = torch.mm(r_low, self.W_low.view(r_low.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        x = torch.bmm(x, W_mat)
        x = x.view(-1, e1.size(1))  # 128*200 * 200*104
        graph_E1 = x

        r_high = self.R_low(r_idx)
        x = e1
        x = x.view(-1, 1, e1.size(1))
        W_mat = torch.mm(r_high, self.W_high.view(r_high.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        x = torch.bmm(x, W_mat)
        x = x.view(-1, e1.size(1))
        graph_E2 = x
        graph_E = (graph_E1 + graph_E2) / 2
        graph_E = graph_E.mean(axis=0)    # 在列向量上做的均值操作。
        self.graph_E.weight.data[i] = graph_E   # 替换对应的embeddings
