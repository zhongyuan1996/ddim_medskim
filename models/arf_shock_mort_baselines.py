import math
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.operations import *

class F_LSTM(nn.Module):
    def __init__(self, vocab_size_list, demo_dim = 76, d_model = 256, dropout = 0.5, length=48):
        super(F_LSTM, self).__init__()
        self.embedding1 = nn.Sequential(nn.Embedding(vocab_size_list[0]+1, d_model), nn.Dropout(dropout))
        self.embedding2 = nn.Sequential(nn.Embedding(vocab_size_list[1]+1, d_model), nn.Dropout(dropout))
        self.embedding3 = nn.Sequential(nn.Embedding(vocab_size_list[2]+1, d_model), nn.Dropout(dropout))
        self.embedding4 = nn.Sequential(nn.Embedding(vocab_size_list[3]+1, d_model), nn.Dropout(dropout))
        self.demo_embedding = nn.Sequential(nn.Linear(demo_dim, d_model), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout)
        self.output_mlp = nn.Sequential(nn.Linear(d_model, 2))
        self.pooler = MaxPoolLayer()
        self.sig = nn.Sigmoid()
        self.rnns = nn.LSTM(d_model * 5, d_model, 1, bidirectional=False, batch_first=True)

    def forward(self, diag, drug, lab, proc, demo):
        diag_emb = self.embedding1(diag).sum(-2)
        drug_emb = self.embedding2(drug).sum(-2)
        lab_emb = self.embedding3(lab).sum(-2)
        proc_emb = self.embedding4(proc).sum(-2)
        demo_emb = self.demo_embedding(demo).unsqueeze(1).repeat(1, diag_emb.size(1), 1)

        concat_emb = torch.cat([diag_emb, drug_emb, lab_emb, proc_emb, demo_emb], dim=-1)
        rnn_out, _ = self.rnns(concat_emb)
        x = self.pooler(rnn_out)
        x = self.output_mlp(x)
        # x = self.sig(x)
        return x

class F_CNN(nn.Module):

    def __init__(self, vocab_size_list, demo_dim = 76, d_model = 256, dropout = 0.5, length=48):
        super(F_CNN, self).__init__()
        self.embedding1 = nn.Sequential(nn.Embedding(vocab_size_list[0]+1, d_model), nn.Dropout(dropout))
        self.embedding2 = nn.Sequential(nn.Embedding(vocab_size_list[1]+1, d_model), nn.Dropout(dropout))
        self.embedding3 = nn.Sequential(nn.Embedding(vocab_size_list[2]+1, d_model), nn.Dropout(dropout))
        self.embedding4 = nn.Sequential(nn.Embedding(vocab_size_list[3]+1, d_model), nn.Dropout(dropout))
        self.demo_embedding = nn.Sequential(nn.Linear(demo_dim, d_model), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout)
        self.output_mlp = nn.Sequential(nn.Linear(d_model, 2))
        self.pooler = MaxPoolLayer()
        self.sig = nn.Sigmoid()
        self.cnn = nn.Conv1d(d_model * 5, d_model, 3, padding=1)

    def forward(self, diag, drug, lab, proc, demo):
        diag_emb = self.embedding1(diag).sum(-2)
        drug_emb = self.embedding2(drug).sum(-2)
        lab_emb = self.embedding3(lab).sum(-2)
        proc_emb = self.embedding4(proc).sum(-2)
        demo_emb = self.demo_embedding(demo).unsqueeze(1).repeat(1, diag_emb.size(1), 1)

        concat_emb = torch.cat([diag_emb, drug_emb, lab_emb, proc_emb, demo_emb], dim=-1)
        concat_emb = concat_emb.transpose(-2, -1)
        cnn_out = self.cnn(concat_emb).transpose(-2, -1)
        x = self.pooler(cnn_out)
        x = self.output_mlp(x)
        # x = self.sig(x)
        return x

class Raim(nn.Module):
    def __init__(self, vocab_size_list, demo_dim = 76, d_model = 256, dropout = 0.5, length=48):
        super(Raim, self).__init__()
        self.embedding1 = nn.Sequential(nn.Embedding(vocab_size_list[0]+1, d_model), nn.Dropout(dropout))
        self.embedding2 = nn.Sequential(nn.Embedding(vocab_size_list[1]+1, d_model), nn.Dropout(dropout))
        self.embedding3 = nn.Sequential(nn.Embedding(vocab_size_list[2]+1, d_model), nn.Dropout(dropout))
        self.embedding4 = nn.Sequential(nn.Embedding(vocab_size_list[3]+1, d_model), nn.Dropout(dropout))
        self.demo_embedding = nn.Sequential(nn.Linear(demo_dim, d_model), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout)
        self.output_mlp = nn.Sequential(nn.Linear(d_model, 2))
        self.pooler = MaxPoolLayer()
        self.hidden_size = d_model
        self.hidden2label = nn.Linear(d_model, 2)
        self.grucell = nn.GRUCell(d_model, d_model)
        self.mlp_for_x = nn.Linear(d_model, 1, bias=False)
        self.mlp_for_hidden = nn.Linear(d_model, length, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(batch_size, self.hidden_size))

    def forward(self, diag, drug, lab, proc, demo):
        diag_emb = self.embedding1(diag).sum(-2)
        drug_emb = self.embedding2(drug).sum(-2)
        lab_emb = self.embedding3(lab).sum(-2)
        proc_emb = self.embedding4(proc).sum(-2)
        demo_emb = self.demo_embedding(demo)

        x = diag_emb + drug_emb + lab_emb + proc_emb

        x2 = lab_emb + proc_emb
        self.hidden = self.init_hidden(x.size(0)).to(x.device)
        for i in range(x.size(1)):
            tt = x[:, 0:i + 1, :].reshape(x.size(0), (i + 1) * x[:, 0:i + 1, :].shape[2])
            if i < x.size(1) - 1:
                padding = torch.zeros(x.size(0), x.size(1)*x.size(2) - tt.shape[1]).to(x.device)
                self.temp1 = torch.cat((tt, padding), 1)
            else:
                self.temp1 = tt

            self.input_padded = self.temp1.reshape(x.size(0), x.size(1), x.size(-1))

            temp_guidance = torch.zeros(x.size(0), x.size(1), 1).to(x.device)

            if i > 0:

                zero_idx = torch.where(torch.sum(x2[:, :i, 0], dim=1) == 0)
                if len(zero_idx[0]) > 0:
                    temp_guidance[zero_idx[0], :i, 0] = 1

            temp_guidance[:, i, :] = 1

            self.guided_input = torch.mul(self.input_padded, temp_guidance)
            self.t1 = self.mlp_for_x(self.guided_input) + self.mlp_for_hidden(self.hidden).reshape(x.size(0), x.size(1), 1)
            self.t1_softmax = self.softmax(self.t1)
            final_output = torch.mul(self.input_padded, self.t1_softmax)

            context_vec = torch.sum(final_output, dim=1)

            self.hx = self.grucell(context_vec, self.hidden)
            self.hidden = self.hx

        y = self.hidden2label(self.hidden + demo_emb)
        return y

class DCMN(nn.Module):
    def __init__(self, vocab_size_list, demo_dim = 76, d_model = 256, dropout = 0.5, length=48):
        super(DCMN, self).__init__()
        self.embedding1 = nn.Sequential(nn.Embedding(vocab_size_list[0]+1, d_model), nn.Dropout(dropout))
        self.embedding2 = nn.Sequential(nn.Embedding(vocab_size_list[1]+1, d_model), nn.Dropout(dropout))
        self.embedding3 = nn.Sequential(nn.Embedding(vocab_size_list[2]+1, d_model), nn.Dropout(dropout))
        self.embedding4 = nn.Sequential(nn.Embedding(vocab_size_list[3]+1, d_model), nn.Dropout(dropout))
        self.demo_embedding = nn.Sequential(nn.Linear(demo_dim, d_model), nn.Dropout(dropout))
        self.batchnorm1 = nn.BatchNorm1d(d_model)
        self.batchnorm2 = nn.BatchNorm1d(d_model)
        self.conv = nn.Conv1d(d_model, d_model, 3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout)
        self.output_mlp = nn.Sequential(nn.Linear(d_model, 2))
        self.c_emb = nn.LSTM(d_model, d_model, 1, bidirectional=False, batch_first=True)
        self.c_out = nn.LSTM(d_model, d_model, 1, bidirectional=False, batch_first=True)
        self.w_emb = nn.LSTM(d_model, d_model, 1, bidirectional=False, batch_first=True)
        self.w_out = nn.LSTM(d_model, d_model, 1, bidirectional=False, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.linear3 = nn.Linear(d_model, d_model)
        self.linear4 = nn.Linear(d_model, d_model)
        self.gate_linear = nn.Linear(d_model, d_model)
        self.gate_linear2 = nn.Linear(d_model, d_model)
        self.pooler = MaxPoolLayer()
        self.sigmoid = nn.Sigmoid()

    def forward(self, diag, drug, lab, proc, demo):
        diag_emb = self.embedding1(diag).sum(-2)
        drug_emb = self.embedding2(drug).sum(-2)
        lab_emb = self.embedding3(lab).sum(-2)
        proc_emb = self.embedding4(proc).sum(-2)
        demo_emb = self.demo_embedding(demo)

        x1 = diag_emb + drug_emb
        x2 = lab_emb + proc_emb
        s = demo_emb

        wm_embedding_memory, _ = self.w_emb(x1)
        wm_out_query, _ = self.w_out(x1)
        cm_embedding_memory, _ = self.c_emb(x2)
        cm_out_query, _ = self.c_out(x2)
        wm_in = cm_out_query[:, -1]
        cm_in = wm_out_query[:, -1]
        w_embedding_E = self.linear1(wm_embedding_memory)
        w_embedding_F = self.linear2(wm_embedding_memory)
        wm_out = torch.matmul(wm_in.unsqueeze(1), w_embedding_E.permute(0, 2, 1))
        wm_prob = torch.softmax(wm_out, dim=-1)
        wm_contex = torch.matmul(wm_prob, w_embedding_F)
        wm_gate_prob = torch.sigmoid(self.gate_linear(wm_in)).unsqueeze(1)
        wm_dout = wm_contex * wm_gate_prob + wm_in.unsqueeze(1) * (1 - wm_gate_prob)

        c_embedding_E = self.linear3(cm_embedding_memory)
        c_embedding_F = self.linear4(cm_embedding_memory)
        cm_out = torch.matmul(cm_in.unsqueeze(1), c_embedding_E.permute(0, 2, 1))
        cm_prob = torch.softmax(cm_out, dim=-1)
        cm_contex = torch.matmul(cm_prob, c_embedding_F)
        cm_gate_prob = torch.sigmoid(self.gate_linear2(cm_in)).unsqueeze(1)
        cm_dout = cm_contex * cm_gate_prob + cm_in.unsqueeze(1) * (1 - cm_gate_prob)
        output = wm_dout + cm_dout
        output = self.output_mlp(output.squeeze() + s)
        return output



