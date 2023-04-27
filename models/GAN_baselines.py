import math
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils.diffUtil import get_beta_schedule
from models.unet import *
import os
import yaml
import argparse


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_seq_len):

        super(PositionalEncoding, self).__init__()

        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(max_seq_len)])

        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
        position_encoding = torch.from_numpy(position_encoding.astype(np.float32))

        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.cat((pad_row, position_encoding))

        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding,
                                                     requires_grad=False)
        self.max_pos = max_seq_len

    def forward(self, input_len):

        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor

        pos = np.zeros([input_len.size(0), self.max_pos])
        for ind, length in enumerate(input_len):
            for pos_ind in range(1, length + 1):
                pos[ind, pos_ind - 1] = pos_ind
        input_pos = tensor(pos)
        return self.position_encoding(input_pos), input_pos


class MaxPoolLayer(nn.Module):
    """
    A layer that performs max pooling along the sequence dimension
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, mask_or_lengths=None):
        """
        inputs: tensor of shape (batch_size, seq_len, hidden_size)
        mask_or_lengths: tensor of shape (batch_size) or (batch_size, seq_len)

        returns: tensor of shape (batch_size, hidden_size)
        """
        bs, sl, _ = inputs.size()
        if mask_or_lengths is not None:
            if len(mask_or_lengths.size()) == 1:
                mask = (torch.arange(sl, device=inputs.device).unsqueeze(0).expand(bs, sl) >= mask_or_lengths.unsqueeze(
                    1))
            else:
                mask = mask_or_lengths
            inputs = inputs.masked_fill(mask.unsqueeze(-1).expand_as(inputs), float('-inf'))
        max_pooled = inputs.max(1)[0]
        return max_pooled


class Attention(nn.Module):
    def __init__(self, in_feature, num_head, dropout):
        super(Attention, self).__init__()
        self.in_feature = in_feature
        self.num_head = num_head
        self.size_per_head = in_feature // num_head
        self.out_dim = num_head * self.size_per_head
        assert self.size_per_head * num_head == in_feature
        self.q_linear = nn.Linear(in_feature, in_feature, bias=False)
        self.k_linear = nn.Linear(in_feature, in_feature, bias=False)
        self.v_linear = nn.Linear(in_feature, in_feature, bias=False)
        self.fc = nn.Linear(in_feature, in_feature, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(in_feature)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = key.size(0)
        res = query
        query = self.q_linear(query)
        key = self.k_linear(key)
        value = self.v_linear(value)

        query = query.view(batch_size, self.num_head, -1, self.size_per_head)
        key = key.view(batch_size, self.num_head, -1, self.size_per_head)
        value = value.view(batch_size, self.num_head, -1, self.size_per_head)

        scale = np.sqrt(self.size_per_head)
        energy = torch.matmul(query, key.permute(0, 1, 3, 2)) / scale
        if attn_mask is not None:
            batch_size, q_len, k_len = attn_mask.size()
            attn_mask = attn_mask.unsqueeze(1).expand(batch_size, self.num_head, q_len, k_len)
            energy = energy.masked_fill(attn_mask == 0, -np.inf)

        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(attention, value)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.in_feature)
        attention = attention.sum(dim=1).squeeze().permute(0, 2, 1) / self.num_head
        x = self.fc(x)
        x = self.dropout(x)
        x += res
        x = self.layer_norm(x)
        return x, attention

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace



class classifyer(nn.Module):

    def __init__(self, d_hiddens_tate):
        super().__init__()
        self.layer1 = nn.Linear(d_hiddens_tate, 4 * d_hiddens_tate)
        self.layer2 = nn.Linear(4 * d_hiddens_tate, 2 * d_hiddens_tate)
        self.out = nn.Linear(2 * d_hiddens_tate, 2)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, h):
        h = self.relu(self.layer1(h))
        h = self.relu(self.layer2(h))
        h = self.out(self.drop(h))
        return h

class Linear_generator(nn.Module):

    def __init__(self, h_model=256):
        super(Linear_generator, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(h_model/2)
        self.batch_norm2 = nn.BatchNorm1d(h_model/4)
        self.linear1 = nn.Linear(h_model, h_model/2)
        self.linear2 = nn.Linear(h_model/2, h_model/4)
        self.relu = nn.ReLU()

    def forward(self, z):
        z = self.relu(self.batch_norm1(self.linear1(z))) + z
        z = self.relu(self.batch_norm2(self.linear2(z))) + z
        return z

class encoder_256_64(nn.Module):
    def __init__(self, in_dim = 256, out_dim = 64):
        super(encoder_256_64, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.batch_norm = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.batch_norm(self.linear(x)))

class decoder_64_256(nn.Module):
    def __init__(self, in_dim = 64, out_dim = 256):
        super(decoder_64_256, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.batch_norm = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.batch_norm(self.linear(x)))

class LSTM_generator(nn.Module):
    def __init__(self, d_model, h_model, num_layers = 1):
        super(LSTM_generator, self).__init__()
        self.lstm = nn.LSTM(h_model, d_model, num_layers = num_layers, batch_first=True)
        # self.out = nn.Linear(d_model, d_model)
        # self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm(x)
        return x
        # x, _ = self.lstm(x)
        # return self.relu(self.out(x))

class FCN_generator(nn.Module):

    def __init__(self, h_model):
        super(FCN_generator, self).__init__()
        self.linear1 = nn.Linear(h_model, h_model)
        self.layer2_conv = nn.Conv1d(102, 51, 1, 1)
        self.layer3 = nn.Linear(h_model, h_model)
        self.layer4_conv = nn.Conv1d(153, 51, 1, 1)
        self.down = nn.Linear(h_model, int(h_model/4))
    def forward(self, gray):

        orange = self.linear1(gray)
        brown = self.layer2_conv(torch.cat([gray, orange], dim=1))
        blue = self.layer3(brown)
        red = self.layer4_conv(torch.cat([blue, brown, gray], dim=1))
        res = self.down(red)
        return res

class Discriminator(nn.Module):
    def __init__(self, d_model):
        super(Discriminator, self).__init__()
        self.dense = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.dense(x))

class LSTM_medGAN(nn.Module):
    def __init__(self, vocab_size, d_model, h_model, dropout, dropout_emb, device, generator, num_layers = 1, m = 0, dense_model = 64):
        super().__init__()
        self.device = device
        self.embbedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.dropout = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.lstm = nn.LSTM(d_model, h_model, num_layers, bidirectional=False, batch_first=True, dropout=dropout)
        self.m = m
        self.tanh = nn.Tanh()
        self.time_layer = nn.Linear(1, 64)
        self.time_updim = nn.Linear(64, d_model)
        self.initial_embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.relu = nn.ReLU()
        self.Linear_generator = generator
        self.decoder = decoder_64_256(dense_model, d_model)
        self.encoder = encoder_256_64(d_model, dense_model)
        self.classifyer = classifyer(d_model)

    def before(self, input_seqs, seq_time_step):

        #time embedding
        seq_time_step = seq_time_step.unsqueeze(2) / 180
        time_feature = 1 - self.tanh(torch.pow(self.time_layer(seq_time_step), 2))
        time_encoding = self.time_updim(time_feature)
        #visit_embedding e_t
        visit_embedding = self.initial_embedding(input_seqs)
        visit_embedding = self.emb_dropout(visit_embedding)
        visit_embedding = self.relu(visit_embedding)

        visit_embedding = visit_embedding.sum(-2) + time_encoding
        return visit_embedding

    def forward(self, input_seqs, seq_time_step, label = None):
        batch_size, visit_size, seq_size = input_seqs.size()
        og_v = self.before(input_seqs, seq_time_step)
        og_h, _ = self.lstm(og_v)

        #encode og_h to dense:
        EncX = self.encoder(og_h)
        z = torch.randn_like(og_h)
        Gz = self.Linear_generator(z)

        DecX = self.decoder(EncX)
        DecGz = self.decoder(Gz)


        og_softmax = torch.zeros(batch_size, visit_size, 2).to(self.device)
        fake_softmax = torch.zeros(batch_size, visit_size, 2).to(self.device)

        for i in range(visit_size):

            og_softmax[:, i:i+1, :] = self.classifyer(og_h[:, i:i+1, :])
            fake_softmax[:, i:i+1, :] = self.classifyer(DecGz[:, i:i+1, :])

        final_prediction_og = og_softmax[:, -1, :]
        final_prediction_fake = fake_softmax[:, -1, :]

        f = final_prediction_og + final_prediction_fake

        return f, final_prediction_fake, _, DecX, DecGz, og_h

class LSTM_actGAN(nn.Module):
    def __init__(self, vocab_size, d_model, h_model, dropout, dropout_emb, device, generator, num_layers = 1, m = 0, dense_model = 64):
        super().__init__()
        self.device = device
        self.embbedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.dropout = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.lstm = nn.LSTM(d_model, h_model, num_layers, bidirectional=False, batch_first=True, dropout=dropout)
        self.m = m
        self.tanh = nn.Tanh()
        self.time_layer = nn.Linear(1, 64)
        self.time_updim = nn.Linear(64, d_model)
        self.initial_embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.relu = nn.ReLU()
        self.LSTM_generator = generator
        self.label_encoder = nn.Linear(2,256)
        self.xandy_encoder = nn.Linear(256,64)
        self.decoder = nn.Linear(64,256)
        self.classifyer = classifyer(d_model)

    def before(self, input_seqs, seq_time_step):

        #time embedding
        seq_time_step = seq_time_step.unsqueeze(2) / 180
        time_feature = 1 - self.tanh(torch.pow(self.time_layer(seq_time_step), 2))
        time_encoding = self.time_updim(time_feature)
        #visit_embedding e_t
        visit_embedding = self.initial_embedding(input_seqs)
        visit_embedding = self.emb_dropout(visit_embedding)
        visit_embedding = self.relu(visit_embedding)

        visit_embedding = visit_embedding.sum(-2) + time_encoding
        return visit_embedding

    def forward(self, input_seqs, seq_time_step, label):
        batch_size, visit_size, seq_size = input_seqs.size()
        og_v = self.before(input_seqs, seq_time_step)
        og_h, _ = self.lstm(og_v)


        ###generate fake visit from z and lstm generator
        z = torch.randn_like(og_v)
        mask = torch.ones_like(og_v)
        for i in range(batch_size):
            if label[i] == 0:
                mask[i, :, :] = 0
        fake_h = self.LSTM_generator(z) * mask
        positive_h = og_h * mask

        og_softmax = torch.zeros(batch_size, visit_size, 2).to(self.device)
        fake_softmax = torch.zeros(batch_size, visit_size, 2).to(self.device)

        for i in range(visit_size):

            og_softmax[:, i:i+1, :] = self.classifyer(og_h[:, i:i+1, :])
            fake_softmax[:, i:i+1, :] = self.classifyer(fake_h[:, i:i+1, :])

        final_prediction_og = og_softmax[:, -1, :]
        final_prediction_fake = fake_softmax[:, -1, :]

        f = final_prediction_og + final_prediction_fake

        return f, final_prediction_fake, _, positive_h, fake_h, _

class LSTM_GcGAN(nn.Module):
    def __init__(self, vocab_size, d_model, h_model, dropout, dropout_emb, device, generator, num_layers = 1, m = 0, dense_model = 64):
        super().__init__()
        self.device = device
        self.embbedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.dropout = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.lstm = nn.LSTM(d_model, h_model, num_layers, bidirectional=False, batch_first=True, dropout=dropout)
        self.m = m
        self.tanh = nn.Tanh()
        self.time_layer = nn.Linear(1, 64)
        self.time_updim = nn.Linear(64, d_model)
        self.initial_embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.relu = nn.ReLU()
        self.FCN_generator = generator
        self.label_encoder = nn.Linear(2,256)
        self.xandy_encoder = nn.Linear(256,64)
        self.decoder = nn.Linear(64,256)
        self.classifyer = classifyer(d_model)

    def before(self, input_seqs, seq_time_step):

        #time embedding
        seq_time_step = seq_time_step.unsqueeze(2) / 180
        time_feature = 1 - self.tanh(torch.pow(self.time_layer(seq_time_step), 2))
        time_encoding = self.time_updim(time_feature)
        #visit_embedding e_t
        visit_embedding = self.initial_embedding(input_seqs)
        visit_embedding = self.emb_dropout(visit_embedding)
        visit_embedding = self.relu(visit_embedding)

        visit_embedding = visit_embedding.sum(-2) + time_encoding
        return visit_embedding

    def forward(self, input_seqs, seq_time_step, label):
        batch_size, visit_size, seq_size = input_seqs.size()
        og_v = self.before(input_seqs, seq_time_step)
        og_h, _ = self.lstm(og_v)
        ####encoder
        temp = self.label_encoder(label)
        temp = temp.unsqueeze(1)
        x_and_y = torch.cat([og_h, temp], dim=-2)
        h = self.relu(self.xandy_encoder(x_and_y))
        z = torch.randn_like(x_and_y)
        gen_h = self.relu(self.FCN_generator(z))

        ####decoder
        gen_D1andD2 = self.relu(self.decoder(gen_h))
        gen_D1 = gen_D1andD2[:, :-1, :]

        D1andD2 = self.relu(self.decoder(h))
        D1 = D1andD2[:, :-1, :]

        og_softmax = torch.zeros(batch_size, visit_size, 2).to(self.device)
        fake_softmax = torch.zeros(batch_size, visit_size, 2).to(self.device)
        decode_softmax = torch.zeros(batch_size, visit_size, 2).to(self.device)

        for i in range(visit_size):

            og_softmax[:, i:i+1, :] = self.classifyer(og_h[:, i:i+1, :])
            fake_softmax[:, i:i+1, :] = self.classifyer(gen_D1[:, i:i+1, :])
            decode_softmax[:, i:i+1, :] = self.classifyer(D1[:, i:i+1, :])

        final_prediction_og = og_softmax[:, -1, :]
        final_prediction_fake = fake_softmax[:, -1, :]
        final_prediction_decode = decode_softmax[:, -1, :]

        f = final_prediction_og + final_prediction_fake

        return f, final_prediction_fake, final_prediction_decode, og_h, gen_D1, D1

class LSTM_ehrGAN(nn.Module):
    def __init__(self, vocab_size, d_model, h_model, dropout, dropout_emb, device, generator, num_layers=1, m=0,
                 dense_model=64):
        super().__init__()
        self.device = device
        self.embbedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.dropout = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.lstm = nn.LSTM(d_model, h_model, num_layers, bidirectional=False, batch_first=True, dropout=dropout)
        self.m = m
        self.tanh = nn.Tanh()
        self.time_layer = nn.Linear(1, 64)
        self.time_updim = nn.Linear(64, d_model)
        self.initial_embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.relu = nn.ReLU()
        self.lstm_generator = generator
        self.dense_model = dense_model
        self.dense = nn.Linear(d_model, dense_model)
        self.classifyer = classifyer(d_model)

    def before(self, input_seqs, seq_time_step):
        # time embedding
        seq_time_step = seq_time_step.unsqueeze(2) / 180
        time_feature = 1 - self.tanh(torch.pow(self.time_layer(seq_time_step), 2))
        time_encoding = self.time_updim(time_feature)
        # visit_embedding e_t
        visit_embedding = self.initial_embedding(input_seqs)
        visit_embedding = self.emb_dropout(visit_embedding)
        visit_embedding = self.relu(visit_embedding)

        visit_embedding = visit_embedding.sum(-2) + time_encoding
        return visit_embedding

    def forward(self, input_seqs, seq_time_step, labels = None):
        batch_size, visit_size, seq_size = input_seqs.size()
        og_v = self.before(input_seqs, seq_time_step)
        og_h, _ = self.lstm(og_v)
        x = self.dense(og_h)
        z = torch.randn_like(x)
        mask = torch.randn_like(z) < self.m
        tilde_x = x * (~mask) + mask * z
        fake_h = self.lstm_generator(tilde_x)
        decode_h = self.lstm_generator(x)

        og_softmax = torch.zeros(batch_size, visit_size, 2).to(self.device)
        fake_softmax = torch.zeros(batch_size, visit_size, 2).to(self.device)
        decode_softmax = torch.zeros(batch_size, visit_size, 2).to(self.device)

        for i in range(visit_size):
            og_softmax[:, i:i + 1, :] = self.classifyer(og_h[:, i:i + 1, :])
            fake_softmax[:, i:i + 1, :] = self.classifyer(fake_h[:, i:i + 1, :])
            decode_softmax[:, i:i + 1, :] = self.classifyer(decode_h[:, i:i + 1, :])

        final_prediction_og = og_softmax[:, -1, :]
        final_prediction_fake = fake_softmax[:, -1, :]
        final_prediction_decode = decode_softmax[:, -1, :]

        f = final_prediction_og + final_prediction_fake

        return f, final_prediction_fake, final_prediction_decode, og_h, fake_h, decode_h

if __name__ == '__main__':
    y_true = np.array([])
    print(len(y_true))