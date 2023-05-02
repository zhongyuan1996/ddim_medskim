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

    def __init__(self, h_model=64, max_len=50):
        super(Linear_generator, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(max_len)
        self.batch_norm2 = nn.BatchNorm1d(max_len)
        self.linear1 = nn.Linear(h_model, h_model)
        self.linear2 = nn.Linear(h_model, h_model)
        self.relu = nn.ReLU()

    def forward(self, z):

        z1 = self.relu(self.batch_norm1(self.linear1(z))) + z
        z2 = self.relu(self.batch_norm2(self.linear2(z1))) + z1
        return z2


class encoder_256_64(nn.Module):
    def __init__(self, in_dim=256, out_dim=64):
        super(encoder_256_64, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.batch_norm = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.batch_norm(self.linear(x)))


class decoder_64_256(nn.Module):
    def __init__(self, in_dim=64, out_dim=256):
        super(decoder_64_256, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.batch_norm = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.batch_norm(self.linear(x)))


class LSTM_generator(nn.Module):
    def __init__(self, d_model, h_model, num_layers=1):
        super(LSTM_generator, self).__init__()
        self.lstm = nn.LSTM(h_model, d_model, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x


class FCN_generator(nn.Module):

    def __init__(self, h_model, max_len=50):
        super(FCN_generator, self).__init__()
        self.lenPlus1 = max_len + 1
        self.linear1 = nn.Linear(h_model, h_model)
        self.layer2_conv = nn.Conv1d(self.lenPlus1 * 2, self.lenPlus1, 1, 1)
        self.layer3 = nn.Linear(h_model, h_model)
        self.layer4_conv = nn.Conv1d(self.lenPlus1 * 3, self.lenPlus1, 1, 1)
        self.ff = nn.Linear(h_model, h_model)

    def forward(self, gray):
        orange = self.linear1(gray)
        brown = self.layer2_conv(torch.cat([gray, orange], dim=1))
        blue = self.layer3(brown)
        red = self.layer4_conv(torch.cat([blue, brown, gray], dim=1))
        res = self.ff(red)
        return res


class Discriminator(nn.Module):
    def __init__(self, vocab):
        super(Discriminator, self).__init__()
        self.dense = nn.Linear(vocab, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.dense(x))


class medGAN(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, device, generator, m=0):
        super().__init__()
        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout)
        self.encoding1 = nn.Linear(d_model, int(d_model / 2))
        self.encoding2 = nn.Linear(int(d_model / 2), int(d_model / 4))
        self.decoding1 = nn.Linear(int(d_model / 2), d_model)
        self.decoding2 = nn.Linear(int(d_model / 4), int(d_model / 2))
        self.decode_to_vocab = nn.Linear(d_model, vocab_size + 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.Linear_generator = generator
        self.classifyer = classifyer(d_model)

    def medGAN_encoder(self, x):
        x = self.encoding1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.encoding2(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x

    def medGAN_decoder(self, x):
        x = self.decoding2(x)
        x = self.relu(x)
        x = self.decoding1(x)
        return x

    def forward(self, input_seqs, time_seqs, label):
        batch_size, visit_size, seq_size = input_seqs.size()
        x = self.medGAN_encoder(input_seqs)

        z = torch.randn_like(x)
        Gz = self.Linear_generator(z)

        DecX = self.medGAN_decoder(x)
        DecGz = self.medGAN_decoder(Gz)

        return DecX, DecGz, time_seqs, label


class actGAN(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, device, generator, m=0):
        super().__init__()
        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.encoding1 = nn.Linear(d_model, int(d_model / 2))
        self.encoding2 = nn.Linear(int(d_model / 2), int(d_model / 4))
        self.decoding1 = nn.Linear(int(d_model / 2), d_model)
        self.decoding2 = nn.Linear(int(d_model / 4), int(d_model / 2))
        self.decode_to_vocab = nn.Linear(d_model, vocab_size + 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.Linear_generator = generator
        self.classifyer = classifyer(d_model)

    def actGAN_encoder(self, x):
        x = self.encoding1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.encoding2(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x

    def actGAN_decoder(self, x):
        x = self.decoding2(x)
        x = self.relu(x)
        x = self.decoding1(x)
        return x

    def forward(self, input_seqs, time_seqs, label):
        batch_size, visit_size, seq_size = input_seqs.size()
        x = self.actGAN_encoder(input_seqs)

        z = torch.randn_like(x)
        mask = torch.ones_like(x)
        for i in range(batch_size):
            if label[i] == 0:
                mask[i, :, :] = 0
        fake_h = self.Linear_generator(z)
        positive_h = x

        fake_x = self.actGAN_decoder(fake_h)
        positive_x = self.actGAN_decoder(positive_h)

        return positive_x, fake_x, time_seqs, label

class GcGAN(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, device, generator, m=0):
        super().__init__()
        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.m = m
        self.encoding1 = nn.Linear(d_model, int(d_model / 2))
        self.encoding2 = nn.Linear(int(d_model / 2), int(d_model / 4))
        self.decoding1 = nn.Linear(int(d_model / 2), d_model)
        self.decoding2 = nn.Linear(int(d_model / 4), int(d_model / 2))
        self.relu = nn.ReLU()
        self.FCN_generator = generator
        self.label_encoder = nn.Embedding(2, 256)
        self.label_decoder = nn.Linear(256, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.classifyer = classifyer(d_model)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def GcGAN_encoder(self, x):
        x = self.encoding1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.encoding2(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x

    def GcGAN_decoder(self, x):
        x = self.decoding2(x)
        x = self.relu(x)
        x = self.decoding1(x)
        return x

    def forward(self, input_seqs, time_seqs, label):
        x = input_seqs
        y = self.label_encoder(label.unsqueeze(1))
        XY = torch.cat([x, y], dim=-2)
        h = self.GcGAN_encoder(XY)
        z = torch.randn_like(h)
        gen_h = self.FCN_generator(z)

        ####decoder
        gen_D1andD2 = self.GcGAN_decoder(gen_h)

        gen_D1 = gen_D1andD2[:, :-1, :]

        gen_D2 = gen_D1andD2[:, -1, :].unsqueeze(1)
        gen_D2 = self.sigmoid(self.label_decoder(gen_D2)).squeeze()

        D1andD2 = self.GcGAN_decoder(h)
        D1 = D1andD2[:, :-1, :]

        return D1, gen_D1, time_seqs, gen_D2

class ehrGAN(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, device, generator, m=0):
        super().__init__()
        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout)
        self.m = m
        self.encoding1 = nn.Linear(d_model, int(d_model / 2))
        self.encoding2 = nn.Linear(int(d_model / 2), int(d_model / 4))
        self.decoding1 = nn.Linear(int(d_model / 2), d_model)
        self.decoding2 = nn.Linear(int(d_model / 4), int(d_model / 2))
        self.decode_to_vocab = nn.Linear(d_model, vocab_size + 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.Linear_generator = generator
        self.classifyer = classifyer(d_model)

    def ehrGAN_encoder(self, x):
        x = self.encoding1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.encoding2(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x

    def ehrGAN_decoder(self, x):
        x = self.decoding2(x)
        x = self.relu(x)
        x = self.decoding1(x)
        return x

    def forward(self, input_seqs, time_seqs, labels):
        batch_size, visit_size, seq_size = input_seqs.size()
        #32 50 256
        h = self.ehrGAN_encoder(input_seqs) #32 50 64

        z = torch.randn_like(h)
        mask = z < self.m
        tilde_h = h * (~mask) + mask * z
        gen_h = self.Linear_generator(tilde_h)
        fake_x = self.ehrGAN_decoder(gen_h)
        decode_x = self.ehrGAN_decoder(h)

        return decode_x, fake_x, time_seqs, labels

class LSTM_base(nn.Module):
    def __init__(self, vocab_size, d_model, h_model, dropout, dropout_emb, device, GAN=None, num_layers=1, m=0,
                 dense_model=64):
        super().__init__()
        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.lstm = nn.LSTM(d_model, h_model, num_layers, bidirectional=False, batch_first=True, dropout=dropout)
        self.m = m
        self.tanh = nn.Tanh()
        self.time_layer = nn.Linear(1, 64)
        self.time_updim = nn.Linear(64, d_model)
        self.initial_embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.relu = nn.ReLU()
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

    def forward(self, input_seqs, seq_time_step, label):
        batch_size, visit_size, seq_size = input_seqs.size()
        v = self.before(input_seqs, seq_time_step)
        h, _ = self.lstm(v)

        og_softmax = torch.zeros(batch_size, visit_size, 2).to(self.device)

        for i in range(visit_size):
            og_softmax[:, i:i + 1, :] = self.classifyer(h[:, i:i + 1, :])

        final_prediction_og = og_softmax[:, -1, :]

        return final_prediction_og, _, _, seq_time_step, label

class LSTM_medGAN(nn.Module):
    def __init__(self, vocab_size, d_model, h_model, dropout, dropout_emb, device, GAN, num_layers=1, m=0,
                 dense_model=64):
        super().__init__()
        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.lstm = nn.LSTM(d_model, h_model, num_layers, bidirectional=False, batch_first=True, dropout=dropout)
        self.m = m
        self.tanh = nn.Tanh()
        self.time_layer = nn.Linear(1, 64)
        self.time_updim = nn.Linear(64, d_model)
        self.initial_embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.relu = nn.ReLU()
        self.GAN = GAN
        self.fuse = nn.Linear(d_model * 2, d_model)
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

    def forward(self, input_seqs, mask, length, seq_time_step, codemask, label):
        batch_size, visit_size, seq_size = input_seqs.size()
        v = self.before(input_seqs, seq_time_step)

        Dec_V, Gen_V, _, _ = self.GAN(v, seq_time_step, label)

        Fused_V = self.fuse(torch.cat([v, Gen_V], dim=-1))

        h, _ = self.lstm(Fused_V)

        og_softmax = torch.zeros(batch_size, visit_size, 2).to(self.device)

        for i in range(visit_size):
            og_softmax[:, i:i + 1, :] = self.classifyer(h[:, i:i + 1, :])

        final_prediction_og = og_softmax[:, -1, :]

        return final_prediction_og, Dec_V, Gen_V, seq_time_step, label

class LSTM_actGAN(nn.Module):
    def __init__(self, vocab_size, d_model, h_model, dropout, dropout_emb, device, GAN, num_layers=1, m=0,
                 dense_model=64):
        super().__init__()
        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.lstm = nn.LSTM(d_model, h_model, num_layers, bidirectional=False, batch_first=True, dropout=dropout)
        self.m = m
        self.tanh = nn.Tanh()
        self.time_layer = nn.Linear(1, 64)
        self.time_updim = nn.Linear(64, d_model)
        self.initial_embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.relu = nn.ReLU()
        self.GAN = GAN
        self.label_encoder = nn.Linear(2, 256)
        self.xandy_encoder = nn.Linear(256, 64)
        self.decoder = nn.Linear(64, 256)
        self.classifyer = classifyer(d_model)
        self.fuse = nn.Linear(d_model * 2, d_model)

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

    def forward(self, input_seqs, mask, length, seq_time_step, codemask, label):
        batch_size, visit_size, seq_size = input_seqs.size()
        v = self.before(input_seqs, seq_time_step)

        Dec_V, Gen_V, _, _ = self.GAN(v, seq_time_step, label)

        Fused_V = self.fuse(torch.cat([v, Gen_V], dim=-1))

        h, _ = self.lstm(Fused_V)

        og_softmax = torch.zeros(batch_size, visit_size, 2).to(self.device)

        for i in range(visit_size):
            og_softmax[:, i:i + 1, :] = self.classifyer(h[:, i:i + 1, :])

        final_prediction_og = og_softmax[:, -1, :]

        return final_prediction_og, Dec_V, Gen_V, seq_time_step, label

class LSTM_GcGAN(nn.Module):
    def __init__(self, vocab_size, d_model, h_model, dropout, dropout_emb, device, GAN, num_layers=1, m=0,
                 dense_model=64):
        super().__init__()
        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.lstm = nn.LSTM(d_model, h_model, num_layers, bidirectional=False, batch_first=True, dropout=dropout)
        self.m = m
        self.tanh = nn.Tanh()
        self.time_layer = nn.Linear(1, 64)
        self.time_updim = nn.Linear(64, d_model)
        self.initial_embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.relu = nn.ReLU()
        self.GAN = GAN
        self.classifyer = classifyer(d_model)
        self.fuse = nn.Linear(d_model * 2, d_model)

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

    def forward(self, input_seqs, mask, length, seq_time_step, codemask, label):
        batch_size, visit_size, seq_size = input_seqs.size()
        v = self.before(input_seqs, seq_time_step)

        Dec_V, Gen_V, _, Gen_label = self.GAN(v, seq_time_step, label)

        Fused_V = self.fuse(torch.cat([v, Gen_V], dim=-1))

        h, _ = self.lstm(Fused_V)

        og_softmax = torch.zeros(batch_size, visit_size, 2).to(self.device)

        for i in range(visit_size):
            og_softmax[:, i:i + 1, :] = self.classifyer(h[:, i:i + 1, :])

        final_prediction_og = og_softmax[:, -1, :]

        return final_prediction_og, Dec_V, Gen_V, seq_time_step, Gen_label

class LSTM_ehrGAN(nn.Module):
    def __init__(self, vocab_size, d_model, h_model, dropout, dropout_emb, device, GAN, num_layers=1, m=0,
                 dense_model=64):
        super().__init__()
        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.lstm = nn.LSTM(d_model, h_model, num_layers, bidirectional=False, batch_first=True, dropout=dropout)
        self.m = m
        self.tanh = nn.Tanh()
        self.time_layer = nn.Linear(1, 64)
        self.time_updim = nn.Linear(64, d_model)
        self.initial_embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.relu = nn.ReLU()
        self.GAN = GAN
        self.classifyer = classifyer(d_model)
        self.fuse = nn.Linear(d_model * 2, d_model)

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

    def forward(self, input_seqs, mask, length, seq_time_step, codemask, label):
        batch_size, visit_size, seq_size = input_seqs.size()
        v = self.before(input_seqs, seq_time_step)

        Dec_V, Gen_V, _, _ = self.GAN(v, seq_time_step, label)

        Fused_V = self.fuse(torch.cat([v, Gen_V], dim=-1))

        h, _ = self.lstm(Fused_V)

        og_softmax = torch.zeros(batch_size, visit_size, 2).to(self.device)

        for i in range(visit_size):
            og_softmax[:, i:i + 1, :] = self.classifyer(h[:, i:i + 1, :])

        final_prediction_og = og_softmax[:, -1, :]

        return final_prediction_og, Dec_V, Gen_V, seq_time_step, label


if __name__ == '__main__':
    y_true = np.array([])
    print(len(y_true))
