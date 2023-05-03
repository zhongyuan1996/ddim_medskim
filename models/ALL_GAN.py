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
    def __init__(self, vocab_size, d_model, h_model, dropout, dropout_emb, device, GAN, initial_d = None, num_layers=1, m=0,
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
        self.initial_d = initial_d
        self.initial_embedding_2 = nn.Linear(self.initial_d, d_model)
        self.relu = nn.ReLU()
        self.classifyer = classifyer(d_model)

    def before(self, input_seqs, seq_time_step):
        # time embedding
        seq_time_step = seq_time_step.unsqueeze(2) / 180
        time_feature = 1 - self.tanh(torch.pow(self.time_layer(seq_time_step), 2))
        time_encoding = self.time_updim(time_feature)
        # visit_embedding e_t
        if not self.initial_d:
            visit_embedding = self.initial_embedding(input_seqs)
        else:
            visit_embedding = self.initial_embedding_2(input_seqs)
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

        return final_prediction_og, _, _, _, seq_time_step, label

class LSTM_medGAN(nn.Module):
    def __init__(self, vocab_size, d_model, h_model, dropout, dropout_emb, device, GAN, initial_d = 0, num_layers=1, m=0,
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
        self.initial_d = initial_d
        self.initial_embedding_2 = nn.Linear(self.initial_d, d_model)
        self.relu = nn.ReLU()
        self.classifyer = classifyer(d_model)

    def before(self, input_seqs, seq_time_step):
        # time embedding
        seq_time_step = seq_time_step.unsqueeze(2) / 180
        time_feature = 1 - self.tanh(torch.pow(self.time_layer(seq_time_step), 2))
        time_encoding = self.time_updim(time_feature)
        # visit_embedding e_t
        if self.initial_d != 0:
            visit_embedding = self.initial_embedding(input_seqs)
        else:
            visit_embedding = self.initial_embedding_2(input_seqs)
        visit_embedding = self.emb_dropout(visit_embedding)
        visit_embedding = self.relu(visit_embedding)

        visit_embedding = visit_embedding.sum(-2) + time_encoding
        return visit_embedding

    def forward(self, input_seqs, mask, length, seq_time_step, codemask, label):
        batch_size, visit_size, seq_size = input_seqs.size()
        v = self.before(input_seqs, seq_time_step)

        Dec_V, Gen_V, _, _ = self.GAN(v, seq_time_step, label)

        # Fused_V = self.fuse(torch.cat([v, Gen_V], dim=-1))

        h, _ = self.lstm(v)
        gen_h, _ = self.lstm(Gen_V)

        og_softmax = torch.zeros(batch_size, visit_size, 2).to(self.device)
        fake_softmax = torch.zeros(batch_size, visit_size, 2).to(self.device)

        for i in range(visit_size):
            og_softmax[:, i:i + 1, :] = self.classifyer(h[:, i:i + 1, :])
            fake_softmax[:, i:i + 1, :] = self.classifyer(gen_h[:, i:i + 1, :])

        final_prediction_og = og_softmax[:, -1, :]

        return final_prediction_og, fake_softmax, Dec_V, Gen_V, seq_time_step, label

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

        # Fused_V = self.fuse(torch.cat([v, Gen_V], dim=-1))

        h, _ = self.lstm(v)
        gen_h, _ = self.lstm(Gen_V)

        og_softmax = torch.zeros(batch_size, visit_size, 2).to(self.device)
        fake_softmax = torch.zeros(batch_size, visit_size, 2).to(self.device)

        for i in range(visit_size):
            og_softmax[:, i:i + 1, :] = self.classifyer(h[:, i:i + 1, :])
            fake_softmax[:, i:i + 1, :] = self.classifyer(gen_h[:, i:i + 1, :])

        final_prediction_og = og_softmax[:, -1, :]

        return final_prediction_og, fake_softmax, Dec_V, Gen_V, seq_time_step, label

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

        Dec_V, Gen_V, _, _ = self.GAN(v, seq_time_step, label)

        # Fused_V = self.fuse(torch.cat([v, Gen_V], dim=-1))

        h, _ = self.lstm(v)
        gen_h, _ = self.lstm(Gen_V)

        og_softmax = torch.zeros(batch_size, visit_size, 2).to(self.device)
        fake_softmax = torch.zeros(batch_size, visit_size, 2).to(self.device)

        for i in range(visit_size):
            og_softmax[:, i:i + 1, :] = self.classifyer(h[:, i:i + 1, :])
            fake_softmax[:, i:i + 1, :] = self.classifyer(gen_h[:, i:i + 1, :])

        final_prediction_og = og_softmax[:, -1, :]

        return final_prediction_og, fake_softmax, Dec_V, Gen_V, seq_time_step, label

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
        # self.fuse = nn.Linear(d_model * 2, d_model)

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

        # Fused_V = self.fuse(torch.cat([v, Gen_V], dim=-1))

        h, _ = self.lstm(v)
        gen_h, _ = self.lstm(Gen_V)

        og_softmax = torch.zeros(batch_size, visit_size, 2).to(self.device)
        fake_softmax = torch.zeros(batch_size, visit_size, 2).to(self.device)

        for i in range(visit_size):
            og_softmax[:, i:i + 1, :] = self.classifyer(h[:, i:i + 1, :])
            fake_softmax[:, i:i + 1, :] = self.classifyer(gen_h[:, i:i + 1, :])

        final_prediction_og = og_softmax[:, -1, :]

        return final_prediction_og, fake_softmax, Dec_V, Gen_V, seq_time_step, label

class Dipole_base(nn.Module):
    def __init__(self, vocab_size, d_model, h_model, dropout, dropout_emb, device, GAN, num_layers=1, m=0,
                 dense_model=64):
        super().__init__()
        self.embbedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.dropout = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.output_mlp = nn.Sequential(nn.Linear(d_model, 2))
        self.gru = nn.GRU(d_model, d_model, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.weight_layer = nn.Linear(d_model, 1)
        self.tanh = nn.Tanh()
        self.time_layer = nn.Linear(1, 64)
        self.time_updim = nn.Linear(64, d_model)
        self.initial_embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.GAN = GAN
        self.relu=nn.ReLU()


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

    def forward(self, input_seqs, mask, lengths, seq_time_step, codemask, label):
        batch_size, seq_len, num_cui_per_visit = input_seqs.size()
        x = self.before(input_seqs, seq_time_step)
        x = self.emb_dropout(x)

        Dec_X, Gen_X, _, _ = self.GAN(x, seq_time_step, label)

        rnn_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        rnn_output, _ = self.gru(rnn_input)
        rnn_output, _ = pad_packed_sequence(rnn_output, batch_first=True, total_length=seq_len)

        gen_rnn_input = pack_padded_sequence(Gen_X, lengths.cpu(), batch_first=True, enforce_sorted=False)
        gen_rnn_output, _ = self.gru(gen_rnn_input)
        gen_rnn_output, _ = pad_packed_sequence(gen_rnn_output, batch_first=True, total_length=seq_len)

        weight = self.weight_layer(rnn_output)
        mask = (torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len) >= lengths.unsqueeze(1))
        att = torch.softmax(weight.squeeze().masked_fill(mask, -np.inf), dim=1).view(batch_size, seq_len)
        weighted_features = rnn_output * att.unsqueeze(2)
        averaged_features = torch.sum(weighted_features, 1)
        averaged_features = self.dropout(averaged_features)
        pred = self.output_mlp(averaged_features)

        gen_weight = self.weight_layer(gen_rnn_output)
        gen_mask = (torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len) >= lengths.unsqueeze(1))
        gen_att = torch.softmax(gen_weight.squeeze().masked_fill(gen_mask, -np.inf), dim=1).view(batch_size, seq_len)
        gen_weighted_features = gen_rnn_output * gen_att.unsqueeze(2)
        gen_averaged_features = torch.sum(gen_weighted_features, 1)
        gen_averaged_features = self.dropout(gen_averaged_features)
        gen_pred = self.output_mlp(gen_averaged_features)

        return pred, gen_pred, Dec_X, Gen_X, seq_time_step, label

class TLSTM_base(nn.Module):

    def __init__(self, vocab_size, d_model, h_model, dropout, dropout_emb, device, GAN, num_layers=1, m=0,
                 dense_model=64):
        super(TLSTM_base, self).__init__()

        self.embbedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.W_all = nn.Linear(d_model, d_model * 4)
        self.U_all = nn.Linear(d_model, d_model * 4)
        self.W_d = nn.Linear(d_model, d_model)
        self.d_model = d_model
        self.time_layer = torch.nn.Linear(64, d_model)
        self.selection_layer = torch.nn.Linear(1, 64)
        self.output_mlp = nn.Sequential(nn.Linear(d_model, 2))
        self.pooler = MaxPoolLayer()
        self.GAN = GAN

    def forward(self, input_seqs, mask, lengths, seq_time_step, codemask, label):
        x = self.embbedding(input_seqs).sum(dim=2)

        Dec_X, Gen_X, _, _ = self.GAN(x, seq_time_step, label)

        b, seq, embed = x.size()
        h = torch.zeros(b, self.d_model, requires_grad=False).to(x.device)
        c = torch.zeros(b, self.d_model, requires_grad=False).to(x.device)
        outputs = []
        seq_time_step = seq_time_step.unsqueeze(2) / 180
        time_feature = 1 - torch.tanh(torch.pow(self.selection_layer(seq_time_step), 2))
        time_feature = self.time_layer(time_feature)
        for s in range(seq):
            c_s1 = torch.tanh(self.W_d(c))
            c_s2 = c_s1 * time_feature[:, s]
            c_l = c - c_s1
            c_adj = c_l + c_s2
            outs = self.W_all(h) + self.U_all(x[:, s])
            # print(outs.size())
            f, i, o, c_tmp = torch.chunk(outs, 4, 1)
            f = torch.sigmoid(f)
            i = torch.sigmoid(i)
            o = torch.sigmoid(o)
            c_tmp = torch.sigmoid(c_tmp)
            c = f * c_adj + i * c_tmp
            h = o * torch.tanh(c)
            outputs.append(h)
        outputs = torch.stack(outputs, 1)

        out = self.pooler(outputs, lengths)
        out = self.output_mlp(out)

        gen_h = torch.zeros(b,self.d_model, requires_grad=False).to(x.device)
        gen_c = torch.zeros(b,self.d_model, requires_grad=False).to(x.device)
        gen_outputs = []
        for s in range(seq):
            gen_c_s1 = torch.tanh(self.W_d(gen_c))
            gen_c_s2 = gen_c_s1 * time_feature[:, s]
            gen_c_l = gen_c - gen_c_s1
            gen_c_adj = gen_c_l + gen_c_s2
            gen_outs = self.W_all(gen_h) + self.U_all(Gen_X[:, s])
            # print(outs.size())
            gen_f, gen_i, gen_o, gen_c_tmp = torch.chunk(gen_outs, 4, 1)
            gen_f = torch.sigmoid(gen_f)
            gen_i = torch.sigmoid(gen_i)
            gen_o = torch.sigmoid(gen_o)
            gen_c_tmp = torch.sigmoid(gen_c_tmp)
            gen_c = gen_f * gen_c_adj + gen_i * gen_c_tmp
            gen_h = gen_o * torch.tanh(gen_c)
            gen_outputs.append(gen_h)
        gen_outputs = torch.stack(gen_outputs, 1)
        gen_out = self.pooler(gen_outputs, lengths)
        gen_out = self.output_mlp(gen_out)

        return out, gen_out, Dec_X, Gen_X, seq_time_step, label

class SAND_base(nn.Module):

    def __init__(self, vocab_size, d_model, h_model, dropout, dropout_emb, device, GAN, num_heads, max_pos, num_layers=1, m=0,
                 dense_model=64):

        super().__init__()
        self.embbedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.pos_emb = PositionalEncoding(d_model, max_pos)
        self.encoder_layer = Attention(d_model, num_heads, dropout)
        self.positional_feed_forward_layer = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.ReLU(),
                                                           nn.Linear(4 * d_model, d_model))
        self.bias_embedding = torch.nn.Parameter(torch.Tensor(d_model))
        bound = 1 / math.sqrt(vocab_size)
        init.uniform_(self.bias_embedding, -bound, bound)
        # self.weight_layer = torch.nn.Linear(d_model, 1)
        self.drop_out = nn.Dropout(dropout)
        self.out_layer = nn.Linear(d_model * 4, 2)
        self.layer_norm = nn.LayerNorm(d_model)
        self.GAN = GAN


    def forward(self, input_seqs, masks, lengths, seq_time_step, codemask, label):
        x = self.embbedding(input_seqs).sum(dim=2) + self.bias_embedding
        bs, sl, dm = x.size()
        x = self.emb_dropout(x)
        output_pos, ind_pos = self.pos_emb(lengths)
        x += output_pos

        Dec_X, Gen_X, _, _ = self.GAN(x, seq_time_step, label)

        x, attention = self.encoder_layer(x, x, x, masks)
        mask = (torch.arange(sl, device=x.device).unsqueeze(0).expand(bs, sl) >= lengths.unsqueeze(
            1))
        x = x.masked_fill(mask.unsqueeze(-1).expand_as(x), 0.0)
        U = torch.zeros((x.size(0), 4, x.size(2))).to(x.device)
        lengths = lengths.float()
        for t in range(1, input_seqs.size(1) + 1):
            s = 4 * t / lengths
            for m in range(1, 4 + 1):
                w = torch.pow(1 - torch.abs(s - m) / 4, 2)
                U[:, m - 1] += w.unsqueeze(-1) * x[:, t - 1]
        U = U.view(input_seqs.size(0), -1)
        U = self.drop_out(U)
        output = self.out_layer(U)

        gen_x, gen_attention = self.encoder_layer(Gen_X, Gen_X, Gen_X, masks)
        gen_mask = (torch.arange(sl, device=gen_x.device).unsqueeze(0).expand(bs, sl) >= lengths.unsqueeze(
            1))
        gen_x = gen_x.masked_fill(gen_mask.unsqueeze(-1).expand_as(gen_x), 0.0)
        gen_U = torch.zeros((gen_x.size(0), 4, gen_x.size(2))).to(gen_x.device)
        lengths = lengths.float()
        for t in range(1, input_seqs.size(1) + 1):
            s = 4 * t / lengths
            for m in range(1, 4 + 1):
                w = torch.pow(1 - torch.abs(s - m) / 4, 2)
                gen_U[:, m - 1] += w.unsqueeze(-1) * gen_x[:, t - 1]
        gen_U = gen_U.view(input_seqs.size(0), -1)
        gen_U = self.drop_out(gen_U)
        gen_output = self.out_layer(gen_U)

        return output, gen_output, Dec_X, Gen_X, seq_time_step, label
# class LSTM_medGAN(nn.Module):
#     def __init__(self, vocab_size, d_model, h_model, dropout, dropout_emb, device, GAN, num_layers=1, m=0,
#                  dense_model=64):
#         super().__init__()
#         self.device = device
#         self.dropout = nn.Dropout(dropout)
#         self.emb_dropout = nn.Dropout(dropout_emb)
#         self.lstm = nn.LSTM(d_model, h_model, num_layers, bidirectional=False, batch_first=True, dropout=dropout)
#         self.m = m
#         self.tanh = nn.Tanh()
#         self.time_layer = nn.Linear(1, 64)
#         self.time_updim = nn.Linear(64, d_model)
#         self.initial_embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
#         self.relu = nn.ReLU()
#         self.GAN = GAN
#         self.fuse = nn.Linear(d_model * 2, d_model)
#         self.classifyer = classifyer(d_model)
#
#     def before(self, input_seqs, seq_time_step):
#         # time embedding
#         seq_time_step = seq_time_step.unsqueeze(2) / 180
#         time_feature = 1 - self.tanh(torch.pow(self.time_layer(seq_time_step), 2))
#         time_encoding = self.time_updim(time_feature)
#         # visit_embedding e_t
#         visit_embedding = self.initial_embedding(input_seqs)
#         visit_embedding = self.emb_dropout(visit_embedding)
#         visit_embedding = self.relu(visit_embedding)
#
#         visit_embedding = visit_embedding.sum(-2) + time_encoding
#         return visit_embedding
#
#     def forward(self, input_seqs, mask, length, seq_time_step, codemask, label):
#         batch_size, visit_size, seq_size = input_seqs.size()
#         v = self.before(input_seqs, seq_time_step)
#
#         Dec_V, Gen_V, _, _ = self.GAN(v, seq_time_step, label)
#
#         Fused_V = self.fuse(torch.cat([v, Gen_V], dim=-1))
#
#         h, _ = self.lstm(Fused_V)
#
#         og_softmax = torch.zeros(batch_size, visit_size, 2).to(self.device)
#
#         for i in range(visit_size):
#             og_softmax[:, i:i + 1, :] = self.classifyer(h[:, i:i + 1, :])
#
#         final_prediction_og = og_softmax[:, -1, :]
#
#         return final_prediction_og, Dec_V, Gen_V, seq_time_step, label
#
# class LSTM_actGAN(nn.Module):
#     def __init__(self, vocab_size, d_model, h_model, dropout, dropout_emb, device, GAN, num_layers=1, m=0,
#                  dense_model=64):
#         super().__init__()
#         self.device = device
#         self.dropout = nn.Dropout(dropout)
#         self.emb_dropout = nn.Dropout(dropout_emb)
#         self.lstm = nn.LSTM(d_model, h_model, num_layers, bidirectional=False, batch_first=True, dropout=dropout)
#         self.m = m
#         self.tanh = nn.Tanh()
#         self.time_layer = nn.Linear(1, 64)
#         self.time_updim = nn.Linear(64, d_model)
#         self.initial_embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
#         self.relu = nn.ReLU()
#         self.GAN = GAN
#         self.label_encoder = nn.Linear(2, 256)
#         self.xandy_encoder = nn.Linear(256, 64)
#         self.decoder = nn.Linear(64, 256)
#         self.classifyer = classifyer(d_model)
#         self.fuse = nn.Linear(d_model * 2, d_model)
#
#     def before(self, input_seqs, seq_time_step):
#
#         # time embedding
#         seq_time_step = seq_time_step.unsqueeze(2) / 180
#         time_feature = 1 - self.tanh(torch.pow(self.time_layer(seq_time_step), 2))
#         time_encoding = self.time_updim(time_feature)
#         # visit_embedding e_t
#         visit_embedding = self.initial_embedding(input_seqs)
#         visit_embedding = self.emb_dropout(visit_embedding)
#         visit_embedding = self.relu(visit_embedding)
#
#         visit_embedding = visit_embedding.sum(-2) + time_encoding
#         return visit_embedding
#
#     def forward(self, input_seqs, mask, length, seq_time_step, codemask, label):
#         batch_size, visit_size, seq_size = input_seqs.size()
#         v = self.before(input_seqs, seq_time_step)
#
#         Dec_V, Gen_V, _, _ = self.GAN(v, seq_time_step, label)
#
#         Fused_V = self.fuse(torch.cat([v, Gen_V], dim=-1))
#
#         h, _ = self.lstm(Fused_V)
#
#         og_softmax = torch.zeros(batch_size, visit_size, 2).to(self.device)
#
#         for i in range(visit_size):
#             og_softmax[:, i:i + 1, :] = self.classifyer(h[:, i:i + 1, :])
#
#         final_prediction_og = og_softmax[:, -1, :]
#
#         return final_prediction_og, Dec_V, Gen_V, seq_time_step, label
#
# class LSTM_GcGAN(nn.Module):
#     def __init__(self, vocab_size, d_model, h_model, dropout, dropout_emb, device, GAN, num_layers=1, m=0,
#                  dense_model=64):
#         super().__init__()
#         self.device = device
#         self.dropout = nn.Dropout(dropout)
#         self.emb_dropout = nn.Dropout(dropout_emb)
#         self.lstm = nn.LSTM(d_model, h_model, num_layers, bidirectional=False, batch_first=True, dropout=dropout)
#         self.m = m
#         self.tanh = nn.Tanh()
#         self.time_layer = nn.Linear(1, 64)
#         self.time_updim = nn.Linear(64, d_model)
#         self.initial_embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
#         self.relu = nn.ReLU()
#         self.GAN = GAN
#         self.classifyer = classifyer(d_model)
#         self.fuse = nn.Linear(d_model * 2, d_model)
#
#     def before(self, input_seqs, seq_time_step):
#         # time embedding
#         seq_time_step = seq_time_step.unsqueeze(2) / 180
#         time_feature = 1 - self.tanh(torch.pow(self.time_layer(seq_time_step), 2))
#         time_encoding = self.time_updim(time_feature)
#         # visit_embedding e_t
#         visit_embedding = self.initial_embedding(input_seqs)
#         visit_embedding = self.emb_dropout(visit_embedding)
#         visit_embedding = self.relu(visit_embedding)
#
#         visit_embedding = visit_embedding.sum(-2) + time_encoding
#         return visit_embedding
#
#     def forward(self, input_seqs, mask, length, seq_time_step, codemask, label):
#         batch_size, visit_size, seq_size = input_seqs.size()
#         v = self.before(input_seqs, seq_time_step)
#
#         Dec_V, Gen_V, _, Gen_label = self.GAN(v, seq_time_step, label)
#
#         Fused_V = self.fuse(torch.cat([v, Gen_V], dim=-1))
#
#         h, _ = self.lstm(Fused_V)
#
#         og_softmax = torch.zeros(batch_size, visit_size, 2).to(self.device)
#
#         for i in range(visit_size):
#             og_softmax[:, i:i + 1, :] = self.classifyer(h[:, i:i + 1, :])
#
#         final_prediction_og = og_softmax[:, -1, :]
#
#         return final_prediction_og, Dec_V, Gen_V, seq_time_step, Gen_label
#
# class LSTM_ehrGAN(nn.Module):
#     def __init__(self, vocab_size, d_model, h_model, dropout, dropout_emb, device, GAN, num_layers=1, m=0,
#                  dense_model=64):
#         super().__init__()
#         self.device = device
#         self.dropout = nn.Dropout(dropout)
#         self.emb_dropout = nn.Dropout(dropout_emb)
#         self.lstm = nn.LSTM(d_model, h_model, num_layers, bidirectional=False, batch_first=True, dropout=dropout)
#         self.m = m
#         self.tanh = nn.Tanh()
#         self.time_layer = nn.Linear(1, 64)
#         self.time_updim = nn.Linear(64, d_model)
#         self.initial_embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
#         self.relu = nn.ReLU()
#         self.GAN = GAN
#         self.classifyer = classifyer(d_model)
#         self.fuse = nn.Linear(d_model * 2, d_model)
#
#     def before(self, input_seqs, seq_time_step):
#         # time embedding
#         seq_time_step = seq_time_step.unsqueeze(2) / 180
#         time_feature = 1 - self.tanh(torch.pow(self.time_layer(seq_time_step), 2))
#         time_encoding = self.time_updim(time_feature)
#         # visit_embedding e_t
#         visit_embedding = self.initial_embedding(input_seqs)
#         visit_embedding = self.emb_dropout(visit_embedding)
#         visit_embedding = self.relu(visit_embedding)
#
#         visit_embedding = visit_embedding.sum(-2) + time_encoding
#         return visit_embedding
#
#     def forward(self, input_seqs, mask, length, seq_time_step, codemask, label):
#         batch_size, visit_size, seq_size = input_seqs.size()
#         v = self.before(input_seqs, seq_time_step)
#
#         Dec_V, Gen_V, _, _ = self.GAN(v, seq_time_step, label)
#
#         Fused_V = self.fuse(torch.cat([v, Gen_V], dim=-1))
#
#         h, _ = self.lstm(Fused_V)
#
#         og_softmax = torch.zeros(batch_size, visit_size, 2).to(self.device)
#
#         for i in range(visit_size):
#             og_softmax[:, i:i + 1, :] = self.classifyer(h[:, i:i + 1, :])
#
#         final_prediction_og = og_softmax[:, -1, :]
#
#         return final_prediction_og, Dec_V, Gen_V, seq_time_step, label

if __name__ == '__main__':
    y_true = np.array([])
    print(len(y_true))
