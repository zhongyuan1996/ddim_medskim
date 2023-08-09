import math
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.unet import *
from utils.diffUtil import get_beta_schedule
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
        fake_h = self.Linear_generator(z) * mask
        positive_h = x * mask

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

class medDiff(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb, num_layers, num_heads, max_pos, device, initial_d=0):
        super().__init__()
        self.device = device
        #####################################
        self.diff_channel = max_pos
        self.device = device
        with open(os.path.join("configs/", 'ehr.yml'), "r") as f:
            config = yaml.safe_load(f)
        self.config = dict2namespace(config)
        self.diffusion = UNetModel(in_channels=self.diff_channel, model_channels=128,
                                   out_channels=self.diff_channel, num_res_blocks=2,
                                   attention_resolutions=[16, ])
        betas = get_beta_schedule(beta_schedule=self.config.diffusion.beta_schedule,
                                  beta_start=self.config.diffusion.beta_start,
                                  beta_end=self.config.diffusion.beta_end,
                                  num_diffusion_timesteps=self.config.diffusion.num_diffusion_timesteps)
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.diffusion_num_timesteps = betas.shape[0]
        self.fuse = nn.Linear(512, 256)
        self.hiddenstate_learner = nn.LSTM(256, 256, 1, batch_first=True)
        self.w_hk = nn.Linear(256, 256)
        self.w1 = nn.Linear(2*256, 64)
        self.w2 = nn.Linear(64,2)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        ####################################

    def forward(self, e_t, h_t, time_seqs, label):

        aligned_e_t_1 = torch.zeros_like(e_t)
        for i in range(aligned_e_t_1.shape[1]):
            if i == 0:
                aligned_e_t_1[:, 0:1,:] = e_t[:, 0:1, :]
            else:
                e_k_1 = e_t[:, 0:1, :]
                w_h_k_prev_1 = self.w_hk(h_t[:,i-1:i,:])
                attn_1 = self.softmax(self.w2(self.tanh(self.w1(torch.cat((e_k_1,w_h_k_prev_1),dim=-1)))))
                alpha1_1 = attn_1[:,:,0:1]
                alpha2_1 = attn_1[:,:,1:2]
                aligned_e_t_1[:, i:i+1,:] = e_k_1 * alpha1_1 + w_h_k_prev_1 * alpha2_1

        ##########diff start
        diffusion_time_t_1 = torch.randint(
            low=0, high=self.config.diffusion.num_diffusion_timesteps, size=[aligned_e_t_1.shape[0], ]).to(
            self.device)
        alpha_1 = (1 - self.betas).cumprod(dim=0).index_select(0, diffusion_time_t_1).view(-1, 1, 1)
        normal_noise_1 = torch.randn_like(aligned_e_t_1)
        final_statues_with_noise_1 = aligned_e_t_1 * alpha_1.sqrt() + normal_noise_1 * (1.0 - alpha_1).sqrt()
        predicted_noise_1 = self.diffusion(final_statues_with_noise_1, timesteps=diffusion_time_t_1)
        noise_loss_1 = normal_noise_1 - predicted_noise_1
        GEN_e_t = aligned_e_t_1 + noise_loss_1
        ####### diff end

        return e_t, GEN_e_t, time_seqs, label



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

class LSTM(nn.Module):
    def __init__(self, vocab_size, d_model, h_model, dropout, dropout_emb, device, GAN, initial_d = 0, num_layers=1, m=0,
                 dense_model=64, balanced=False):
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
        self.relu = nn.ReLU()
        self.classifyer = classifyer(d_model)
        self.initial_d = initial_d
        self.initial_embedding_2 = nn.Linear(self.initial_d, d_model)
        self.balanced=balanced

    def before(self, input_seqs, seq_time_step):
        # time embedding
        # seq_time_step = seq_time_step.unsqueeze(2) / 180
        # time_feature = 1 - self.tanh(torch.pow(self.time_layer(seq_time_step), 2))
        # time_encoding = self.time_updim(time_feature)
        # visit_embedding e_t
        if self.initial_d != 0:
            visit_embedding = self.initial_embedding_2(input_seqs)
            visit_embedding = self.emb_dropout(visit_embedding)
            visit_embedding = self.relu(visit_embedding)
            # visit_embedding = visit_embedding + time_encoding
        else:
            visit_embedding = self.initial_embedding(input_seqs)
            visit_embedding = self.emb_dropout(visit_embedding)
            visit_embedding = self.relu(visit_embedding)

            # visit_embedding = visit_embedding.sum(-2) + time_encoding
            visit_embedding = visit_embedding.sum(-2)
        return visit_embedding

    def forward(self, input_seqs, mask, length, seq_time_step, codemask, label):
        batch_size, visit_size, seq_size = input_seqs.size()
        v = self.before(input_seqs, seq_time_step)

        if not self.balanced:

            Dec_V, Gen_V, _, _ = self.GAN(v, seq_time_step, label)

            h, _ = self.lstm(v)
            gen_h, _ = self.lstm(Gen_V)

            og_softmax = torch.zeros(batch_size, visit_size, 2).to(self.device)
            fake_softmax = torch.zeros(batch_size, visit_size, 2).to(self.device)

            for i in range(visit_size):
                og_softmax[:, i:i + 1, :] = self.classifyer(h[:, i:i + 1, :])
                fake_softmax[:, i:i + 1, :] = self.classifyer(gen_h[:, i:i + 1, :])

            final_prediction_og = og_softmax[:, -1, :]
            final_prediction_fake = fake_softmax[:, -1, :]

            return final_prediction_og, final_prediction_fake, Dec_V, Gen_V, seq_time_step, label
        else:
            v_p = v.clone()
            positive_mask = label == 1
            v_p = torch.cat([v_p, v_p[positive_mask], v_p[positive_mask]], dim=0)
            seq_time_step_p = torch.cat([seq_time_step, seq_time_step[positive_mask], seq_time_step[positive_mask]], dim=0)
            label_p = torch.cat([label, label[positive_mask], label[positive_mask]], dim=0)
            Dec_V, Gen_V, _, _ = self.GAN(v_p, seq_time_step_p, label_p)

            h, _ = self.lstm(v)
            gen_h, _ = self.lstm(Gen_V)

            og_softmax = torch.zeros(batch_size, visit_size, 2).to(self.device)
            new_size = v_p.size(0)
            fake_softmax = torch.zeros(new_size, visit_size, 2).to(self.device)

            for i in range(visit_size):
                og_softmax[:, i:i + 1, :] = self.classifyer(h[:, i:i + 1, :])
                fake_softmax[:, i:i + 1, :] = self.classifyer(gen_h[:, i:i + 1, :])

            final_prediction_og = og_softmax[:, -1, :]
            final_prediction_fake = fake_softmax[0:batch_size, -1, :]

            return final_prediction_og, final_prediction_fake, Dec_V, Gen_V, seq_time_step, label

class LSTM_medGAN(nn.Module):
    def __init__(self, vocab_size, d_model, h_model, dropout, dropout_emb, device, GAN, initial_d = 0, num_layers=1, m=0,
                 dense_model=64, balanced=False):
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
        self.relu = nn.ReLU()
        self.classifyer = classifyer(d_model)
        self.initial_d = initial_d
        self.initial_embedding_2 = nn.Linear(self.initial_d, d_model)
        self.balanced=balanced

    def before(self, input_seqs, seq_time_step):
        # time embedding
        # seq_time_step = seq_time_step.unsqueeze(2) / 180
        # time_feature = 1 - self.tanh(torch.pow(self.time_layer(seq_time_step), 2))
        # time_encoding = self.time_updim(time_feature)
        # visit_embedding e_t
        if self.initial_d != 0:
            visit_embedding = self.initial_embedding_2(input_seqs)
            visit_embedding = self.emb_dropout(visit_embedding)
            visit_embedding = self.relu(visit_embedding)
            # visit_embedding = visit_embedding + time_encoding
        else:
            visit_embedding = self.initial_embedding(input_seqs)
            visit_embedding = self.emb_dropout(visit_embedding)
            visit_embedding = self.relu(visit_embedding)

            # visit_embedding = visit_embedding.sum(-2) + time_encoding
            visit_embedding = visit_embedding.sum(-2)
        return visit_embedding

    def forward(self, input_seqs, mask, length, seq_time_step, codemask, label):
        batch_size, visit_size, seq_size = input_seqs.size()
        v = self.before(input_seqs, seq_time_step)

        if not self.balanced:

            Dec_V, Gen_V, _, _ = self.GAN(v, seq_time_step, label)

            h, _ = self.lstm(v)
            gen_h, _ = self.lstm(Gen_V)

            og_softmax = torch.zeros(batch_size, visit_size, 2).to(self.device)
            fake_softmax = torch.zeros(batch_size, visit_size, 2).to(self.device)

            for i in range(visit_size):
                og_softmax[:, i:i + 1, :] = self.classifyer(h[:, i:i + 1, :])
                fake_softmax[:, i:i + 1, :] = self.classifyer(gen_h[:, i:i + 1, :])

            final_prediction_og = og_softmax[:, -1, :]
            final_prediction_fake = fake_softmax[:, -1, :]

            return final_prediction_og, final_prediction_fake, Dec_V, Gen_V, seq_time_step, label
        else:
            v_p = v.clone()
            positive_mask = label == 1
            v_p = torch.cat([v_p, v_p[positive_mask], v_p[positive_mask]], dim=0)
            seq_time_step_p = torch.cat([seq_time_step, seq_time_step[positive_mask], seq_time_step[positive_mask]], dim=0)
            label_p = torch.cat([label, label[positive_mask], label[positive_mask]], dim=0)
            Dec_V, Gen_V, _, _ = self.GAN(v_p, seq_time_step_p, label_p)

            h, _ = self.lstm(v)
            gen_h, _ = self.lstm(Gen_V)

            og_softmax = torch.zeros(batch_size, visit_size, 2).to(self.device)
            new_size = v_p.size(0)
            fake_softmax = torch.zeros(new_size, visit_size, 2).to(self.device)

            for i in range(visit_size):
                og_softmax[:, i:i + 1, :] = self.classifyer(h[:, i:i + 1, :])
                fake_softmax[:, i:i + 1, :] = self.classifyer(gen_h[:, i:i + 1, :])

            final_prediction_og = og_softmax[:, -1, :]
            final_prediction_fake = fake_softmax[0:batch_size, -1, :]

            return final_prediction_og, final_prediction_fake, Dec_V, Gen_V, seq_time_step, label



class LSTM_actGAN(nn.Module):
    def __init__(self, vocab_size, d_model, h_model, dropout, dropout_emb, device, GAN, initial_d=0, num_layers=1, m=0,
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
        self.initial_d = initial_d
        self.initial_embedding_2 = nn.Linear(self.initial_d, d_model)

    def before(self, input_seqs, seq_time_step):
        # time embedding
        seq_time_step = seq_time_step.unsqueeze(2) / 180
        time_feature = 1 - self.tanh(torch.pow(self.time_layer(seq_time_step), 2))
        time_encoding = self.time_updim(time_feature)
        # visit_embedding e_t
        if self.initial_d != 0:
            visit_embedding = self.initial_embedding_2(input_seqs)
            visit_embedding = self.emb_dropout(visit_embedding)
            visit_embedding = self.relu(visit_embedding)
            visit_embedding = visit_embedding + time_encoding
        else:
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
        final_prediction_fake = fake_softmax[:, -1, :]
        return final_prediction_og, final_prediction_fake, Dec_V, Gen_V, seq_time_step, label

class LSTM_GcGAN(nn.Module):
    def __init__(self, vocab_size, d_model, h_model, dropout, dropout_emb, device, GAN, initial_d=0, num_layers=1, m=0,
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
        self.initial_d = initial_d
        self.initial_embedding_2 = nn.Linear(self.initial_d, d_model)

    def before(self, input_seqs, seq_time_step):
        # time embedding
        seq_time_step = seq_time_step.unsqueeze(2) / 180
        time_feature = 1 - self.tanh(torch.pow(self.time_layer(seq_time_step), 2))
        time_encoding = self.time_updim(time_feature)
        # visit_embedding e_t
        if self.initial_d != 0:
            visit_embedding = self.initial_embedding_2(input_seqs)
            visit_embedding = self.emb_dropout(visit_embedding)
            visit_embedding = self.relu(visit_embedding)
            visit_embedding = visit_embedding + time_encoding
        else:
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
        final_prediction_fake = fake_softmax[:, -1, :]
        return final_prediction_og, final_prediction_fake, Dec_V, Gen_V, seq_time_step, label

class LSTM_ehrGAN(nn.Module):
    def __init__(self, vocab_size, d_model, h_model, dropout, dropout_emb, device, GAN, initial_d=0, num_layers=1, m=0,
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
        self.initial_d = initial_d
        self.initial_embedding_2 = nn.Linear(self.initial_d, d_model)

    def before(self, input_seqs, seq_time_step):
        # time embedding
        seq_time_step = seq_time_step.unsqueeze(2) / 180
        time_feature = 1 - self.tanh(torch.pow(self.time_layer(seq_time_step), 2))
        time_encoding = self.time_updim(time_feature)
        # visit_embedding e_t
        if self.initial_d != 0:
            visit_embedding = self.initial_embedding_2(input_seqs)
            visit_embedding = self.emb_dropout(visit_embedding)
            visit_embedding = self.relu(visit_embedding)
            visit_embedding = visit_embedding + time_encoding
        else:
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
        final_prediction_fake = fake_softmax[:, -1, :]
        return final_prediction_og, final_prediction_fake, Dec_V, Gen_V, seq_time_step, label

class LSTMmedGANwatt(nn.Module):
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
        self.relu = nn.ReLU()
        self.classifyer = classifyer(d_model)
        self.initial_d = initial_d
        self.initial_embedding_2 = nn.Linear(self.initial_d, d_model)

        self.w_hk = nn.Linear(h_model, d_model)
        self.w1 = nn.Linear(2 * h_model, 64)
        self.w2 = nn.Linear(64, 2)
    def before(self, input_seqs, seq_time_step):
        # time embedding
        # seq_time_step = seq_time_step.unsqueeze(2) / 180
        # time_feature = 1 - self.tanh(torch.pow(self.time_layer(seq_time_step), 2))
        # time_encoding = self.time_updim(time_feature)
        # visit_embedding e_t
        if self.initial_d != 0:
            visit_embedding = self.initial_embedding_2(input_seqs)
            visit_embedding = self.emb_dropout(visit_embedding)
            visit_embedding = self.relu(visit_embedding)
            # visit_embedding = visit_embedding + time_encoding
        else:
            visit_embedding = self.initial_embedding(input_seqs)
            visit_embedding = self.emb_dropout(visit_embedding)
            visit_embedding = self.relu(visit_embedding)

            # visit_embedding = visit_embedding.sum(-2) + time_encoding
            visit_embedding = visit_embedding.sum(-2)
        return visit_embedding

    def forward(self, input_seqs, mask, length, seq_time_step, codemask, label):
        batch_size, visit_size, seq_size = input_seqs.size()
        v = self.before(input_seqs, seq_time_step)
        h, _ = self.lstm(v)

        bar_v = torch.zeros_like(v)

        for i in range(visit_size):
            if i == 0:
                bar_v[:, 0:1, :] = v[:, 0:1, :]
            else:
                e_k = v[:, 0:1, :]
                w_h_k_prev = self.w_hk(h[:, i - 1:i, :])
                attn = self.softmax(self.w2(self.tanh(self.w1(torch.cat((e_k, w_h_k_prev), dim=-1)))))
                alpha1 = attn[:, :, 0:1]
                alpha2 = attn[:, :, 1:2]
                bar_v[:, i:i + 1, :] = e_k * alpha1 + w_h_k_prev * alpha2

        Dec_V, Gen_V, _, _ = self.GAN(bar_v, seq_time_step, label)

        gen_h, _ = self.lstm(Gen_V)

        og_softmax = torch.zeros(batch_size, visit_size, 2).to(self.device)
        fake_softmax = torch.zeros(batch_size, visit_size, 2).to(self.device)

        for i in range(visit_size):
            og_softmax[:, i:i + 1, :] = self.classifyer(h[:, i:i + 1, :])
            fake_softmax[:, i:i + 1, :] = self.classifyer(gen_h[:, i:i + 1, :])

        final_prediction_og = og_softmax[:, -1, :]
        final_prediction_fake = fake_softmax[:, -1, :]
        return final_prediction_og, final_prediction_fake, Dec_V, Gen_V, seq_time_step, label




class LSTMactGANwatt(nn.Module):
    def __init__(self, vocab_size, d_model, h_model, dropout, dropout_emb, device, GAN, initial_d=0, num_layers=1, m=0,
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
        self.initial_d = initial_d
        self.initial_embedding_2 = nn.Linear(self.initial_d, d_model)

        self.w_hk = nn.Linear(h_model,d_model)
        self.w1 = nn.Linear(2*h_model, 64)
        self.w2 = nn.Linear(64,2)

    def before(self, input_seqs, seq_time_step):
        # time embedding
        seq_time_step = seq_time_step.unsqueeze(2) / 180
        time_feature = 1 - self.tanh(torch.pow(self.time_layer(seq_time_step), 2))
        time_encoding = self.time_updim(time_feature)
        # visit_embedding e_t
        if self.initial_d != 0:
            visit_embedding = self.initial_embedding_2(input_seqs)
            visit_embedding = self.emb_dropout(visit_embedding)
            visit_embedding = self.relu(visit_embedding)
            visit_embedding = visit_embedding + time_encoding
        else:
            visit_embedding = self.initial_embedding(input_seqs)
            visit_embedding = self.emb_dropout(visit_embedding)
            visit_embedding = self.relu(visit_embedding)

            visit_embedding = visit_embedding.sum(-2) + time_encoding
        return visit_embedding

    def forward(self, input_seqs, mask, length, seq_time_step, codemask, label):
        batch_size, visit_size, seq_size = input_seqs.size()
        v = self.before(input_seqs, seq_time_step)
        h, _ = self.lstm(v)

        bar_v = torch.zeros_like(v)

        for i in range(visit_size):
            if i == 0:
                bar_v[:, 0:1, :] = v[:, 0:1, :]
            else:
                e_k = v[:, 0:1, :]
                w_h_k_prev = self.w_hk(h[:, i - 1:i, :])
                attn = self.softmax(self.w2(self.tanh(self.w1(torch.cat((e_k, w_h_k_prev), dim=-1)))))
                alpha1 = attn[:, :, 0:1]
                alpha2 = attn[:, :, 1:2]
                bar_v[:, i:i + 1, :] = e_k * alpha1 + w_h_k_prev * alpha2

        Dec_V, Gen_V, _, _ = self.GAN(bar_v, seq_time_step, label)

        gen_h, _ = self.lstm(Gen_V)

        og_softmax = torch.zeros(batch_size, visit_size, 2).to(self.device)
        fake_softmax = torch.zeros(batch_size, visit_size, 2).to(self.device)

        for i in range(visit_size):
            og_softmax[:, i:i + 1, :] = self.classifyer(h[:, i:i + 1, :])
            fake_softmax[:, i:i + 1, :] = self.classifyer(gen_h[:, i:i + 1, :])

        final_prediction_og = og_softmax[:, -1, :]
        final_prediction_fake = fake_softmax[:, -1, :]
        return final_prediction_og, final_prediction_fake, Dec_V, Gen_V, seq_time_step, label

class LSTMGcGANwatt(nn.Module):
    def __init__(self, vocab_size, d_model, h_model, dropout, dropout_emb, device, GAN, initial_d=0, num_layers=1, m=0,
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
        self.initial_d = initial_d
        self.initial_embedding_2 = nn.Linear(self.initial_d, d_model)

        self.w_hk = nn.Linear(h_model, d_model)
        self.w1 = nn.Linear(2 * h_model, 64)
        self.w2 = nn.Linear(64, 2)
    def before(self, input_seqs, seq_time_step):
        # time embedding
        seq_time_step = seq_time_step.unsqueeze(2) / 180
        time_feature = 1 - self.tanh(torch.pow(self.time_layer(seq_time_step), 2))
        time_encoding = self.time_updim(time_feature)
        # visit_embedding e_t
        if self.initial_d != 0:
            visit_embedding = self.initial_embedding_2(input_seqs)
            visit_embedding = self.emb_dropout(visit_embedding)
            visit_embedding = self.relu(visit_embedding)
            visit_embedding = visit_embedding + time_encoding
        else:
            visit_embedding = self.initial_embedding(input_seqs)
            visit_embedding = self.emb_dropout(visit_embedding)
            visit_embedding = self.relu(visit_embedding)

            visit_embedding = visit_embedding.sum(-2) + time_encoding
        return visit_embedding

    def forward(self, input_seqs, mask, length, seq_time_step, codemask, label):
        batch_size, visit_size, seq_size = input_seqs.size()
        v = self.before(input_seqs, seq_time_step)
        h, _ = self.lstm(v)

        bar_v = torch.zeros_like(v)

        for i in range(visit_size):
            if i == 0:
                bar_v[:, 0:1, :] = v[:, 0:1, :]
            else:
                e_k = v[:, 0:1, :]
                w_h_k_prev = self.w_hk(h[:, i - 1:i, :])
                attn = self.softmax(self.w2(self.tanh(self.w1(torch.cat((e_k, w_h_k_prev), dim=-1)))))
                alpha1 = attn[:, :, 0:1]
                alpha2 = attn[:, :, 1:2]
                bar_v[:, i:i + 1, :] = e_k * alpha1 + w_h_k_prev * alpha2

        Dec_V, Gen_V, _, _ = self.GAN(bar_v, seq_time_step, label)

        gen_h, _ = self.lstm(Gen_V)

        og_softmax = torch.zeros(batch_size, visit_size, 2).to(self.device)
        fake_softmax = torch.zeros(batch_size, visit_size, 2).to(self.device)

        for i in range(visit_size):
            og_softmax[:, i:i + 1, :] = self.classifyer(h[:, i:i + 1, :])
            fake_softmax[:, i:i + 1, :] = self.classifyer(gen_h[:, i:i + 1, :])

        final_prediction_og = og_softmax[:, -1, :]
        final_prediction_fake = fake_softmax[:, -1, :]
        return final_prediction_og, final_prediction_fake, Dec_V, Gen_V, seq_time_step, label

class LSTMehrGANwatt(nn.Module):
    def __init__(self, vocab_size, d_model, h_model, dropout, dropout_emb, device, GAN, initial_d=0, num_layers=1, m=0,
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
        self.initial_d = initial_d
        self.initial_embedding_2 = nn.Linear(self.initial_d, d_model)

        self.w_hk = nn.Linear(h_model, d_model)
        self.w1 = nn.Linear(2 * h_model, 64)
        self.w2 = nn.Linear(64, 2)
    def before(self, input_seqs, seq_time_step):
        # time embedding
        seq_time_step = seq_time_step.unsqueeze(2) / 180
        time_feature = 1 - self.tanh(torch.pow(self.time_layer(seq_time_step), 2))
        time_encoding = self.time_updim(time_feature)
        # visit_embedding e_t
        if self.initial_d != 0:
            visit_embedding = self.initial_embedding_2(input_seqs)
            visit_embedding = self.emb_dropout(visit_embedding)
            visit_embedding = self.relu(visit_embedding)
            visit_embedding = visit_embedding + time_encoding
        else:
            visit_embedding = self.initial_embedding(input_seqs)
            visit_embedding = self.emb_dropout(visit_embedding)
            visit_embedding = self.relu(visit_embedding)

            visit_embedding = visit_embedding.sum(-2) + time_encoding
        return visit_embedding

    def forward(self, input_seqs, mask, length, seq_time_step, codemask, label):
        batch_size, visit_size, seq_size = input_seqs.size()
        v = self.before(input_seqs, seq_time_step)
        h, _ = self.lstm(v)

        bar_v = torch.zeros_like(v)

        for i in range(visit_size):
            if i == 0:
                bar_v[:, 0:1, :] = v[:, 0:1, :]
            else:
                e_k = v[:, 0:1, :]
                w_h_k_prev = self.w_hk(h[:, i - 1:i, :])
                attn = self.softmax(self.w2(self.tanh(self.w1(torch.cat((e_k, w_h_k_prev), dim=-1)))))
                alpha1 = attn[:, :, 0:1]
                alpha2 = attn[:, :, 1:2]
                bar_v[:, i:i + 1, :] = e_k * alpha1 + w_h_k_prev * alpha2

        Dec_V, Gen_V, _, _ = self.GAN(bar_v, seq_time_step, label)

        gen_h, _ = self.lstm(Gen_V)

        og_softmax = torch.zeros(batch_size, visit_size, 2).to(self.device)
        fake_softmax = torch.zeros(batch_size, visit_size, 2).to(self.device)

        for i in range(visit_size):
            og_softmax[:, i:i + 1, :] = self.classifyer(h[:, i:i + 1, :])
            fake_softmax[:, i:i + 1, :] = self.classifyer(gen_h[:, i:i + 1, :])

        final_prediction_og = og_softmax[:, -1, :]
        final_prediction_fake = fake_softmax[:, -1, :]
        return final_prediction_og, final_prediction_fake, Dec_V, Gen_V, seq_time_step, label

class Dipole_base(nn.Module):
    def __init__(self, vocab_size, d_model, h_model, dropout, dropout_emb, device, GAN, initial_d=0, num_layers=1, m=0,
                 dense_model=64, balanced = False):
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
        self.initial_d = initial_d
        self.initial_embedding_2 = nn.Linear(self.initial_d, d_model)
        self.balanced = balanced

    # def before(self, input_seqs, seq_time_step):
    #     # time embedding
    #     seq_time_step = seq_time_step.unsqueeze(2) / 180
    #     time_feature = 1 - self.tanh(torch.pow(self.time_layer(seq_time_step), 2))
    #     time_encoding = self.time_updim(time_feature)
    #     # visit_embedding e_t
    #     if self.initial_d != 0:
    #         visit_embedding = self.initial_embedding_2(input_seqs)
    #         visit_embedding = self.emb_dropout(visit_embedding)
    #         visit_embedding = self.relu(visit_embedding)
    #         visit_embedding = visit_embedding + time_encoding
    #     else:
    #         visit_embedding = self.initial_embedding(input_seqs)
    #         visit_embedding = self.emb_dropout(visit_embedding)
    #         visit_embedding = self.relu(visit_embedding)
    #
    #         visit_embedding = visit_embedding.sum(-2) + time_encoding
    #     return visit_embedding

    def forward(self, input_seqs, mask, lengths, seq_time_step, codemask, label):
        batch_size, seq_len, num_cui_per_visit = input_seqs.size()
        if self.initial_d != 0:
            x = self.initial_embedding_2(input_seqs)
        else:
            x = self.embbedding(input_seqs).sum(dim=2)
        x = self.emb_dropout(x)

        if not self.balanced:


            Dec_V, Gen_V, _, _ = self.GAN(x, seq_time_step, label)

            # rnn_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            rnn_output, _ = self.gru(x)
            # rnn_output, _ = pad_packed_sequence(rnn_output, batch_first=True, total_length=seq_len)

            # gen_rnn_input = pack_padded_sequence(Gen_X, lengths.cpu(), batch_first=True, enforce_sorted=False)
            gen_rnn_output, _ = self.gru(Gen_V)
            # gen_rnn_output, _ = pad_packed_sequence(gen_rnn_output, batch_first=True, total_length=seq_len)

            weight = self.weight_layer(rnn_output)
            # mask = (torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len) >= lengths.unsqueeze(1))
            # att = torch.softmax(weight.squeeze().masked_fill(mask, -np.inf), dim=1).view(batch_size, seq_len)
            att = torch.softmax(weight.squeeze(), dim=1).view(batch_size, seq_len)
            weighted_features = rnn_output * att.unsqueeze(2)
            averaged_features = torch.sum(weighted_features, 1)
            averaged_features = self.dropout(averaged_features)
            pred = self.output_mlp(averaged_features)

            gen_weight = self.weight_layer(gen_rnn_output)
            # gen_mask = (torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len) >= lengths.unsqueeze(1))
            # gen_att = torch.softmax(gen_weight.squeeze().masked_fill(gen_mask, -np.inf), dim=1).view(batch_size, seq_len)
            gen_att = torch.softmax(gen_weight.squeeze(), dim=1).view(batch_size, seq_len)
            gen_weighted_features = gen_rnn_output * gen_att.unsqueeze(2)
            gen_averaged_features = torch.sum(gen_weighted_features, 1)
            gen_averaged_features = self.dropout(gen_averaged_features)
            gen_pred = self.output_mlp(gen_averaged_features)

        else:
            v_p = x.clone()
            positive_mask = label == 1
            v_p = torch.cat([v_p, v_p[positive_mask], v_p[positive_mask]], dim=0)
            seq_time_step_p = torch.cat([seq_time_step, seq_time_step[positive_mask], seq_time_step[positive_mask]], dim=0)
            label_p = torch.cat([label, label[positive_mask], label[positive_mask]], dim=0)
            Dec_V, Gen_V, _, _ = self.GAN(v_p, seq_time_step_p, label_p)
            new_size, new_seq_len = v_p.size(0), v_p.size(1)

            rnn_output, _ = self.gru(x)

            weight = self.weight_layer(rnn_output)
            att = torch.softmax(weight.squeeze(), dim=1).view(batch_size, seq_len)
            weighted_features = rnn_output * att.unsqueeze(2)
            averaged_features = torch.sum(weighted_features, 1)
            averaged_features = self.dropout(averaged_features)
            pred = self.output_mlp(averaged_features)


            gen_rnn_output, _ = self.gru(Gen_V)
            gen_weight = self.weight_layer(gen_rnn_output)
            gen_att = torch.softmax(gen_weight.squeeze(), dim=1).view(new_size, new_seq_len)
            gen_weighted_features = gen_rnn_output * gen_att.unsqueeze(2)
            gen_averaged_features = torch.sum(gen_weighted_features, 1)
            gen_averaged_features = self.dropout(gen_averaged_features)
            gen_pred = self.output_mlp(gen_averaged_features)
            gen_pred = gen_pred[:batch_size]

        return pred, gen_pred, Dec_V, Gen_V, seq_time_step, label

class TLSTM_base(nn.Module):

    def __init__(self, vocab_size, d_model, h_model, dropout, dropout_emb, device, GAN, initial_d=0, num_layers=1, m=0,
                 dense_model=64, balanced=False):
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
        self.initial_d = initial_d
        self.initial_embedding_2 = nn.Linear(self.initial_d, d_model)
        self.balance = balanced
    def forward(self, input_seqs, mask, lengths, seq_time_step, codemask, label):
        if self.initial_d != 0:
            x = self.initial_embedding_2(input_seqs)
        else:
            x = self.embbedding(input_seqs).sum(dim=2)
        if not self.balance:

            Dec_V, Gen_V, _, _ = self.GAN(x, seq_time_step, label)

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

            # out = self.pooler(outputs, lengths)
            out = self.pooler(outputs)
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
            gen_out = self.pooler(gen_outputs)
            gen_out = self.output_mlp(gen_out)

        else:
            v_p = x.clone()
            positive_mask = label == 1
            v_p = torch.cat([v_p, v_p[positive_mask], v_p[positive_mask]], dim=0)
            seq_time_step_p = torch.cat([seq_time_step, seq_time_step[positive_mask], seq_time_step[positive_mask]], dim=0)
            label_p = torch.cat([label, label[positive_mask], label[positive_mask]], dim=0)
            Dec_V, Gen_V, _, _ = self.GAN(v_p, seq_time_step_p, label_p)
            new_size, new_seq_len = v_p.size(0), v_p.size(1)

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
            out = self.pooler(outputs)
            out = self.output_mlp(out)

            new_b, new_seq, _ = Gen_V.size()
            gen_h = torch.zeros(new_b, self.d_model, requires_grad=False).to(x.device)
            gen_c = torch.zeros(new_b, self.d_model, requires_grad=False).to(x.device)
            gen_outputs = []
            seq_time_step_p = seq_time_step_p.unsqueeze(2) / 180
            time_feature_p = 1 - torch.tanh(torch.pow(self.selection_layer(seq_time_step_p), 2))
            time_feature_p = self.time_layer(time_feature_p)
            for s in range(new_seq):
                gen_c_s1 = torch.tanh(self.W_d(gen_c))
                gen_c_s2 = gen_c_s1 * time_feature_p[:, s]
                gen_c_l = gen_c - gen_c_s1
                gen_c_adj = gen_c_l + gen_c_s2
                gen_outs = self.W_all(gen_h) + self.U_all(Gen_V[:, s])
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
            gen_out = self.pooler(gen_outputs)
            gen_out = self.output_mlp(gen_out)
            gen_out = gen_out[:b]


        return out, gen_out, Dec_V, Gen_V, seq_time_step, label

class SAND_base(nn.Module):

    def __init__(self, vocab_size, d_model, h_model, dropout, dropout_emb, device, GAN, num_heads, max_pos, initial_d=0, num_layers=1, m=0,
                 dense_model=64, balanced=False):

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
        self.out_layer = nn.Linear(d_model, 2)
        self.layer_norm = nn.LayerNorm(d_model)
        self.GAN = GAN
        self.initial_d = initial_d
        self.initial_embedding_2 = nn.Linear(self.initial_d, d_model)
        self.balanced = balanced


    def forward(self, input_seqs, masks, lengths, seq_time_step, codemask, label):
        if self.initial_d != 0:
            x = self.initial_embedding_2(input_seqs)
        else:
            x = self.embbedding(input_seqs).sum(dim=2) + self.bias_embedding
        bs, sl, dm = x.size()
        # x = self.emb_dropout(x)
        # output_pos, ind_pos = self.pos_emb(lengths)
        # x += output_pos
        if not self.balanced:
            Dec_X, Gen_X, _, _ = self.GAN(x, seq_time_step, label)

            x, attention = self.encoder_layer(x, x, x)

            x = self.drop_out(x)
            output = self.out_layer(x.sum(-2))

            gen_x, gen_attention = self.encoder_layer(Gen_X, Gen_X, Gen_X)

            gen_x = self.drop_out(gen_x)
            gen_output = self.out_layer(gen_x.sum(-2))
        else:
            v_p = x.clone()
            positive_mask = label == 1
            v_p = torch.cat([v_p, v_p[positive_mask], v_p[positive_mask]], dim=0)
            seq_time_step_p = torch.cat([seq_time_step, seq_time_step[positive_mask], seq_time_step[positive_mask]], dim=0)
            label_p = torch.cat([label, label[positive_mask], label[positive_mask]], dim=0)
            Dec_V, Gen_V, _, _ = self.GAN(v_p, seq_time_step_p, label_p)

            batch_size=x.size(0)
            new_size, new_seq_len = v_p.size(0), v_p.size(1)

            x, attention = self.encoder_layer(x, x, x)
            x = self.drop_out(x)
            output = self.out_layer(x.sum(-2))

            gen_x, gen_attention = self.encoder_layer(Gen_V, Gen_V, Gen_V)
            gen_x = self.drop_out(gen_x)
            gen_output = self.out_layer(gen_x.sum(-2))
            gen_output = gen_output[:batch_size]
        return output, gen_output, Dec_V, Gen_V, seq_time_step, label

class HitaNet(nn.Module):
    def __init__(self, vocab_size, d_model, h_model, dropout, dropout_emb, device, GAN, num_heads, max_pos, initial_d=0, num_layers=1, m=0,
                 dense_model=64, balanced=False):
    # def __init__(self, vocab_size, d_model, dropout, dropout_emb, num_layers, device, GAN, num_heads, max_pos, initial_d=0):
        super(HitaNet, self).__init__()
        self.embbedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.bias_embedding = torch.nn.Parameter(torch.Tensor(d_model))
        bound = 1 / math.sqrt(vocab_size)
        init.uniform_(self.bias_embedding, -bound, bound)
        self.encoder_layers = nn.ModuleList([Attention(d_model, num_heads, dropout) for _ in range(1)])
        self.positional_feed_forward_layers = nn.ModuleList([nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.ReLU(),
                                                                           nn.Linear(4 * d_model, d_model))
                                                             for _ in range(1)])
        self.pos_emb = PositionalEncoding(d_model, max_pos)
        self.time_layer = torch.nn.Linear(64, 256)
        self.selection_layer = torch.nn.Linear(1, 64)
        self.selection_time_layer = nn.Linear(1, 64)
        self.weight_layer = torch.nn.Linear(64, 64)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.self_layer = torch.nn.Linear(256, 1)
        self.quiry_layer = torch.nn.Linear(256, 64)
        self.quiry_weight_layer = torch.nn.Linear(256, 2)
        self.dropout = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_mlp = nn.Sequential(nn.Linear(d_model, 2))
        self.GAN = GAN
        self.initial_d = initial_d
        self.embedding2 = nn.Linear(self.initial_d, d_model)
        self.balance = balanced

    def forward(self, input_seqs, masks, lengths, seq_time_step, code_masks, label):
        seq_time_step = seq_time_step.unsqueeze(2) / 180
        time_feature = 1 - self.tanh(torch.pow(self.selection_layer(seq_time_step), 2))
        # time_feature_cache = time_feature
        time_feature = self.time_layer(time_feature)
        if self.initial_d != 0:
            x = self.embedding2(input_seqs) + self.bias_embedding
        else:
            x = self.embbedding(input_seqs).sum(dim=2) + self.bias_embedding
        x = self.emb_dropout(x)
        bs, seq_length, d_model = x.size()
        # output_pos, ind_pos = self.pos_emb(lengths)
        # x += output_pos
        x += time_feature
        attentions = []
        outputs = []
        for i in range(len(self.encoder_layers)):
            x, attention = self.encoder_layers[i](x, x, x, masks)
            res = x
            x = self.positional_feed_forward_layers[i](x)
            x = self.dropout(x)
            x = self.layer_norm(x + res)
            attentions.append(attention)
            outputs.append(x)

        # final_statues = outputs[-1].gather(1, lengths[:, None, None].expand(bs, 1, d_model) - 1).expand(bs, seq_length, d_model)
        final_statues = outputs[-1].gather(1, torch.ones(bs, 1, d_model).long().to(x.device)).expand(bs, seq_length, d_model)

        if not self.balance:

            Dec_V, Gen_V, _, _ = self.GAN(final_statues, seq_time_step, label)

            quiryes = self.relu(self.quiry_layer(final_statues))
            # mask = (torch.arange(seq_length, device=x.device).unsqueeze(0).expand(bs, seq_length) >= lengths.unsqueeze(1))
            self_weight = torch.softmax(self.self_layer(outputs[-1]).squeeze(), dim=1).view(bs,seq_length).unsqueeze(2)
            selection_feature = self.relu(self.weight_layer(self.selection_time_layer(seq_time_step)))
            selection_feature = torch.sum(selection_feature * quiryes, 2) / 8
            time_weight = torch.softmax(selection_feature, dim=1).view(bs, seq_length).unsqueeze(
                2)
            #attention_weight = torch.softmax(self.quiry_weight_layer(final_statues), 2).view(bs, seq_length, 2)
            attention_weight = torch.softmax(self.quiry_weight_layer(final_statues), 2).view(bs, seq_length, 2)
            total_weight = torch.cat((time_weight, self_weight), 2)
            total_weight = torch.sum(total_weight * attention_weight, 2)
            total_weight = total_weight / (torch.sum(total_weight, 1, keepdim=True) + 1e-5)
            weighted_features = outputs[-1] * total_weight.unsqueeze(2)
            averaged_features = torch.sum(weighted_features, 1)
            averaged_features = self.dropout(averaged_features)
            prediction = self.output_mlp(averaged_features)

            GEN_quires = self.relu(self.quiry_layer(Gen_V))
            GEN_self_weight = torch.softmax(self.self_layer(outputs[-1]).squeeze(), dim=1).view(bs, seq_length).unsqueeze(2)
            GEN_selection_feature = self.relu(self.weight_layer(self.selection_time_layer(seq_time_step)))
            GEN_selection_feature = torch.sum(GEN_selection_feature * GEN_quires, 2) / 8
            GEN_time_weight = torch.softmax(GEN_selection_feature, dim=1).view(bs, seq_length).unsqueeze(
                2)
            GEN_attention_weight = torch.softmax(self.quiry_weight_layer(Gen_V), 2).view(bs, seq_length, 2)
            GEN_total_weight = torch.cat((GEN_time_weight, GEN_self_weight), 2)
            GEN_total_weight = torch.sum(GEN_total_weight * GEN_attention_weight, 2)
            GEN_total_weight = GEN_total_weight / (torch.sum(GEN_total_weight, 1, keepdim=True) + 1e-5)
            GEN_weighted_features = Gen_V * GEN_total_weight.unsqueeze(2)
            GEN_averaged_features = torch.sum(GEN_weighted_features, 1)
            GEN_averaged_features = self.dropout(GEN_averaged_features)
            GEN_prediction = self.output_mlp(GEN_averaged_features)

        else:
            v_p = final_statues.clone()
            positive_mask = label == 1
            v_p = torch.cat([v_p, v_p[positive_mask], v_p[positive_mask]], dim=0)
            outputs_p = torch.cat([outputs[-1], outputs[-1][positive_mask], outputs[-1][positive_mask]], dim=0)
            seq_time_step_p = torch.cat([seq_time_step, seq_time_step[positive_mask], seq_time_step[positive_mask]], dim=0)
            label_p = torch.cat([label, label[positive_mask], label[positive_mask]], dim=0)
            Dec_V, Gen_V, _, _ = self.GAN(v_p, seq_time_step_p, label_p)
            new_size, new_seq_len = v_p.size(0), v_p.size(1)

            quiryes = self.relu(self.quiry_layer(final_statues))
            # mask = (torch.arange(seq_length, device=x.device).unsqueeze(0).expand(bs, seq_length) >= lengths.unsqueeze(1))
            self_weight = torch.softmax(self.self_layer(outputs[-1]).squeeze(), dim=1).view(bs, seq_length).unsqueeze(2)
            selection_feature = self.relu(self.weight_layer(self.selection_time_layer(seq_time_step)))
            selection_feature = torch.sum(selection_feature * quiryes, 2) / 8
            time_weight = torch.softmax(selection_feature, dim=1).view(bs, seq_length).unsqueeze(
                2)
            # attention_weight = torch.softmax(self.quiry_weight_layer(final_statues), 2).view(bs, seq_length, 2)
            attention_weight = torch.softmax(self.quiry_weight_layer(final_statues), 2).view(bs, seq_length, 2)
            total_weight = torch.cat((time_weight, self_weight), 2)
            total_weight = torch.sum(total_weight * attention_weight, 2)
            total_weight = total_weight / (torch.sum(total_weight, 1, keepdim=True) + 1e-5)
            weighted_features = outputs[-1] * total_weight.unsqueeze(2)
            averaged_features = torch.sum(weighted_features, 1)
            averaged_features = self.dropout(averaged_features)
            prediction = self.output_mlp(averaged_features)

            GEN_quires = self.relu(self.quiry_layer(Gen_V))
            GEN_self_weight = torch.softmax(self.self_layer(outputs_p).squeeze(), dim=1).view(new_size, new_seq_len).unsqueeze(2)
            GEN_selection_feature = self.relu(self.weight_layer(self.selection_time_layer(seq_time_step_p)))
            GEN_selection_feature = torch.sum(GEN_selection_feature * GEN_quires, 2) / 8
            GEN_time_weight = torch.softmax(GEN_selection_feature, dim=1).view(new_size, new_seq_len).unsqueeze(
                2)
            GEN_attention_weight = torch.softmax(self.quiry_weight_layer(Gen_V), 2).view(new_size, new_seq_len, 2)
            GEN_total_weight = torch.cat((GEN_time_weight, GEN_self_weight), 2)
            GEN_total_weight = torch.sum(GEN_total_weight * GEN_attention_weight, 2)
            GEN_total_weight = GEN_total_weight / (torch.sum(GEN_total_weight, 1, keepdim=True) + 1e-5)
            GEN_weighted_features = outputs_p * GEN_total_weight.unsqueeze(2)
            GEN_averaged_features = torch.sum(GEN_weighted_features, 1)
            GEN_averaged_features = self.dropout(GEN_averaged_features)
            GEN_prediction = self.output_mlp(GEN_averaged_features)
            GEN_prediction = GEN_prediction[:bs]




        return prediction, GEN_prediction, Dec_V, Gen_V, seq_time_step, label

class Retain(nn.Module):
    def __init__(self, vocab_size, d_model, h_model, dropout, dropout_emb, device, GAN, num_heads, max_pos, initial_d=0, num_layers=1, m=0,
                 dense_model=64, balanced=False):
        super().__init__()
        self.embbedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.dropout = nn.Dropout(dropout)
        self.variable_level_rnn = nn.GRU(d_model, d_model, batch_first=True)
        self.visit_level_rnn = nn.GRU(d_model, d_model, batch_first=True)
        self.variable_level_attention = nn.Linear(d_model, d_model)
        self.visit_level_attention = nn.Linear(d_model, 1)
        self.output_dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(d_model, 2)

        self.var_hidden_size = d_model
        self.GAN = GAN
        self.visit_hidden_size = d_model

        self.initial_d = initial_d
        self.embedding2 = nn.Linear(self.initial_d, d_model)
        self.balance = balanced

    def forward(self, input_seqs, masks, lengths, seq_time_step, code_masks, label):
        if self.initial_d != 0:
            x = self.embedding2(input_seqs)
        else:
            x = self.embbedding(input_seqs).sum(dim=2)
        x = self.dropout(x)

        if not self.balance:

            Dec_V, GEN_V, _, _ = self.GAN(x, seq_time_step, label)

            visit_rnn_output, visit_rnn_hidden = self.visit_level_rnn(x)
            alpha = self.visit_level_attention(visit_rnn_output)
            visit_attn_w = torch.softmax(alpha, dim=1)
            var_rnn_output, var_rnn_hidden = self.variable_level_rnn(x)
            beta = self.variable_level_attention(var_rnn_output)
            var_attn_w = torch.tanh(beta)
            attn_w = visit_attn_w * var_attn_w
            c_all = attn_w * x
            c = torch.sum(c_all, dim=1)
            c = self.output_dropout(c)

            GEN_visit_rnn_output, GEN_visit_rnn_hidden = self.visit_level_rnn(GEN_V)
            GEN_alpha = self.visit_level_attention(GEN_visit_rnn_output)
            GEN_visit_attn_w = torch.softmax(GEN_alpha, dim=1)
            GEN_var_rnn_output, GEN_var_rnn_hidden = self.variable_level_rnn(GEN_V)
            GEN_beta = self.variable_level_attention(GEN_var_rnn_output)
            GEN_var_attn_w = torch.tanh(GEN_beta)
            GEN_attn_w = GEN_visit_attn_w * GEN_var_attn_w
            GEN_c_all = GEN_attn_w * GEN_V
            GEN_c = torch.sum(GEN_c_all, dim=1)
            GEN_c = self.output_dropout(GEN_c)

            output = self.output_layer(c)
            Gen_output = self.output_layer(GEN_c)
        else:
            v_p = x.clone()
            positive_mask = label == 1
            v_p = torch.cat([v_p, v_p[positive_mask], v_p[positive_mask]], dim=0)
            seq_time_step_p = torch.cat([seq_time_step, seq_time_step[positive_mask], seq_time_step[positive_mask]], dim=0)
            label_p = torch.cat([label, label[positive_mask], label[positive_mask]], dim=0)
            Dec_V, Gen_V, _, _ = self.GAN(v_p, seq_time_step_p, label_p)

            visit_rnn_output, visit_rnn_hidden = self.visit_level_rnn(x)
            alpha = self.visit_level_attention(visit_rnn_output)
            visit_attn_w = torch.softmax(alpha, dim=1)
            var_rnn_output, var_rnn_hidden = self.variable_level_rnn(x)
            beta = self.variable_level_attention(var_rnn_output)
            var_attn_w = torch.tanh(beta)
            attn_w = visit_attn_w * var_attn_w
            c_all = attn_w * x
            c = torch.sum(c_all, dim=1)
            c = self.output_dropout(c)
            output = self.output_layer(c)

            GEN_visit_rnn_output, GEN_visit_rnn_hidden = self.visit_level_rnn(Gen_V)
            GEN_alpha = self.visit_level_attention(GEN_visit_rnn_output)
            GEN_visit_attn_w = torch.softmax(GEN_alpha, dim=1)
            GEN_var_rnn_output, GEN_var_rnn_hidden = self.variable_level_rnn(Gen_V)
            GEN_beta = self.variable_level_attention(GEN_var_rnn_output)
            GEN_var_attn_w = torch.tanh(GEN_beta)
            GEN_attn_w = GEN_visit_attn_w * GEN_var_attn_w
            GEN_c_all = GEN_attn_w * Gen_V
            GEN_c = torch.sum(GEN_c_all, dim=1)
            GEN_c = self.output_dropout(GEN_c)
            Gen_output = self.output_layer(GEN_c)
            Gen_output = Gen_output[:x.shape[0]]

        return output, Gen_output, Dec_V, Gen_V, seq_time_step, label

class RetainEx(nn.Module):
    def __init__(self, vocab_size, d_model, h_model, dropout, dropout_emb, device, GAN, num_heads, max_pos, initial_d=0, num_layers=1, m=0,
                 dense_model=64, balanced=False):
        super().__init__()
        self.embbedding1 = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.embbedding2 = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.RNN1 = nn.LSTM(d_model + 3, d_model,
                            1, batch_first=True)
        self.RNN2 = nn.LSTM(d_model + 3, d_model,
                            1, batch_first=True)
        self.wa = nn.Linear(d_model, 1, bias=False)
        self.Wb = nn.Linear(d_model, d_model, bias=False)
        self.W_out = nn.Linear(d_model, 2, bias=False)
        self.drop_out = nn.Dropout(dropout)
        self.output_layer = nn.Linear(d_model, 2)
        self.GAN = GAN
        self.initial_d = initial_d
        self.linearembedding = nn.Linear(self.initial_d, d_model)
        self.balance = balanced

    def forward(self, input_seqs, masks, lengths, time_step, code_masks, label):
        if self.initial_d != 0:
            embedded = self.linearembedding(input_seqs)
            embedded2 = self.linearembedding(input_seqs)
        else:
            embedded = self.embbedding1(input_seqs).sum(dim=2)
            embedded2 = self.embbedding2(input_seqs).sum(dim=2)

        b, seq, features = embedded.size()
        dates = torch.stack([time_step, 1 / (time_step + 1), 1 / torch.log(np.e + time_step)], 2)  # [b x seq x 3]
        if not self.balance:
            og_Dec_embedded, og_GEN_embedded, _, _ = self.GAN(embedded, time_step, label)

            embedded = torch.cat([embedded, dates], 2)
            outputs1 = self.RNN1(embedded)[0]
            outputs2 = self.RNN2(embedded)[0]
            # print(outputs2.size())

            GEN_embedded = torch.cat([og_GEN_embedded, dates], 2)
            GEN_outputs1 = self.RNN1(GEN_embedded)[0]
            GEN_outputs2 = self.RNN2(GEN_embedded)[0]

            E = self.wa(outputs1.contiguous().view(b * seq, -1))
            alpha = F.softmax(E.view(b, seq), 1)
            outputs2 = self.Wb(outputs2.contiguous().view(b * seq, -1))  # [b*seq x hid]
            Beta = torch.tanh(outputs2).view(b, seq, features)
            v_all = (embedded2 * Beta) * alpha.unsqueeze(2).expand(b, seq, features)

            outputs = v_all.sum(1)  # [b x hidden]
            outputs = self.drop_out(outputs)
            outputs = self.output_layer(outputs)

            GEN_E = self.wa(GEN_outputs1.contiguous().view(b * seq, -1))
            GEN_alpha = F.softmax(GEN_E.view(b, seq), 1)
            GEN_outputs2 = self.Wb(GEN_outputs2.contiguous().view(b * seq, -1))  # [b*seq x hid]
            GEN_Beta = torch.tanh(GEN_outputs2).view(b, seq, features)
            GEN_v_all = (embedded2 * GEN_Beta) * GEN_alpha.unsqueeze(2).expand(b, seq, features)

            GEN_outputs = GEN_v_all.sum(1)  # [b x hidden]
            GEN_outputs = self.drop_out(GEN_outputs)
            GEN_outputs = self.output_layer(GEN_outputs)
        else:
            v_p = embedded.clone()
            positive_mask = label == 1
            v_p = torch.cat([v_p, v_p[positive_mask], v_p[positive_mask]], dim=0)
            embedded2_p = torch.cat([embedded2, embedded2[positive_mask], embedded2[positive_mask]], dim=0)
            dates_p = torch.cat([dates, dates[positive_mask], dates[positive_mask]], dim=0)
            seq_time_step_p = torch.cat([time_step, time_step[positive_mask], time_step[positive_mask]], dim=0)
            label_p = torch.cat([label, label[positive_mask], label[positive_mask]], dim=0)
            og_Dec_embedded, og_GEN_embedded, _, _ = self.GAN(v_p, seq_time_step_p, label_p)

            embedded = torch.cat([embedded, dates], 2)
            outputs1 = self.RNN1(embedded)[0]
            outputs2 = self.RNN2(embedded)[0]
            # print(outputs2.size())
            new_b, new_seq, new_features = v_p.size()

            GEN_embedded = torch.cat([og_GEN_embedded, dates_p], 2)
            GEN_outputs1 = self.RNN1(GEN_embedded)[0]
            GEN_outputs2 = self.RNN2(GEN_embedded)[0]

            E = self.wa(outputs1.contiguous().view(b * seq, -1))
            alpha = F.softmax(E.view(b, seq), 1)
            outputs2 = self.Wb(outputs2.contiguous().view(b * seq, -1))  # [b*seq x hid]
            Beta = torch.tanh(outputs2).view(b, seq, features)
            v_all = (embedded2 * Beta) * alpha.unsqueeze(2).expand(b, seq, features)

            outputs = v_all.sum(1)  # [b x hidden]
            outputs = self.drop_out(outputs)
            outputs = self.output_layer(outputs)

            GEN_E = self.wa(GEN_outputs1.contiguous().view(new_b * new_seq, -1))
            GEN_alpha = F.softmax(GEN_E.view(new_b, new_seq), 1)
            GEN_outputs2 = self.Wb(GEN_outputs2.contiguous().view(new_b * new_seq, -1))  # [b*seq x hid]
            GEN_Beta = torch.tanh(GEN_outputs2).view(new_b, new_seq, new_features)
            GEN_v_all = (embedded2_p * GEN_Beta) * GEN_alpha.unsqueeze(2).expand(new_b, new_seq, new_features)

            GEN_outputs = GEN_v_all.sum(1)  # [b x hidden]
            GEN_outputs = self.drop_out(GEN_outputs)
            GEN_outputs = self.output_layer(GEN_outputs)
            GEN_outputs = GEN_outputs[:b]


        return outputs, GEN_outputs, og_Dec_embedded, og_GEN_embedded, time_step, label

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
