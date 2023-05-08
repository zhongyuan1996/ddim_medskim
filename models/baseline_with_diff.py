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

class HitaNet(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb, num_layers, num_heads, max_pos, device, initial_d=0):
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
####################################
        self.initial_d = initial_d
        self.embedding2 = nn.Linear(self.initial_d, d_model)

    def forward(self, input_seqs, masks, lengths, seq_time_step, code_masks):
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

        hiddenstate, _ = self.hiddenstate_learner(final_statues)
        aligned_e_t = torch.zeros_like(hiddenstate)
        for i in range(aligned_e_t.shape[1]):
            if i == 0:
                aligned_e_t[:, 0:1,:] = final_statues[:, 0:1, :]
            else:
                e_k = final_statues[:, 0:1, :]
                w_h_k_prev = self.w_hk(hiddenstate[:,i-1:i,:])
                attn = self.softmax(self.w2(self.tanh(self.w1(torch.cat((e_k,w_h_k_prev),dim=-1)))))
                alpha1 = attn[:,:,0:1]
                alpha2 = attn[:,:,1:2]
                aligned_e_t[:, i:i+1,:] = e_k * alpha1 + w_h_k_prev * alpha2
        ##########diff start
        diffusion_time_t = torch.randint(
            low=0, high=self.config.diffusion.num_diffusion_timesteps, size=[aligned_e_t.shape[0], ]).to(
            self.device)
        alpha = (1 - self.betas).cumprod(dim=0).index_select(0, diffusion_time_t).view(-1, 1, 1)
        normal_noise = torch.randn_like(aligned_e_t)
        final_statues_with_noise = aligned_e_t * alpha.sqrt() + normal_noise * (1.0 - alpha).sqrt()
        predicted_noise = self.diffusion(final_statues_with_noise, timesteps=diffusion_time_t)
        noise_loss = normal_noise - predicted_noise
        GEN_status = aligned_e_t + noise_loss
        ####### diff end

        #######fuse gen result with original ones

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

        GEN_quires = self.relu(self.quiry_layer(GEN_status))
        GEN_self_weight = torch.softmax(self.self_layer(outputs[-1]).squeeze(), dim=1).view(bs, seq_length).unsqueeze(2)
        GEN_selection_feature = self.relu(self.weight_layer(self.selection_time_layer(seq_time_step)))
        GEN_selection_feature = torch.sum(GEN_selection_feature * GEN_quires, 2) / 8
        GEN_time_weight = torch.softmax(GEN_selection_feature, dim=1).view(bs, seq_length).unsqueeze(
            2)
        GEN_attention_weight = torch.softmax(self.quiry_weight_layer(GEN_status), 2).view(bs, seq_length, 2)
        GEN_total_weight = torch.cat((GEN_time_weight, GEN_self_weight), 2)
        GEN_total_weight = torch.sum(GEN_total_weight * GEN_attention_weight, 2)
        GEN_total_weight = GEN_total_weight / (torch.sum(GEN_total_weight, 1, keepdim=True) + 1e-5)
        GEN_weighted_features = GEN_status * GEN_total_weight.unsqueeze(2)
        GEN_averaged_features = torch.sum(GEN_weighted_features, 1)
        GEN_averaged_features = self.dropout(GEN_averaged_features)
        GEN_prediction = self.output_mlp(GEN_averaged_features)

        return prediction, GEN_prediction, predicted_noise, normal_noise


class LSAN(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb, num_layers, num_heads, max_pos, device):
        super().__init__()
        self.embbedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=vocab_size)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.encoder_layers = nn.ModuleList([Attention(d_model, num_heads, dropout) for _ in range(1)])
        self.positional_feed_forward_layers = nn.ModuleList([nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.ReLU(),
                                                                           nn.Linear(4 * d_model, d_model))
                                                             for _ in range(1)])
        self.pooler = MaxPoolLayer()
        self.pos_emb = PositionalEncoding(d_model, max_pos)
        self.MATT = nn.Sequential(nn.Linear(d_model, int(d_model / 4), bias=False),
                                  nn.ReLU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(int(d_model / 4), int(d_model / 8), bias=False),
                                  nn.ReLU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(int(d_model / 8), 1))
        visit_ATT_dim = 2 * d_model
        self.visit_ATT = nn.Sequential(nn.Linear(visit_ATT_dim, int(visit_ATT_dim / 4)),
                                       nn.ReLU(),
                                       nn.Dropout(dropout),
                                       nn.Linear(int(visit_ATT_dim / 4), int(visit_ATT_dim / 8)),
                                       nn.ReLU(),
                                       nn.Dropout(dropout),
                                       nn.Linear(int(visit_ATT_dim / 8), 1))
        self.Classifier = nn.Linear(2 * d_model, 2)
        self.local_conv_layer = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3,
                                          padding=1)
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
        self.fuse = nn.Linear(1024, 512)
        self.hiddenstate_learner = nn.LSTM(512, 512, 1, batch_first=True)
        self.w_hk = nn.Linear(512, 512)
        self.w1 = nn.Linear(2*512, 64)
        self.w2 = nn.Linear(64,2)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        ####################################

    def forward(self, input_seqs, masks, lengths, seq_time_step, code_masks):
        input_embedding = self.embbedding(input_seqs)
        bs, seqlen, numcode, d_model = input_embedding.size()
        input_embedding = input_embedding.view(bs * seqlen, numcode, d_model)
        attn_weight = F.softmax(self.MATT(input_embedding), dim=1)
        diag_result_att = torch.matmul(attn_weight.permute(0, 2, 1), input_embedding).squeeze(1)
        diag_result_att = diag_result_att.view(bs, seqlen, d_model)
        diag_result_att = self.emb_dropout(diag_result_att)
        output_pos, ind_pos = self.pos_emb(lengths)
        x = diag_result_att + output_pos
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

        local_conv_feat = self.local_conv_layer(diag_result_att.permute(0, 2, 1))
        concat_feat = torch.cat((outputs[-1], local_conv_feat.permute(0, 2, 1)), dim=2)

        hiddenstate, _ = self.hiddenstate_learner(concat_feat)
        aligned_e_t = torch.zeros_like(hiddenstate)
        for i in range(aligned_e_t.shape[1]):
            if i == 0:
                aligned_e_t[:, 0:1,:] = concat_feat[:, 0:1, :]
            else:
                e_k = concat_feat[:, 0:1, :]
                w_h_k_prev = self.w_hk(hiddenstate[:,i-1:i,:])
                attn = self.softmax(self.w2(self.tanh(self.w1(torch.cat((e_k,w_h_k_prev),dim=-1)))))
                alpha1 = attn[:,:,0:1]
                alpha2 = attn[:,:,1:2]
                aligned_e_t[:, i:i+1,:] = e_k * alpha1 + w_h_k_prev * alpha2
        ##########diff start
        diffusion_time_t = torch.randint(
            low=0, high=self.config.diffusion.num_diffusion_timesteps, size=[aligned_e_t.shape[0], ]).to(
            self.device)
        alpha = (1 - self.betas).cumprod(dim=0).index_select(0, diffusion_time_t).view(-1, 1, 1)
        normal_noise = torch.randn_like(aligned_e_t)
        final_statues_with_noise = aligned_e_t * alpha.sqrt() + normal_noise * (1.0 - alpha).sqrt()
        predicted_noise = self.diffusion(final_statues_with_noise, timesteps=diffusion_time_t)
        noise_loss = normal_noise - predicted_noise
        GEN_concat_feat = aligned_e_t + noise_loss
        ####### diff end
        fused_feat = self.fuse(torch.cat([concat_feat, GEN_concat_feat], dim=-1))
        #######fuse gen result with original ones

        visit_attn_weight = torch.softmax(self.visit_ATT(fused_feat), dim=1)
        visit_result_att = torch.matmul(visit_attn_weight.permute(0, 2, 1), fused_feat).squeeze(1)
        prediction_output = self.Classifier(visit_result_att)
        return prediction_output #visit_result_att, outputs[-1]


class LSTM_encoder(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb, num_layers, num_heads, max_pos, device, initial_d = 0):
        super().__init__()
        self.embbedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.dropout = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.output_mlp = nn.Sequential(nn.Linear(d_model, 2))
        self.pooler = MaxPoolLayer()
        self.rnns = nn.LSTM(d_model, d_model, num_layers, bidirectional=False, batch_first=True, dropout=dropout)
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
        self.initial_d = initial_d
        self.embedding2 = nn.Linear(self.initial_d, d_model)

    def forward(self, input_seqs, masks, lengths, seq_time_step, code_masks):
        batch_size, seq_len, num_cui_per_visit = input_seqs.size()
        if self.initial_d != 0:
            x = self.embedding2(input_seqs)
        else:
            x = self.embbedding(input_seqs).sum(dim=2)
        x = self.emb_dropout(x)
        # rnn_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        rnn_output, _ = self.rnns(x)


        # x, _ = pad_packed_sequence(rnn_output, batch_first=True, total_length=seq_len)


        aligned_e_t = torch.zeros_like(x)
        for i in range(aligned_e_t.shape[1]):
            if i == 0:
                aligned_e_t[:, 0:1,:] = x[:, 0:1, :]
            else:
                e_k = x[:, 0:1, :]
                w_h_k_prev = self.w_hk(rnn_output[:,i-1:i,:])
                attn = self.softmax(self.w2(self.tanh(self.w1(torch.cat((e_k,w_h_k_prev),dim=-1)))))
                alpha1 = attn[:,:,0:1]
                alpha2 = attn[:,:,1:2]
                aligned_e_t[:, i:i+1,:] = e_k * alpha1 + w_h_k_prev * alpha2

        ##########diff start
        diffusion_time_t = torch.randint(
            low=0, high=self.config.diffusion.num_diffusion_timesteps, size=[aligned_e_t.shape[0], ]).to(
            self.device)
        alpha = (1 - self.betas).cumprod(dim=0).index_select(0, diffusion_time_t).view(-1, 1, 1)
        normal_noise = torch.randn_like(aligned_e_t)
        final_statues_with_noise = aligned_e_t * alpha.sqrt() + normal_noise * (1.0 - alpha).sqrt()
        predicted_noise = self.diffusion(final_statues_with_noise, timesteps=diffusion_time_t)
        noise_loss = normal_noise - predicted_noise
        GEN_x = aligned_e_t + noise_loss
        ####### diff end
        # fused_x = self.fuse(torch.cat([x, GEN_x], dim=-1))
        #######fuse gen result with original ones
        pool_x = self.pooler(x)
        gen_pool_x = self.pooler(GEN_x)
        pool_x = self.output_mlp(pool_x)
        gen_pool_x = self.output_mlp(gen_pool_x)
        return pool_x, gen_pool_x, predicted_noise, normal_noise


class GRUSelf(nn.Module):
    #Dipole
    def __init__(self, vocab_size, d_model, dropout, dropout_emb, num_layers, num_heads, max_pos, device, initial_d=0):
        super().__init__()
        self.embbedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.dropout = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.output_mlp = nn.Sequential(nn.Linear(d_model, 2))
        self.gru = nn.GRU(d_model, d_model, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.weight_layer = nn.Linear(d_model, 1)
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
        self.initial_d = initial_d
        self.embeding2 = nn.Linear(self.initial_d, d_model)

    def forward(self, input_seqs, masks, lengths, seq_time_step, code_masks):
        batch_size, seq_len, num_cui_per_visit = input_seqs.size()
        if self.initial_d != 0:
            x = self.embeding2(input_seqs)
        else:
            x = self.embbedding(input_seqs).sum(dim=2)
        x = self.emb_dropout(x)
        # rnn_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        rnn_output, _ = self.gru(x)
        # rnn_output, _ = pad_packed_sequence(rnn_output, batch_first=True, total_length=seq_len)

        # hiddenstate, _ = self.hiddenstate_learner(rnn_output)
        aligned_e_t = torch.zeros_like(x)
        for i in range(aligned_e_t.shape[1]):
            if i == 0:
                aligned_e_t[:, 0:1,:] = x[:, 0:1, :]
            else:
                e_k = x[:, 0:1, :]
                w_h_k_prev = self.w_hk(rnn_output[:,i-1:i,:])
                attn = self.softmax(self.w2(self.tanh(self.w1(torch.cat((e_k,w_h_k_prev),dim=-1)))))
                alpha1 = attn[:,:,0:1]
                alpha2 = attn[:,:,1:2]
                aligned_e_t[:, i:i+1,:] = e_k * alpha1 + w_h_k_prev * alpha2
        ##########diff start
        diffusion_time_t = torch.randint(
            low=0, high=self.config.diffusion.num_diffusion_timesteps, size=[aligned_e_t.shape[0], ]).to(
            self.device)
        alpha = (1 - self.betas).cumprod(dim=0).index_select(0, diffusion_time_t).view(-1, 1, 1)
        normal_noise = torch.randn_like(aligned_e_t)
        final_statues_with_noise = aligned_e_t * alpha.sqrt() + normal_noise * (1.0 - alpha).sqrt()
        predicted_noise = self.diffusion(final_statues_with_noise, timesteps=diffusion_time_t)
        noise_loss = normal_noise - predicted_noise
        GEN_x = aligned_e_t + noise_loss
        GEN_rnn_output, _ = self.gru(GEN_x)
        ####### diff end
        # fused_rnn_output = self.fuse(torch.cat([rnn_output, GEN_rnn_output], dim=-1))
        #######fuse gen result with original ones

        weight = self.weight_layer(rnn_output)
        # mask = (torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len) >= lengths.unsqueeze(1))
        # att = torch.softmax(weight.squeeze().masked_fill(mask, -np.inf), dim=1).view(batch_size, seq_len)
        att = torch.softmax(weight.squeeze(), dim=1).view(batch_size, seq_len)
        weighted_features = rnn_output * att.unsqueeze(2)
        averaged_features = torch.sum(weighted_features, 1)
        averaged_features = self.dropout(averaged_features)
        pred = self.output_mlp(averaged_features)

        GEN_weight = self.weight_layer(GEN_rnn_output)
        # GEN_mask = (torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len) >= lengths.unsqueeze(1))
        # GEN_att = torch.softmax(GEN_weight.squeeze().masked_fill(GEN_mask, -np.inf), dim=1).view(batch_size, seq_len)
        GEN_att = torch.softmax(GEN_weight.squeeze(), dim=1).view(batch_size, seq_len)
        GEN_weighted_features = GEN_rnn_output * GEN_att.unsqueeze(2)
        GEN_averaged_features = torch.sum(GEN_weighted_features, 1)
        GEN_averaged_features = self.dropout(GEN_averaged_features)
        GEN_pred = self.output_mlp(GEN_averaged_features)
        return pred, GEN_pred, predicted_noise, normal_noise


class Timeline(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb, num_layers, num_heads, max_pos, device):
        super().__init__()
        self.hidden_dim = d_model
        # self.batchsi = batch_size
        self.word_embeddings = nn.Embedding(vocab_size + 1, d_model, padding_idx=vocab_size)
        self.lstm = nn.LSTM(d_model, d_model, bidirectional=False)
        self.hidden2label = nn.Linear(d_model, 2)
        self.attention = nn.Linear(d_model, d_model)
        self.vector1 = nn.Parameter(torch.randn(d_model, 1))
        self.decay = nn.Parameter(torch.FloatTensor([-0.1] * (vocab_size + 1)))
        self.initial = nn.Parameter(torch.FloatTensor([1.0] * (vocab_size + 1)))
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.attention_dimensionality = d_model
        self.WQ1 = nn.Linear(d_model, d_model, bias=False)
        self.WK1 = nn.Linear(d_model, d_model, bias=False)
        self.embed_drop = nn.Dropout(p=dropout_emb)
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

    def forward(self, input_seqs, masks, lengths, seq_time_step, code_masks):
        sentence = input_seqs, code_masks, seq_time_step
        numcode = sentence[0].size()[2]
        numvisit = sentence[0].size()[1]
        numbatch = sentence[0].size()[0]
        thisembeddings = self.word_embeddings(sentence[0].view(-1, numcode))
        thisembeddings = self.embed_drop(thisembeddings)
        myQ1 = self.WQ1(thisembeddings)
        myK1 = self.WK1(thisembeddings)
        dproduct1 = torch.bmm(myQ1, torch.transpose(myK1, 1, 2)).view(numbatch, numvisit, numcode, numcode)
        dproduct1 = dproduct1 - sentence[1].view(numbatch, numvisit, 1, numcode) - sentence[1].view(numbatch, numvisit,
                                                                                                    numcode, 1)
        sproduct1 = self.softmax(dproduct1.view(-1, numcode) / np.sqrt(self.attention_dimensionality)).view(-1, numcode,
                                                                                                            numcode)
        fembedding11 = torch.bmm(sproduct1, thisembeddings)
        fembedding11 = (((sentence[1] - (1e+20)) / (-1e+20)).view(-1, numcode, 1) * fembedding11)
        mydecay = self.decay[sentence[0].view(-1)].view(numvisit * numbatch, numcode, 1)
        myini = self.initial[sentence[0].view(-1)].view(numvisit * numbatch, numcode, 1)
        temp1 = torch.bmm(mydecay, sentence[2].view(-1, 1, 1))
        temp2 = self.sigmoid(temp1 + myini)
        vv = torch.bmm(temp2.view(-1, 1, numcode), fembedding11)
        vv = vv.view(numbatch, numvisit, -1).transpose(0, 1)
        lstm_out, hidden = self.lstm(vv)
        t_lstm_out = torch.transpose(lstm_out, 0, 1)

        hiddenstate, _ = self.hiddenstate_learner(t_lstm_out)
        aligned_e_t = torch.zeros_like(hiddenstate)
        for i in range(aligned_e_t.shape[1]):
            if i == 0:
                aligned_e_t[:, 0:1,:] = t_lstm_out[:, 0:1, :]
            else:
                e_k = t_lstm_out[:, 0:1, :]
                w_h_k_prev = self.w_hk(hiddenstate[:,i-1:i,:])
                attn = self.softmax(self.w2(self.tanh(self.w1(torch.cat((e_k,w_h_k_prev),dim=-1)))))
                alpha1 = attn[:,:,0:1]
                alpha2 = attn[:,:,1:2]
                aligned_e_t[:, i:i+1,:] = e_k * alpha1 + w_h_k_prev * alpha2
        ##########diff start
        diffusion_time_t = torch.randint(
            low=0, high=self.config.diffusion.num_diffusion_timesteps, size=[aligned_e_t.shape[0], ]).to(
            self.device)
        alpha = (1 - self.betas).cumprod(dim=0).index_select(0, diffusion_time_t).view(-1, 1, 1)
        normal_noise = torch.randn_like(aligned_e_t)
        final_statues_with_noise = aligned_e_t * alpha.sqrt() + normal_noise * (1.0 - alpha).sqrt()
        predicted_noise = self.diffusion(final_statues_with_noise, timesteps=diffusion_time_t)
        noise_loss = normal_noise - predicted_noise
        GEN_t_lstm_out = aligned_e_t + noise_loss
        ####### diff end
        fused_t_lstm_out = self.fuse(torch.cat([t_lstm_out, GEN_t_lstm_out], dim=-1))
        fused_lstm_out = torch.transpose(fused_t_lstm_out, 0, 1)
        #######fuse gen result with original ones

        mask_final = torch.arange(input_seqs.size(1), device=input_seqs.device).unsqueeze(0).expand(input_seqs.size(0),
                                                                                                    input_seqs.size(
                                                                                                        1)) == lengths.unsqueeze(
            1) - 1
        lstm_out_final = fused_lstm_out * mask_final.float().transpose(0, 1).view(numvisit, numbatch, 1)
        lstm_out_final = lstm_out_final.sum(dim=0)
        # lstm_out_final = self.embed_drop(lstm_out_final)
        label_space = self.hidden2label(lstm_out_final)
        return label_space


class TLSTM(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb, num_layers, num_heads, max_pos, device, initial_d):
        super(TLSTM, self).__init__()
        self.embbedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.W_all = nn.Linear(d_model, d_model * 4)
        self.U_all = nn.Linear(d_model, d_model * 4)
        self.W_d = nn.Linear(d_model, d_model)
        self.d_model = d_model
        self.time_layer = torch.nn.Linear(64, d_model)
        self.selection_layer = torch.nn.Linear(1, 64)
        self.output_mlp = nn.Sequential(nn.Linear(d_model, 2))
        self.pooler = MaxPoolLayer()
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
        self.intial_d = initial_d
        self.embedding2 = nn.Linear(self.intial_d, d_model)


    def forward(self, input_seqs, masks, lengths, seq_time_step, code_masks):
        if self.intial_d != 0:
            x = self.embedding2(input_seqs)
        else:
            x = self.embbedding(input_seqs).sum(dim=2)
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


        aligned_e_t = torch.zeros_like(x)
        for i in range(aligned_e_t.shape[1]):
            if i == 0:
                aligned_e_t[:, 0:1,:] = x[:, 0:1, :]
            else:
                e_k = x[:, 0:1, :]
                w_h_k_prev = self.w_hk(outputs[:,i-1:i,:])
                attn = self.softmax(self.w2(self.tanh(self.w1(torch.cat((e_k,w_h_k_prev),dim=-1)))))
                alpha1 = attn[:,:,0:1]
                alpha2 = attn[:,:,1:2]
                aligned_e_t[:, i:i+1,:] = e_k * alpha1 + w_h_k_prev * alpha2
        ##########diff start
        diffusion_time_t = torch.randint(
            low=0, high=self.config.diffusion.num_diffusion_timesteps, size=[aligned_e_t.shape[0], ]).to(
            self.device)
        alpha = (1 - self.betas).cumprod(dim=0).index_select(0, diffusion_time_t).view(-1, 1, 1)
        normal_noise = torch.randn_like(aligned_e_t)
        final_statues_with_noise = aligned_e_t * alpha.sqrt() + normal_noise * (1.0 - alpha).sqrt()
        predicted_noise = self.diffusion(final_statues_with_noise, timesteps=diffusion_time_t)
        noise_loss = normal_noise - predicted_noise
        GEN_x = aligned_e_t + noise_loss
        ####### diff end
        # fused_outputs = self.fuse(torch.cat([outputs, GEN_outputs], dim=-1))
        #######fuse gen result with original ones

        Gen_h = torch.zeros(b, self.d_model, requires_grad=False).to(x.device)
        Gen_c = torch.zeros(b, self.d_model, requires_grad=False).to(x.device)
        Gen_outputs = []
        for s in range(seq):
            Gen_c_s1 = torch.tanh(self.W_d(Gen_c))
            Gen_c_s2 = Gen_c_s1 * time_feature[:, s]
            Gen_c_l = Gen_c - Gen_c_s1
            Gen_c_adj = Gen_c_l + Gen_c_s2
            Gen_outs = self.W_all(Gen_h) + self.U_all(GEN_x[:, s])
            # print(outs.size())
            Gen_f, Gen_i, Gen_o, Gen_c_tmp = torch.chunk(Gen_outs, 4, 1)
            Gen_f = torch.sigmoid(Gen_f)
            Gen_i = torch.sigmoid(Gen_i)
            Gen_o = torch.sigmoid(Gen_o)
            Gen_c_tmp = torch.sigmoid(Gen_c_tmp)
            Gen_c = Gen_f * Gen_c_adj + Gen_i * Gen_c_tmp
            Gen_h = Gen_o * torch.tanh(Gen_c)
            Gen_outputs.append(Gen_h)
        Gen_outputs = torch.stack(Gen_outputs, 1)
        out = self.pooler(outputs)
        Gen_out = self.pooler(Gen_outputs)
        out = self.output_mlp(out)
        Gen_out = self.output_mlp(Gen_out)
        return out, Gen_out, predicted_noise, normal_noise

class SAND(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb, num_layers, num_heads, max_pos, device, initial_d=0):
        super().__init__()
        self.embbedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        # self.emb_dropout = nn.Dropout(dropout_emb)
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
        self.selfatt = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.initial_d = initial_d
        self.embedding2 = nn.Linear(self.initial_d, d_model)

    def forward(self, input_seqs, masks, lengths, seq_time_step, code_masks):
        if self.initial_d != 0:
            x = self.embedding2(input_seqs)
        else:
            x = self.embbedding(input_seqs).sum(dim=2) + self.bias_embedding
        bs, sl, dm = x.size()
        # x = self.emb_dropout(x)
        # output_pos, ind_pos = self.pos_emb(lengths)
        # x += output_pos
        x, _ = self.selfatt(x, x, x)

        hiddenstate, _ = self.hiddenstate_learner(x)
        aligned_e_t = torch.zeros_like(hiddenstate)
        for i in range(aligned_e_t.shape[1]):
            if i == 0:
                aligned_e_t[:, 0:1,:] = x[:, 0:1, :]
            else:
                e_k = x[:, 0:1, :]
                w_h_k_prev = self.w_hk(hiddenstate[:,i-1:i,:])
                attn = self.softmax(self.w2(self.tanh(self.w1(torch.cat((e_k,w_h_k_prev),dim=-1)))))
                alpha1 = attn[:,:,0:1]
                alpha2 = attn[:,:,1:2]
                aligned_e_t[:, i:i+1,:] = e_k * alpha1 + w_h_k_prev * alpha2
        ##########diff start
        diffusion_time_t = torch.randint(
            low=0, high=self.config.diffusion.num_diffusion_timesteps, size=[aligned_e_t.shape[0], ]).to(
            self.device)
        alpha = (1 - self.betas).cumprod(dim=0).index_select(0, diffusion_time_t).view(-1, 1, 1)
        normal_noise = torch.randn_like(aligned_e_t)
        final_statues_with_noise = aligned_e_t * alpha.sqrt() + normal_noise * (1.0 - alpha).sqrt()
        predicted_noise = self.diffusion(final_statues_with_noise, timesteps=diffusion_time_t)
        noise_loss = normal_noise - predicted_noise
        GEN_x = aligned_e_t + noise_loss
        ####### diff end
        # fused_x = self.fuse(torch.cat([x, GEN_x], dim=-1))
        #######fuse gen result with original ones

        x = self.drop_out(x)
        output = self.out_layer(x.sum(-2))

        GEN_x = self.drop_out(GEN_x)
        GEN_output = self.out_layer(GEN_x.sum(-2))

        return output, GEN_output, predicted_noise, normal_noise


class Retain(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb, num_layers, num_heads, max_pos, device, initial_d=0):
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

        self.visit_hidden_size = d_model
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
        self.initial_d = initial_d
        self.embedding2 = nn.Linear(self.initial_d, d_model)

    def forward(self, input_seqs, masks, lengths, seq_time_step, code_masks):
        if self.initial_d != 0:
            x = self.embedding2(input_seqs)
        else:
            x = self.embbedding(input_seqs).sum(dim=2)
        x = self.dropout(x)
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


        aligned_e_t_1 = torch.zeros_like(x)
        for i in range(aligned_e_t_1.shape[1]):
            if i == 0:
                aligned_e_t_1[:, 0:1,:] = x[:, 0:1, :]
            else:
                e_k_1 = x[:, 0:1, :]
                w_h_k_prev_1 = self.w_hk(visit_rnn_output[:,i-1:i,:])
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
        GEN_x1 = aligned_e_t_1 + noise_loss_1
        ####### diff end
        #######fuse gen result with original ones

        aligned_e_t_2 = torch.zeros_like(x)
        for i in range(aligned_e_t_2.shape[1]):
            if i == 0:
                aligned_e_t_2[:, 0:1,:] = x[:, 0:1, :]
            else:
                e_k_2 = x[:, 0:1, :]
                w_h_k_prev_2 = self.w_hk(var_rnn_output[:,i-1:i,:])
                attn_2 = self.softmax(self.w2(self.tanh(self.w1(torch.cat((e_k_2,w_h_k_prev_2),dim=-1)))))
                alpha1_2 = attn_2[:,:,0:1]
                alpha2_2 = attn_2[:,:,1:2]
                aligned_e_t_2[:, i:i+1,:] = e_k_2 * alpha1_2 + w_h_k_prev_2 * alpha2_2
        ##########diff start
        diffusion_time_t_2 = torch.randint(
            low=0, high=self.config.diffusion.num_diffusion_timesteps, size=[aligned_e_t_2.shape[0], ]).to(
            self.device)
        alpha_2 = (1 - self.betas).cumprod(dim=0).index_select(0, diffusion_time_t_2).view(-1, 1, 1)
        normal_noise_2 = torch.randn_like(aligned_e_t_2)
        final_statues_with_noise_2 = aligned_e_t_2 * alpha_2.sqrt() + normal_noise_2 * (1.0 - alpha_2).sqrt()
        predicted_noise_2 = self.diffusion(final_statues_with_noise_2, timesteps=diffusion_time_t_2)
        noise_loss_2 = normal_noise_2 - predicted_noise_2
        GEN_x2 = aligned_e_t_2 + noise_loss_2
        ####### diff end
        #######fuse gen result with original ones

        GEN_visit_rnn_output, _ = self.visit_level_rnn(GEN_x1)
        Gen_alpha = self.visit_level_attention(GEN_visit_rnn_output)
        Gen_visit_attn_w = torch.softmax(Gen_alpha, dim=1)
        GEN_var_rnn_output, _ = self.variable_level_rnn(GEN_x2)
        Gen_beta = self.variable_level_attention(GEN_var_rnn_output)
        Gen_var_attn_w = torch.tanh(Gen_beta)
        Gen_attn_w = Gen_visit_attn_w * Gen_var_attn_w
        Gen_c_all = Gen_attn_w * GEN_x1
        Gen_c = torch.sum(Gen_c_all, dim=1)
        Gen_c = self.output_dropout(Gen_c)

        output = self.output_layer(c)
        Gen_output = self.output_layer(Gen_c)
        return output, Gen_output, predicted_noise_1+predicted_noise_2, normal_noise_1+normal_noise_2


class RetainEx(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb, num_layers, num_heads, max_pos, device, initial_d = 0):
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
        self.initial_d = initial_d
        self.linearembedding = nn.Linear(self.initial_d, d_model)

    def forward(self, input_seqs, masks, lengths, time_step, code_masks):
        if self.initial_d != 0:
            embedded = self.linearembedding(input_seqs)
            embedded2 = self.linearembedding(input_seqs)
        else:
            embedded = self.embbedding1(input_seqs).sum(dim=2)
            embedded2 = self.embbedding2(input_seqs).sum(dim=2)

        b, seq, features = embedded.size()
        dates = torch.stack([time_step, 1 / (time_step + 1), 1 / torch.log(np.e + time_step)], 2)  # [b x seq x 3]
        og_embedded = embedded.clone()
        embedded = torch.cat([embedded, dates], 2)
        outputs1 = self.RNN1(embedded)[0]
        outputs2 = self.RNN2(embedded)[0]
        # print(outputs2.size())

        aligned_e_t = torch.zeros_like(og_embedded)
        for i in range(aligned_e_t.shape[1]):
            if i == 0:
                aligned_e_t[:, 0:1,:] = og_embedded[:, 0:1, :]
            else:
                e_k = og_embedded[:, 0:1, :]
                w_h_k_prev = self.w_hk(outputs1[:,i-1:i,:])
                attn = self.softmax(self.w2(self.tanh(self.w1(torch.cat((e_k,w_h_k_prev),dim=-1)))))
                alpha1 = attn[:,:,0:1]
                alpha2 = attn[:,:,1:2]
                aligned_e_t[:, i:i+1,:] = e_k * alpha1 + w_h_k_prev * alpha2
        ##########diff start
        diffusion_time_t = torch.randint(
            low=0, high=self.config.diffusion.num_diffusion_timesteps, size=[aligned_e_t.shape[0], ]).to(
            self.device)
        alpha = (1 - self.betas).cumprod(dim=0).index_select(0, diffusion_time_t).view(-1, 1, 1)
        normal_noise = torch.randn_like(aligned_e_t)
        final_statues_with_noise = aligned_e_t * alpha.sqrt() + normal_noise * (1.0 - alpha).sqrt()
        predicted_noise = self.diffusion(final_statues_with_noise, timesteps=diffusion_time_t)
        noise_loss = normal_noise - predicted_noise
        GEN_embedded = aligned_e_t + noise_loss
        ####### diff end
        GEN_embedded = torch.cat([GEN_embedded, dates], 2)

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

        return outputs, GEN_outputs, predicted_noise, normal_noise


def build_tree(treeFile):
    treeMap = pickle.load(open(treeFile, 'rb'))
    ancestors = np.array(list(treeMap.values())).astype('int32')
    ancSize = ancestors.shape[1]
    leaves = []
    for k in treeMap.keys():
        leaves.append([k] * ancSize)
    leaves = np.array(leaves).astype('int32')
    return leaves, ancestors


class Gram(nn.Module):
    def __init__(self, inputDimSize, numAncestors, d_model, dropout, num_layers, treeFile, device):
        super().__init__()
        self.inputDimSize = inputDimSize
        self.W_emb = nn.Embedding(inputDimSize + numAncestors, d_model)
        self.att_MLP = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1, bias=False)
        )
        self.softmax = nn.Softmax(1)
        self.rnns = nn.LSTM(d_model, d_model, num_layers, bidirectional=False, batch_first=True, dropout=dropout)
        self.pooler = MaxPoolLayer()
        self.output_mlp = nn.Linear(d_model, 2)
        self.input_dropout = nn.Dropout(0.1)
        leavesList = []
        ancestorsList = []
        for i in range(5, 0, -1):
            leaves, ancestors = build_tree(treeFile + '.level' + str(i) + '.pk')
            sharedLeaves = torch.LongTensor(leaves).to(device)
            sharedAncestors = torch.LongTensor(ancestors).to(device)
            leavesList.append(sharedLeaves)
            ancestorsList.append(sharedAncestors)
        self.leavesList = leavesList
        self.ancestorsList = ancestorsList

    def forward(self, x, lengths):
        embList = []
        for leaves, ancestors in zip(self.leavesList, self.ancestorsList):
            attentionInput = torch.cat((self.W_emb(leaves), self.W_emb(ancestors)), dim=2)
            preAttention = self.att_MLP(attentionInput)
            attention = self.softmax(preAttention)
            tempEmb = self.W_emb(ancestors) * attention
            tempEmb = torch.sum(tempEmb, dim=1)
            embList.append(tempEmb)
        emb = torch.cat(embList, dim=0)
        pad_emb = emb.new_zeros((1, emb.size(1)))
        emb = torch.cat((emb, pad_emb), dim=0)
        assert (lengths > 0).all()
        assert emb.size(0) == self.inputDimSize + 1
        bz, seq_len, num_per_visit = x.size()
        x = emb[x]
        x = self.input_dropout(x)
        x = torch.sum(x, dim=2)
        rnn_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        rnn_output, _ = self.rnns(rnn_input)
        x, _ = pad_packed_sequence(rnn_output, batch_first=True, total_length=seq_len)
        x = self.pooler(x, lengths)
        x = self.output_mlp(x)
        return x

class Gram(nn.Module):
    def __init__(self, inputDimSize, numAncestors, d_model, dropout, num_layers, treeFile, device):
        super().__init__()
        self.inputDimSize = inputDimSize
        self.W_emb = nn.Embedding(inputDimSize + numAncestors, d_model)
        self.att_MLP = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1, bias=False)
        )
        self.softmax = nn.Softmax(1)
        self.rnns = nn.LSTM(d_model, d_model, num_layers, bidirectional=False, batch_first=True, dropout=dropout)
        self.pooler = MaxPoolLayer()
        self.output_mlp = nn.Linear(d_model, 2)
        self.input_dropout = nn.Dropout(0.1)
        leavesList = []
        ancestorsList = []
        for i in range(5, 0, -1):
            leaves, ancestors = build_tree(treeFile + '.level' + str(i) + '.pk')
            sharedLeaves = torch.LongTensor(leaves).to(device)
            sharedAncestors = torch.LongTensor(ancestors).to(device)
            leavesList.append(sharedLeaves)
            ancestorsList.append(sharedAncestors)
        self.leavesList = leavesList
        self.ancestorsList = ancestorsList

    def forward(self, x, lengths):
        embList = []
        for leaves, ancestors in zip(self.leavesList, self.ancestorsList):
            attentionInput = torch.cat((self.W_emb(leaves), self.W_emb(ancestors)), dim=2)
            preAttention = self.att_MLP(attentionInput)
            attention = self.softmax(preAttention)
            tempEmb = self.W_emb(ancestors) * attention
            tempEmb = torch.sum(tempEmb, dim=1)
            embList.append(tempEmb)
        emb = torch.cat(embList, dim=0)
        pad_emb = emb.new_zeros((1, emb.size(1)))
        emb = torch.cat((emb, pad_emb), dim=0)
        assert (lengths > 0).all()
        assert emb.size(0) == self.inputDimSize + 1
        bz, seq_len, num_per_visit = x.size()
        x = emb[x]
        x = self.input_dropout(x)
        x = torch.sum(x, dim=2)
        rnn_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        rnn_output, _ = self.rnns(rnn_input)
        x, _ = pad_packed_sequence(rnn_output, batch_first=True, total_length=seq_len)
        x = self.pooler(x, lengths)
        x = self.output_mlp(x)
        return x

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

if __name__ == '__main__':
    y_true = np.array([])
    print(len(y_true))