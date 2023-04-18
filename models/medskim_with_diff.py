import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
import yaml
from models.unet import *
from utils.diffUtil import get_beta_schedule

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

class Selected(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, dropout_emb, device, max_len):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.target_embedding = nn.Embedding(1, d_model)
        self.target_linear = nn.Linear(d_model, d_model)
        self.target_linear2 = nn.Linear(d_model, d_model)
        self.code_selection = nn.Sequential(nn.Linear(2*d_model, d_model), nn.Tanh(), nn.Linear(d_model, 2))
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.rnn_reverse = nn.LSTM(d_model, d_model // 8, 1, bias=False, batch_first=True)
        self.visit_skip = nn.Sequential(nn.Linear(3 * d_model + d_model // 8, d_model), nn.Tanh(), nn.Linear(d_model, 2))
        self.code_query = nn.Sequential(nn.Linear(2*d_model + d_model // 8, d_model), nn.ReLU())
        self.code_gate = nn.Linear(1, 2)
        # self.global_query = nn.Sequential(nn.Linear(2*d_model, d_model), nn.ReLU())
        self.output_mlp = nn.Sequential(nn.Linear(d_model, 2))
        self.layer_norm = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(d_model, d_model)
        self.feed_forward = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.ReLU(), nn.Linear(4*d_model, d_model), nn.Dropout(dropout))
        # self.feed_forward2 = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.ReLU(), nn.Linear(4*d_model, d_model))
        self.linear = nn.Linear(d_model, d_model)
        self.rnn_cell = nn.GRUCell(d_model, d_model, bias=False)
        self.time_layer = nn.Linear(1, 64)
        self.time_layer2 = nn.Linear(1, 64)
        self.time_updim = nn.Linear(64, d_model)
        self.mem_linear = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.LayerNorm(d_model))
        self.step_linear = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.LayerNorm(d_model))
        self.global_att = nn.Sequential(nn.Linear(2 * d_model + 64, d_model), nn.Tanh(), nn.Linear(d_model, 1))
        #####################################
        self.diff_channel = max_len
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

    def forward(self, input_seqs, lengths, seq_time_step, code_mask):
        seq_time_step = seq_time_step.unsqueeze(2) / 180
        time_feature = 1 - self.tanh(torch.pow(self.time_layer(seq_time_step), 2))
        time_encoding = self.time_updim(time_feature)
        batch_size, seq_len, num_cui_per_visit = input_seqs.size()
        x = self.embedding(input_seqs)
        x = self.emb_dropout(x)
        x = self.relu(x)
        cat_features = torch.cat((self.embedding(torch.arange(0, self.vocab_size+1).to(x.device)), self.target_embedding(x.new_zeros(self.vocab_size + 1).long())), dim=-1)
        p = self.code_selection(cat_features)
        p = F.gumbel_softmax(torch.log_softmax(p, -1), hard=True)
        p_mask = p[:, 0].scatter_(dim=0, index=torch.LongTensor([self.vocab_size]).to(x.device), src=torch.zeros(1).to(x.device))[input_seqs]
        x_selected = p_mask.unsqueeze(-1) * x
        selected = x_selected.sum(-2) + time_encoding
        following_feature, _ = self.rnn_reverse(selected.flip([1]))
        following_feature = torch.flip(following_feature, [1])
        hx = x.new_zeros((batch_size, self.rnn_cell.hidden_size))
        hiddens = []
        v_skips = []
        c_skips = []
        for i in range(seq_len):
            v_skip = self.visit_skip(torch.cat((hx, selected[:, i], following_feature[:, i], self.target_linear2(self.target_embedding(x.new_zeros(batch_size).long()))), dim=-1))
            v_skip = F.gumbel_softmax(torch.log_softmax(v_skip, dim=-1), hard=True)
            # print(v_skip.size())
            v_skips.append(v_skip)
            code_query = self.code_query(torch.cat((hx, following_feature[:, i], self.target_linear(self.target_embedding(x.new_zeros(batch_size).long()))), dim=-1))
            scale = np.sqrt(self.d_model)
            energy = (torch.bmm(code_query.unsqueeze(1), x[:, i].permute(0, 2, 1)) / scale)
            attention = torch.softmax(energy, dim=-1) * p_mask[:, i].unsqueeze(1)
            c_skips.append(attention.squeeze())
            z_i = torch.matmul(attention, x[:, i]).squeeze()
            z_i = self.fc(z_i) + time_encoding[:, i]
            z_i = self.dropout2(z_i)
            # z_i = self.layer_norm(z_i)
            step_hx = self.rnn_cell(z_i, hx)
            step_hx = hx * v_skip[:, 1].unsqueeze(-1) + step_hx * v_skip[:, 0].unsqueeze(-1)
            # cx = cx * v_skip[:, 1].unsqueeze(-1) + step_cx * v_skip[:, 0].unsqueeze(-1)
            if i == 0:
                hx = step_hx
            else:
                memory = self.mem_linear(torch.stack(hiddens, dim=1))
                step_hx = self.step_linear(step_hx)
                mem_energy = torch.bmm(step_hx.unsqueeze(1), memory.permute(0, 2, 1)) / scale
                mem_attention = torch.softmax(mem_energy, dim=-1)
                hx = torch.matmul(mem_attention, memory).squeeze()
                hx = self.dropout(hx)
                hx = self.layer_norm2(hx + step_hx)
                hx = self.layer_norm(self.feed_forward(hx) + hx)
            hiddens.append(hx)
            # skips.append(v_skip[:, 1])
        hiddens = torch.stack(hiddens, dim=1)

        hiddenstate, _ = self.hiddenstate_learner(hiddens)
        aligned_e_t = torch.zeros_like(hiddenstate)
        for i in range(aligned_e_t.shape[1]):
            if i == 0:
                aligned_e_t[:, 0:1,:] = hiddens[:, 0:1, :]
            else:
                e_k = hiddens[:, 0:1, :]
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
        GEN_hiddens = aligned_e_t + noise_loss
        ####### diff end
        fused_hiddens = self.fuse(torch.cat([hiddens, GEN_hiddens], dim=-1))
        #######fuse gen result with original ones

        v_skips = torch.stack(v_skips, dim=1)
        c_skips = torch.stack(c_skips, dim=1)
        one = torch.ones_like(c_skips)
        c_skips = torch.where(c_skips > 0, one, c_skips)
        code_mask = torch.where(code_mask > 0, one, code_mask)
        skips = v_skips[:, :, 0].unsqueeze(-1) * c_skips
        skip_rate = 1 - torch.divide(torch.sum(skips), torch.sum(1 - code_mask))
        length_mask = (torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len) >= lengths.unsqueeze(1))
        # hiddens = hiddens.masked_fill(length_mask.unsqueeze(-1).expand_as(hiddens), float('-inf'))
        time_feature2 = 1 - self.tanh(torch.pow(self.time_layer2(seq_time_step), 2))
        global_energy = self.global_att(torch.cat((fused_hiddens, time_feature2, self.target_embedding(x.new_zeros(batch_size, seq_len).long())), dim=-1))
        global_energy = global_energy.masked_fill(length_mask.unsqueeze(-1).expand_as(global_energy), float('-inf'))
        global_attention = torch.softmax(global_energy, dim=1)
        out_feature = global_attention * fused_hiddens
        # final_hidden = torch.gather(hiddens, 1, lengths.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, self.rnn_cell.hidden_size) - 1).squeeze()
        out = self.output_mlp(out_feature.sum(1))
        return out, skip_rate

    def infer(self, input_seqs, lengths, seq_time_step):
        seq_time_step = seq_time_step.unsqueeze(2) / 180
        time_feature = 1 - self.tanh(torch.pow(self.time_layer(seq_time_step), 2))
        time_encoding = self.time_updim(time_feature)
        batch_size, seq_len, num_cui_per_visit = input_seqs.size()
        x = self.embedding(input_seqs)
        x = self.emb_dropout(x)
        x = self.relu(x)
        cat_features = torch.cat((self.embedding(torch.arange(0, self.vocab_size + 1).to(x.device)),
                                  self.target_embedding(x.new_zeros(self.vocab_size + 1).long())), dim=-1)
        p = self.code_selection(cat_features)
        p = F.gumbel_softmax(torch.log_softmax(p, -1), hard=True)
        p_mask = p[:, 0].scatter_(dim=0, index=torch.LongTensor([self.vocab_size]).to(x.device), src=torch.zeros(1).to(x.device))[input_seqs]
        x_selected = p_mask.unsqueeze(-1) * x
        selected = x_selected.sum(-2) + time_encoding
        following_feature, _ = self.rnn_reverse(selected.flip([1]))
        following_feature = torch.flip(following_feature, [1])
        hx = x.new_zeros((batch_size, self.rnn_cell.hidden_size))
        hiddens = []
        v_skips = []
        c_skips = []
        for i in range(seq_len):
            v_skip = self.visit_skip(torch.cat((hx, selected[:, i], following_feature[:, i], self.target_linear2(
                self.target_embedding(x.new_zeros(batch_size).long()))), dim=-1))
            v_skip = F.gumbel_softmax(torch.log_softmax(v_skip, dim=-1), hard=True)
            # print(v_skip.size())
            v_skips.append(v_skip)
            code_query = self.code_query(torch.cat((hx, following_feature[:, i], self.target_linear(
                self.target_embedding(x.new_zeros(batch_size).long()))), dim=-1))
            scale = np.sqrt(self.d_model)
            energy = torch.bmm(code_query.unsqueeze(1), x[:, i].permute(0, 2, 1)) / scale
            attention = torch.softmax(energy, dim=-1) * p_mask[:, i].unsqueeze(1)
            # zero = torch.zeros_like(attention)
            # attention = torch.where(attention < 0.1, zero, attention)
            # print(attention.size())
            c_skips.append(attention.squeeze())
            # attention = self.gumbel_softmax_sample(attention)
            z_i = torch.matmul(attention, x[:, i]).squeeze()
            z_i = self.fc(z_i) + time_encoding[:, i]
            z_i = self.dropout2(z_i)
            # z_i = self.layer_norm(z_i)
            step_hx = self.rnn_cell(z_i, hx)
            step_hx = hx * v_skip[:, 1].unsqueeze(-1) + step_hx * v_skip[:, 0].unsqueeze(-1)
            # cx = cx * v_skip[:, 1].unsqueeze(-1) + step_cx * v_skip[:, 0].unsqueeze(-1)
            if i == 0:
                hx = step_hx
            else:
                memory = self.mem_linear(torch.stack(hiddens, dim=1))
                step_hx = self.step_linear(step_hx)
                mem_energy = torch.bmm(step_hx.unsqueeze(1), memory.permute(0, 2, 1)) / scale
                mem_attention = torch.softmax(mem_energy, dim=-1)
                hx = torch.matmul(mem_attention, memory).squeeze()
                hx = self.dropout(hx)
                hx = self.layer_norm2(hx + step_hx)
                hx = self.layer_norm(self.feed_forward(hx) + hx)
            hiddens.append(hx)
            # skips.append(v_skip[:, 1])
        hiddens = torch.stack(hiddens, dim=1)
        v_skips = torch.stack(v_skips, dim=1)
        c_skips = torch.stack(c_skips, dim=1)
        # skips = torch.stack(skips, dim=1)
        length_mask = (torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size,
                                                                                  seq_len) >= lengths.unsqueeze(1))
        # hiddens = hiddens.masked_fill(length_mask.unsqueeze(-1).expand_as(hiddens), float('-inf'))
        time_feature2 = 1 - self.tanh(torch.pow(self.time_layer2(seq_time_step), 2))
        global_energy = self.global_att(
            torch.cat((hiddens, time_feature2, self.target_embedding(x.new_zeros(batch_size, seq_len).long())), dim=-1))
        global_energy = global_energy.masked_fill(length_mask.unsqueeze(-1).expand_as(global_energy), float('-inf'))
        global_attention = torch.softmax(global_energy, dim=1)
        out_feature = global_attention * hiddens
        # final_hidden = torch.gather(hiddens, 1, lengths.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, self.rnn_cell.hidden_size) - 1).squeeze()
        out = self.output_mlp(out_feature.sum(1))
        return out, v_skips, c_skips, p[:, 0]


if __name__ == '__main__':
    model = Selected(60, 32, 0.1, 0.1, 0.0)
    input = torch.randint(60, (16, 20, 5))
    length = torch.randint(low=1, high=20, size=(16,))
    time_step = torch.randint(50, size=(16, 20))
    out = model(input, length, time_step, None)
    print(out.size())
    # print(output.size())
    # print(s)
    # print(torch.arange(0, 3))