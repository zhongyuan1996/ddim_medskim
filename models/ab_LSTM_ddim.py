import math
import torch
import numpy as np
import torch.nn as nn
from models.diffusion import diffModel
from utils.diffUtil import get_beta_schedule
from models.unet import *
torch.autograd.set_detect_anomaly(True)
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
        # if self.temperature == 'none':
        #     h = self.softmax(h)
        # elif self.temperature == 'temperature':
        #
        #     h = self.softmax(h/self.tau)
        #
        # elif self.temperature == 'gumbel':
        #     h = nn.functional.gumbel_softmax(h, tau=self.tau)

        return h
class SelfAttention(nn.Module):
    def __init__(self, in_feature, num_head=4, dropout=0.1):
        super(SelfAttention, self).__init__()
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

    def forward(self, x, y, attn_mask, lengths):
        batch_size = x.size(0)
        res = x
        query = self.q_linear(x)
        key = self.k_linear(y)
        value = self.v_linear(y)

        query = query.view(batch_size, self.num_head, -1, self.size_per_head)
        key = key.view(batch_size, self.num_head, -1, self.size_per_head)
        value = value.view(batch_size, self.num_head, -1, self.size_per_head)

        scale = np.sqrt(self.size_per_head)
        energy = torch.matmul(query, key.permute(0, 1, 3, 2)) / scale

        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(attention, value)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.in_feature)
        x = self.fc(x)
        x = self.dropout(x)
        x += res
        x = self.layer_norm(x)
        return x

class RNNdiff(nn.Module):

    def __init__(self, config, vocab_size, d_model, h_model, dropout, dropout_emb, device):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.device = device
        self.h_model = h_model
        self.model_var_type = self.config.model.var_type
        self.initial_embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.cross_attention = nn.MultiheadAttention(d_model, 8, batch_first=False)
        self.cross_attention_alt = SelfAttention(d_model)
        # CrossAttentionBlock(d_model, 2, drop=0.1, attn_drop=0.1)
        self.lstm = nn.LSTM(d_model, h_model, num_layers=1, batch_first=False, dropout=dropout)

        # self.diffusion = diffModel(self.config)
        self.diffusion = UNetModel(in_channels=50, model_channels=128,
                                   out_channels=50, num_res_blocks=2,
                                   attention_resolutions=[16, ])
        betas = get_beta_schedule(beta_schedule=self.config.diffusion.beta_schedule,
                                  beta_start=self.config.diffusion.beta_start,
                                  beta_end=self.config.diffusion.beta_end,
                                  num_diffusion_timesteps=self.config.diffusion.num_diffusion_timesteps)
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.diffusion_num_timesteps = betas.shape[0]

        # alphas = 1.0 - betas
        # alphas_cumprod = alphas.cumprod(dim=0)
        # alphas_cumprod_prev = torch.cat(
        #     [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        # )
        # posterior_variance = (
        #         betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        # )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        self.classifyer = classifyer(h_model)
        self.embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.target_embedding = nn.Embedding(1, d_model)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.time_layer = nn.Linear(1, 64)
        self.time_updim = nn.Linear(64, d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

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


    def forward(self, input_seqs, seq_time_step):
        batch_size, visit_size, icd_code_size = input_seqs.size()

        visit_embedding = self.before(input_seqs, seq_time_step)

        e_t_prime_all = torch.zeros_like(visit_embedding)
        E_gen_t_prime_all = torch.zeros_like(e_t_prime_all)
        #ht and et'
        hidden_state_all_visit = torch.zeros(batch_size, visit_size, self.h_model).to(self.device)
        hidden_state_all_visit_generated = torch.zeros(batch_size, visit_size, self.h_model).to(self.device)

        c_all_visit = torch.zeros(batch_size, visit_size, self.h_model).to(self.device)
        c_all_visit_generated = torch.zeros(batch_size, visit_size, self.h_model).to(self.device)

        hidden_state_softmax_res = torch.zeros(batch_size, visit_size, 2).to(self.device)
        hidden_state_softmax_res_generated = torch.zeros(batch_size, visit_size, 2).to(self.device)

        # seq_h = visit_embedding.new_zeros((1,batch_size, self.h_model))
        # #size seq_h = [1,32,256] here
        # seq_c = visit_embedding.new_zeros((1,batch_size, self.h_model))
        #
        # seq_h_gen = visit_embedding.new_zeros((1,batch_size, self.h_model))
        # seq_c_gen = visit_embedding.new_zeros((1,batch_size, self.h_model))

        for i in range(visit_size):
            e_t = visit_embedding[:, i:i + 1, :].permute(1, 0, 2)
            # ablation:no e_t_prime
            # e_t_prime, _ = self.cross_attention(seq_h.clone(), e_t, e_t)
            if i == 0:
                e_t_prime = e_t.clone()
            else:
                attenOut, _ = self.cross_attention(e_t.clone(), hidden_state_all_visit[:, 0:i, :].clone().permute(1, 0, 2), hidden_state_all_visit[:, 0:i, :].clone().permute(1, 0, 2))
                attenOut = self.fc(attenOut)
                attenOut = self.dropout(attenOut)
                e_t += attenOut
                e_t_prime = self.layer_norm(e_t.clone())

                # e_t_prime = self.cross_attention_alt(e_t, )

            e_t_prime_all[:, i:i + 1, :] = e_t_prime.permute(1, 0, 2)

            _, (seq_h, seq_c) = self.lstm(e_t_prime.clone(),
                                          (hidden_state_all_visit[:, i:i+1, :].clone().permute(1, 0, 2), c_all_visit[:, i:i+1, :].clone().permute(1, 0, 2)))
            # _, (seq_h, seq_c) = self.lstm(e_t,
            #                               (seq_h.clone(), seq_c.clone()))
            hidden_state_all_visit[:, i:i + 1, :] = seq_h.permute(1, 0, 2)
            c_all_visit[:, i:i + 1, :] = seq_c.permute(1, 0, 2)
        #
        # ##########diff start
        # diffusion_time_t = torch.randint(
        #     low=0, high=self.config.diffusion.num_diffusion_timesteps, size=[visit_embedding.shape[0], ]).to(
        #     self.device)
        #
        # alpha = (1 - self.betas).cumprod(dim=0).index_select(0, diffusion_time_t).view(-1, 1, 1)
        #
        # normal_noise = torch.randn_like(e_t_prime_all)
        #
        # e_t_prime_b_first_with_noise = e_t_prime_all * alpha.sqrt() + normal_noise * (1.0 - alpha).sqrt()
        #
        # predicted_noise = self.diffusion(e_t_prime_b_first_with_noise, timesteps=diffusion_time_t)
        #
        # noise_loss = normal_noise - predicted_noise
        #
        # GEN_E_t = e_t_prime_all + noise_loss
        # ####### diff end
        #
        # for i in range(visit_size):
        #     E_gen_t = GEN_E_t[:, i:i + 1, :].permute(1, 0, 2)
        #     # ablation:no e_t_prime
        #     # e_t_prime, _ = self.cross_attention(seq_h.clone(), e_t, e_t)
        #     if i == 0:
        #         E_gen_t_prime = E_gen_t.clone()
        #     else:
        #         gen_attenOut, _ = self.cross_attention(E_gen_t.clone(), hidden_state_all_visit_generated[:, 0:i, :].clone().permute(1, 0, 2), hidden_state_all_visit_generated[:, 0:i, :].clone().permute(1, 0, 2))
        #         gen_attenOut = self.fc(gen_attenOut)
        #         gen_attenOut = self.dropout(gen_attenOut)
        #         E_gen_t += gen_attenOut
        #         E_gen_t_prime = self.layer_norm(E_gen_t.clone())
        #     E_gen_t_prime_all[:, i:i + 1, :] = E_gen_t_prime.permute(1, 0, 2)
        #
        #     _, (seq_h, seq_c) = self.lstm(E_gen_t_prime.clone(),
        #                                   (hidden_state_all_visit_generated[:, i:i+1, :].clone().permute(1, 0, 2), c_all_visit_generated[:, i:i+1, :].clone().permute(1, 0, 2)))
        #     # _, (seq_h, seq_c) = self.lstm(e_t,
        #     #                               (seq_h.clone(), seq_c.clone()))
        #     hidden_state_all_visit_generated[:, i:i + 1, :] = seq_h.permute(1, 0, 2)
        #     c_all_visit_generated[:, i:i + 1, :] = seq_c.permute(1, 0, 2)

            # #```new stuff that generate Et_prime from Et```
            # E_gen_t = GEN_E_t[:, i:i + 1, :].permute(1, 0, 2)
            # E_gen_t_prime, _ = self.cross_attention(seq_h_gen.clone(), E_gen_t, E_gen_t)
            # E_t_prime_all[:, i:i + 1, :] = E_gen_t_prime.permute(1, 0, 2)
            # _, (seq_h_gen, seq_c_gen) = self.lstm(E_gen_t_prime,
            #                               (seq_h_gen.clone(), seq_c_gen.clone()))
            #
            #
            # # _, (seq_h_gen, seq_c_gen) = self.lstm(GEN_E_t[:, i:i + 1, :].permute(1, 0, 2),
            # #                               (seq_h_gen.clone(), seq_c_gen.clone()))
            # hidden_state_all_visit_generated[:, i:i + 1, :] = seq_h_gen.permute(1, 0, 2)

        for i in range(visit_size):

            hidden_state_softmax_res[:, i:i+1, :] = self.classifyer(hidden_state_all_visit[:, i:i + 1, :])
            # hidden_state_softmax_res_generated[:, i:i+1, :] = self.classifyer(hidden_state_all_visit_generated[:, i:i + 1, :])

        final_prediction = hidden_state_softmax_res[:, -1, :]
        # final_prediction_generated = hidden_state_softmax_res_generated[:, -1, :]

        #
        # return hidden_state_softmax_res, hidden_state_softmax_res_generated, \
        #        final_prediction, final_prediction_generated, \
        #        normal_noise, predicted_noise


        return hidden_state_softmax_res, hidden_state_softmax_res_generated, \
               final_prediction


