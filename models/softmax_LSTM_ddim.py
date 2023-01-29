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

        self.w_hk = nn.Linear(h_model,d_model)
        self.w1 = nn.Linear(2*h_model, 64)
        self.w2 = nn.Linear(64,2)


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


        for i in range(visit_size):
            e_t = visit_embedding[:, i:i + 1, :].permute(1, 0, 2)

            # if i == 0:
            #     e_t_prime = e_t.clone()
            # else:
            #     attenOut, _ = self.cross_attention(e_t.clone(), hidden_state_all_visit[:, 0:i, :].clone().permute(1, 0, 2), hidden_state_all_visit[:, 0:i, :].clone().permute(1, 0, 2))
            #     attenOut = self.fc(attenOut)
            #     attenOut = self.dropout(attenOut)
            #     e_t += attenOut
            #     e_t_prime = self.layer_norm(e_t.clone())
            #
            # e_t_prime_all[:, i:i + 1, :] = e_t_prime.permute(1, 0, 2)
            #
            # if i ==0:
            #     _, (seq_h, seq_c) = self.lstm(e_t_prime.clone())
            # else:
            #     _, (seq_h, seq_c) = self.lstm(e_t_prime.clone(),
            #                               (hidden_state_all_visit[:, i-1: i, :].clone().permute(1, 0, 2), c_all_visit[:, i-1: i, :].clone().permute(1, 0, 2)))
            if i ==0:
                _, (seq_h, seq_c) = self.lstm(e_t.clone())
            else:
                _, (seq_h, seq_c) = self.lstm(e_t.clone(),
                                          (hidden_state_all_visit[:, i-1: i, :].clone().permute(1, 0, 2), c_all_visit[:, i-1: i, :].clone().permute(1, 0, 2)))
            hidden_state_all_visit[:, i:i + 1, :] = seq_h.permute(1, 0, 2)
            c_all_visit[:, i:i + 1, :] = seq_c.permute(1, 0, 2)

        bar_e_k = torch.zeros_like(visit_embedding)

        for i in range(visit_size):
            if i == 0:
                bar_e_k[:, 0:1,:] = visit_embedding[:, 0:1, :]
            else:
                e_k = visit_embedding[:, 0:1, :]
                w_h_k_prev = self.w_hk(hidden_state_all_visit[:,i-1:i,:])
                attn = self.w2(self.tanh(self.w1(torch.cat((e_k,w_h_k_prev),dim=-1))))
                alpha1 = attn[:,:,0:1]
                alpha2 = attn[:,:,1:2]
                bar_e_k[:, i:i+1,:] = e_k * alpha1 + w_h_k_prev * alpha2


        ##########diff start
        diffusion_time_t = torch.randint(
            low=0, high=self.config.diffusion.num_diffusion_timesteps, size=[visit_embedding.shape[0], ]).to(
            self.device)

        alpha = (1 - self.betas).cumprod(dim=0).index_select(0, diffusion_time_t).view(-1, 1, 1)

        normal_noise = torch.randn_like(bar_e_k)

        e_t_prime_b_first_with_noise = bar_e_k * alpha.sqrt() + normal_noise * (1.0 - alpha).sqrt()

        predicted_noise = self.diffusion(e_t_prime_b_first_with_noise, timesteps=diffusion_time_t)

        noise_loss = normal_noise - predicted_noise

        GEN_E_t = bar_e_k + noise_loss
        ####### diff end

        for i in range(visit_size):
            E_gen_t = GEN_E_t[:, i:i + 1, :].permute(1, 0, 2)

            # if i == 0:
            #     E_gen_t_prime = E_gen_t.clone()
            # else:
            #     gen_attenOut, _ = self.cross_attention(E_gen_t.clone(), hidden_state_all_visit_generated[:, 0:i, :].clone().permute(1, 0, 2), hidden_state_all_visit_generated[:, 0:i, :].clone().permute(1, 0, 2))
            #     gen_attenOut = self.fc(gen_attenOut)
            #     gen_attenOut = self.dropout(gen_attenOut)
            #     E_gen_t += gen_attenOut
            #     E_gen_t_prime = self.layer_norm(E_gen_t.clone())
            # E_gen_t_prime_all[:, i:i + 1, :] = E_gen_t_prime.permute(1, 0, 2)

            # if i ==0:
            #     _, (seq_h, seq_c) = self.lstm(E_gen_t_prime.clone())
            # else:
            #     _, (seq_h, seq_c) = self.lstm(E_gen_t_prime.clone(),
            #                               (hidden_state_all_visit_generated[:, i-1: i, :].clone().permute(1, 0, 2), c_all_visit_generated[:, i-1: i, :].clone().permute(1, 0, 2)))
            if i ==0:
                _, (seq_h, seq_c) = self.lstm(E_gen_t.clone())
            else:
                _, (seq_h, seq_c) = self.lstm(E_gen_t.clone(),
                                          (hidden_state_all_visit_generated[:, i-1: i, :].clone().permute(1, 0, 2), c_all_visit_generated[:, i-1: i, :].clone().permute(1, 0, 2)))
            hidden_state_all_visit_generated[:, i:i + 1, :] = seq_h.permute(1, 0, 2)
            c_all_visit_generated[:, i:i + 1, :] = seq_c.permute(1, 0, 2)


        for i in range(visit_size):

            hidden_state_softmax_res[:, i:i+1, :] = self.classifyer(hidden_state_all_visit[:, i:i + 1, :])
            hidden_state_softmax_res_generated[:, i:i+1, :] = self.classifyer(hidden_state_all_visit_generated[:, i:i + 1, :])

        final_prediction = hidden_state_softmax_res[:, -1, :]
        final_prediction_generated = hidden_state_softmax_res_generated[:, -1, :]

        # final_prediction = self.classifyer(hidden_state_all_visit[:, visit_size-1:visit_size, :]).squeeze()
        # final_prediction_generated = self.classifyer(hidden_state_all_visit[:, visit_size-1:visit_size, :]).squeeze()

        return hidden_state_all_visit, hidden_state_all_visit_generated, \
               final_prediction, final_prediction_generated, \
               normal_noise, predicted_noise

