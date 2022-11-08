import math
import torch
import numpy as np
import torch.nn as nn
from models.diffusion import diffModel
from utils.diffUtil import get_beta_schedule
from models.unet import *

class classifyer(nn.Module):

    def __init__(self, d_hiddens_tate):
        super().__init__()
        self.layer1 = nn.Linear(d_hiddens_tate, 4 * d_hiddens_tate)
        self.layer2 = nn.Linear(4 * d_hiddens_tate, 4 * d_hiddens_tate)
        self.out = nn.Linear(4 * d_hiddens_tate, 2)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, h):
        h = self.relu(self.layer1(h))
        h = self.relu(self.layer2(h))
        h = self.drop(h)
        h = self.softmax(self.out(h))

        return h


class diffRNN(nn.Module):

    def __init__(self, config, vocab_size, d_model, h_model, dropout, dropout_emb, device):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.device = device
        self.h_model = h_model
        self.model_var_type = self.config.model.var_type
        self.initial_embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.cross_attention = nn.MultiheadAttention(d_model, 8)
        # CrossAttentionBlock(d_model, 2, drop=0.1, attn_drop=0.1)
        self.lstm = nn.LSTM(d_model, h_model, num_layers=1, batch_first=True, dropout=dropout)

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

    def forward(self, input_seqs, seq_time_step):
        # outputs, skip_rate = model(ehr, pad_id, time_step, code_mask)

        # seq_time_step = seq_time_step.unsqueeze(2) / 180
        # time_feature = 1 - self.tanh(torch.pow(self.time_layer(seq_time_step), 2))
        # time_encoding = self.time_updim(time_feature)

        batch_size, visit_size, icd_code_size = input_seqs.size()

        seq_time_step = seq_time_step.unsqueeze(2) / 180
        time_feature = 1 - self.tanh(torch.pow(self.time_layer(seq_time_step), 2))
        time_encoding = self.time_updim(time_feature)


        # for each visit the data is in the form of 20 * 256
        visit_embedding = self.initial_embedding(input_seqs)
        visit_embedding = self.emb_dropout(visit_embedding)
        visit_embedding = self.relu(visit_embedding)
        # visit_embedding.shape = [64, 50, 20, 64]
        # print(visit_embedding.shape[0])
        visit_embedding = visit_embedding.sum(-2) + time_encoding

        diffusion_time_t = torch.randint(
            low=0, high=self.config.diffusion.num_diffusion_timesteps, size=[visit_embedding.shape[0], ]).to(
            self.device)
        #TODO: Fix
        # a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        # x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
        # output = model(x, t.float())

        alpha = (1 - self.betas).cumprod(dim=0).index_select(0, diffusion_time_t).view(-1, 1, 1)
        normal_noise = torch.randn_like(visit_embedding)
        # a = visit_embedding * alpha.sqrt()
        # b = normal_noise * (1.0 - alpha).sqrt()
        visit_embedding_with_noise = visit_embedding * alpha.sqrt() + normal_noise * (1.0 - alpha).sqrt()

        predicted_noise = self.diffusion(visit_embedding_with_noise, timesteps=diffusion_time_t)

        noise_loss = visit_embedding_with_noise - predicted_noise

        visit_embedding_generated1 = visit_embedding + noise_loss
        visit_embedding_generated2 = torch.zeros_like(visit_embedding_generated1).fill_(self.vocab_size)

        hidden_state_all_visit = torch.zeros(batch_size, visit_size, self.h_model).to(self.device)
        hidden_state_all_visit_generated = torch.zeros(batch_size, visit_size, self.h_model).to(self.device)

        hidden_state_softmax_res = torch.zeros(batch_size, visit_size, 2).to(self.device)
        hidden_state_softmax_res_generated = torch.zeros(batch_size, visit_size, 2).to(self.device)


        for i in range(visit_size):

            try:
                visit_embedding_generated2[:, i, :], _ = self.cross_attention(
                    visit_embedding_generated2[:, i - 1, :].clone(), visit_embedding_generated1[:, i, :].clone(),
                    visit_embedding_generated1[:, i, :].clone())

            except:
                visit_embedding_generated2[:, i, :], _ = self.cross_attention(
                    torch.zeros_like(visit_embedding_generated1[:, i, :].clone()), visit_embedding_generated1[:, i, :].clone(),
                    visit_embedding_generated1[:, i, :].clone())

            try:
                _, (seq_h, seq_c) = self.lstm(visit_embedding[:, i, :].clone(), (seq_h.clone(), seq_c.clone()))
            except:
                _, (seq_h, seq_c) = self.lstm(visit_embedding[:, i, :].clone())

            seq_h = torch.squeeze(seq_h)
            # seq_c = torch.squeeze(seq_c)

            hidden_state_all_visit[:, i, :] = seq_h

            try:
                _, (seq_h_gen, seq_c_gen) = self.lstm(visit_embedding_generated2[:, i, :].clone(), (seq_h_gen.clone(), seq_c_gen.clone()))
            except:
                _, (seq_h_gen, seq_c_gen) = self.lstm(visit_embedding_generated2[:, i, :].clone())
            # torch.Size([64, 256])
            seq_h_gen = torch.squeeze(seq_h_gen)
            # seq_c_gen = torch.squeeze(seq_c_gen)

            hidden_state_all_visit_generated[:, i, :] = seq_h_gen

            hidden_state_softmax_res[:, i, :] = self.classifyer(seq_h)
            hidden_state_softmax_res_generated[:, i, :] = self.classifyer(seq_h_gen)

        final_prediction = hidden_state_softmax_res[:,-1,:]
        final_prediction_generated = hidden_state_softmax_res_generated[:,-1,:]
        #[64,2]

        return hidden_state_softmax_res, hidden_state_softmax_res_generated, \
               final_prediction, final_prediction_generated, \
               normal_noise, predicted_noise
