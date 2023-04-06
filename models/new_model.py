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

        return h

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

class TargetDiff(nn.Module):

    def __init__(self, config, vocab_size, d_model, h_model, dropout, dropout_emb, device, num_visit_chosen = 3, num_patients_gen = 5, eta = 0.05, info_control = 0.9):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.device = device
        self.h_model = h_model
        self.num_visit_chosen = num_visit_chosen
        self.num_patients_gen = num_patients_gen
        self.eta = eta
        self.ic = info_control
        self.model_var_type = self.config.model.var_type
        self.initial_embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        # self.cross_attention = nn.MultiheadAttention(d_model, 8, batch_first=False)
        self.self_attention = nn.MultiheadAttention(d_model, 8, batch_first=True)

        self.lstm = nn.LSTM(d_model, h_model, num_layers=2, batch_first=True, dropout=dropout)
        self.LRnn = nn.LSTM(d_model, h_model, num_layers=2, batch_first=True, dropout=dropout)
        self.RRnn = nn.LSTM(d_model, h_model, num_layers=2, batch_first=True, dropout=dropout)
        self.diffusion = UNetModel(in_channels=self.num_visit_chosen, model_channels=128,
                                   out_channels=self.num_visit_chosen, num_res_blocks=2,
                                   attention_resolutions=[16, ])
        betas = get_beta_schedule(beta_schedule=self.config.diffusion.beta_schedule,
                                  beta_start=self.config.diffusion.beta_start,
                                  beta_end=self.config.diffusion.beta_end,
                                  num_diffusion_timesteps=self.config.diffusion.num_diffusion_timesteps)
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.diffusion_num_timesteps = betas.shape[0]

        # self.visit_diffusion = UNetModel(in_channels=3, model_channels=128,
        #                                  out_channels=1, num_res_blocks=2,
        #                                  attention_resolutions=[16, ])

        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        self.classifyer = classifyer(h_model*2)
        # self.embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
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
        self.softmax = torch.nn.Softmax(dim=-1)
        self.attn_softmax = torch.nn.Softmax(dim=-1)

        self.visit_attention_layer = nn.Linear(d_model * 2, 1)
        self.voting_layer = nn.Linear((1+self.num_patients_gen)*2, 2)
        self.aggregate_hl_hr_vi = nn.Linear(3 * d_model, d_model)

    def time_embedding_block(self, seq_time_step):

        seq_time_step = seq_time_step.unsqueeze(2) / 180
        time_feature = 1 - self.tanh(torch.pow(self.time_layer(seq_time_step), 2))
        time_encoding = self.time_updim(time_feature)

        return time_encoding

    def forward(self, input_seqs, seq_time_step):

        batch_size, visit_size, icd_code_size = input_seqs.size()

        visit_embedding = self.initial_embedding(input_seqs)
        visit_embedding = self.emb_dropout(visit_embedding)
        visit_embedding = torch.sum(self.relu(visit_embedding), dim=-2)
        time_embedding = self.time_embedding_block(seq_time_step)
        time_aware_visit_embedding = visit_embedding + time_embedding

        #generating attention by visits


        #maybe add attention on code selection to reduce dimension
        #changed transformer hidden states to LSTM
        #visit_attention_output, visit_attention_weights = self.self_attention(visit_embedding, visit_embedding, visit_embedding)
        L_h, L_c = self.LRnn(time_aware_visit_embedding)
        R_h, R_c = self.RRnn(torch.flip(time_aware_visit_embedding, dims=[1]))
        LR_h = torch.cat((L_h, R_h), dim=-1)

        visit_attention = self.softmax(self.visit_attention_layer(LR_h).squeeze(-1))

        #maybe consider time information in generating visit attention

        #visit_attention = self.softmax(self.visit_attention_layer(visit_attention_output)).squeeze(-1)

        attention_list = visit_attention.clone().tolist()
        selected_indices = []
        for i in range(batch_size):
            top_pct_indices, agg_pct, id = [], 0.0, 0
            idx_a = [(idx, a) for idx, a in enumerate(attention_list[i])]
            idx_a.sort(key=lambda x: x[1], reverse=True)
            while agg_pct < self.ic:
                top_pct_indices.append(idx_a[id][0])
                agg_pct += idx_a[id][1]
                id += 1
            selected_indices.append(set(top_pct_indices))


        #choose the top by highest attention

        _, indices = torch.topk(visit_attention, self.num_visit_chosen, dim=-1)

        #add nosie to the top visit

        top_visit = torch.stack([torch.index_select(visit_embedding[patient], 0, indices[patient]) for patient, ind in zip(range(batch_size), indices)],dim=0)

        top_visit_with_noise = top_visit + self.eta * torch.randn_like(top_visit)

        for i in range(visit_size):

            og_pred = self.classifyer(LR_h[:, i:i + 1, :])

        final_prediction = og_pred

        for i in range(self.num_patients_gen):
            diffusion_time_t = torch.randint(
                low=0, high=self.config.diffusion.num_diffusion_timesteps, size=[top_visit_with_noise.shape[0], ]).to(
                self.device)

            alpha = (1 - self.betas).cumprod(dim=0).index_select(0, diffusion_time_t).view(-1, 1, 1)

            normal_noise = torch.randn_like(top_visit_with_noise)

            topk_with_noise = top_visit_with_noise * alpha.sqrt() + normal_noise * (1.0 - alpha).sqrt()

            predicted_noise = self.diffusion(topk_with_noise, timesteps=diffusion_time_t)

            noise_loss = normal_noise - predicted_noise

            generated_top_visits = top_visit_with_noise + noise_loss #tensor(batch_size, num_visit_chosen:1, d_model:256)

            temp_patients = time_aware_visit_embedding.clone()

            for patient in range(batch_size):
                for i, visit in enumerate(indices[patient]):
                    temp_patients[patient][visit] = generated_top_visits[patient][i]

            L_h, _ = self.LRnn(temp_patients.clone())
            R_h, _ = self.RRnn(torch.flip(temp_patients.clone(), dims=[1]))

            for patient in range(batch_size):
                diffusion_time_t = torch.randint(
                    low=0, high=self.config.diffusion.num_diffusion_timesteps, size=[1, ]).to(
                    self.device)
                alpha = (1 - self.betas).cumprod(dim=0).index_select(0, diffusion_time_t).view(-1, 1, 1)

                for visit in selected_indices[patient]:

                    if visit != indices[patient][0] and visit != 0 and visit != visit_size-1:

                        normal_noise = torch.randn_like(temp_patients[patient][visit])
                        h_LR_og = self.aggregate_hl_hr_vi(torch.cat([temp_patients[patient][visit],L_h[patient][visit-1],R_h[patient][visit+1]], dim=-1))
                        h_LR_og_with_noise = h_LR_og * alpha.sqrt() + normal_noise * (1.0 - alpha).sqrt()
                        predicted_noise = self.diffusion(h_LR_og_with_noise, timesteps=diffusion_time_t)
                        noise_loss = normal_noise - predicted_noise
                        temp_patients[patient][visit] = h_LR_og + noise_loss

                    elif visit != indices[patient][0] and visit ==0:

                        normal_noise = torch.randn_like(temp_patients[patient][visit])
                        h_LR_og = self.aggregate_hl_hr_vi(torch.cat([temp_patients[patient][visit],torch.zeros_like(R_h[patient][visit+1]),R_h[patient][visit+1]],dim=-1))
                        h_LR_og_with_noise = h_LR_og * alpha.sqrt() + normal_noise * (1.0 - alpha).sqrt()
                        predicted_noise = self.diffusion(h_LR_og_with_noise, timesteps=diffusion_time_t)
                        noise_loss = normal_noise - predicted_noise
                        temp_patients[patient][visit] = h_LR_og + noise_loss

                    elif visit != indices[patient][0] and visit == visit_size-1:

                        normal_noise = torch.randn_like(temp_patients[patient][visit])
                        h_LR_og = self.aggregate_hl_hr_vi(torch.cat([temp_patients[patient][visit],L_h[patient][visit-1],torch.zeros_like(L_h[patient][visit-1])],dim=-1))
                        h_LR_og_with_noise = h_LR_og * alpha.sqrt() + normal_noise * (1.0 - alpha).sqrt()
                        predicted_noise = self.diffusion(h_LR_og_with_noise, timesteps=diffusion_time_t)
                        noise_loss = normal_noise - predicted_noise
                        temp_patients[patient][visit] = h_LR_og + noise_loss

                    else:
                        continue

            # temp_patients = temp_patients + time_embedding

            # temp_h, temp_c = self.lstm(temp_patients)
            temp_L_h, _ = self.LRnn(temp_patients.clone())
            temp_R_h, _ = self.RRnn(torch.flip(temp_patients.clone(), dims=[1]))
            temp_LR_h = torch.cat((temp_L_h, temp_R_h), dim=-1)

            for i in range(visit_size):
                temp_pred = self.classifyer(temp_LR_h[:, i:i + 1, :])

            final_prediction = torch.cat((final_prediction, temp_pred), dim=1)

        final_prediction = torch.unflatten(self.voting_layer(torch.flatten(final_prediction, start_dim=-2, end_dim=-1)),
                                           dim=-1, sizes=(1, 2))
        final_prediction = final_prediction.squeeze(-2)
        return final_prediction


        #put all the visits into the classifier





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
        self.softmax = torch.nn.Softmax(dim=-1)


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
                attn = self.softmax(self.w2(self.tanh(self.w1(torch.cat((e_k,w_h_k_prev),dim=-1)))))
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

        return hidden_state_softmax_res, hidden_state_softmax_res_generated, \
               final_prediction, final_prediction_generated, \
               normal_noise, predicted_noise

