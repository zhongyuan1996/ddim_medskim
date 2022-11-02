import math
import torch
import numpy as np
import torch.nn as nn
from models.diffusion import diffModel
from utils.diffUtil import get_beta_schedule


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)

        self.cross_multi_attn = nn.MultiheadAttention(dim, self.num_heads)
        # self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B, N, C = x.shape
        q = self.wq(y[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1,
                                                                                               3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1,
                                                                                  3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1,
                                                                                  3)  # BNC -> BNH(C/H) -> BHN(C/H)

        output, _weights = self.cross_multi_attn(q, k, v)

        return output

        # attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        #
        # x = (attn @ v).transpose(1, 2).reshape(B, 1, C)   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        # x = self.proj(x)
        # x = self.proj_drop(x)
        # return x


class CrossAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x, y):
        x = x[:, 0:1, ...] + self.attn(self.norm1(x), self.norm1(y))
        return x


class classifyer(nn.Module):

    def __init__(self, d_hiddens_tate):
        super.__init__()
        self.layer1 = nn.Linear(d_hiddens_tate, 4 * d_hiddens_tate)
        self.layer2 = nn.Linear(4 * d_hiddens_tate, 4 * d_hiddens_tate)
        self.out = nn.Linear(4 * d_hiddens_tate, 2)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.1)
        self.softmax = nn.Softmax()

    def forward(self, h):
        h = self.relu(self.layer1(h))
        h = self.relu(self.layer2(h))
        h = self.drop(h)
        h = self.softmax(self.out(h))

        return h


class diffRNN(nn.Module):

    def __init__(self, config, vocab_size, d_model, h_model, dropout, dropout_emb):
        super.__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.initial_embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.cross_attention = CrossAttentionBlock(d_model, 1, drop=0.1, attn_drop=0.1)
        self.lstm = nn.LSTM(d_model, h_model, num_layers=1, batch_first=True, dropout= dropout)

        self.diffusion = diffModel(config)
        betas = get_beta_schedule(beta_schedule=config.diffusion.beta_schedule,
                                  beta_start=config.diffusion.beta_start,
                                  beta_end=config.diffusion.beta_end,
                                  num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps)
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
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.time_layer = nn.Linear(1, 64)
        self.time_updim = nn.Linear(64, d_model)

    def forward(self, input_seqs, lengths, seq_time_step):
        # outputs, skip_rate = model(ehr, pad_id, time_step, code_mask)

        # seq_time_step = seq_time_step.unsqueeze(2) / 180
        # time_feature = 1 - self.tanh(torch.pow(self.time_layer(seq_time_step), 2))
        # time_encoding = self.time_updim(time_feature)

        _, seq_len, _ = input_seqs.size()

        seq_e_i = []
        seq_h = []
        seq_c = []
        seq_noise_loss = []

        seq_e_i_gen1 = []
        seq_e_i_gen2 = []
        seq_h_gen = []
        seq_c_gen = []
        seq_h_prob = []
        seq_h_gen_prob = []
        for i in range(seq_len):
            e_i = self.embedding(input_seqs)
            e_i = self.emb_dropout(e_i)
            e_i = self.relu(e_i)
            # TODO: time embedding
            seq_e_i.append(e_i)

            diffusion_time_t = np.random.randint(0, 1000, size=1)
            predicted_noise = self.diffusion(e_i, t=diffusion_time_t)
            normal_noise = torch.randn_like(e_i)
            noise_loss = normal_noise - predicted_noise
            e_i_gen1 = e_i + noise_loss
            seq_noise_loss.append(noise_loss)
            try:
                e_i_gen2 = self.cross_attention(e_i_gen1[i], e_i_gen2[i - 1])
            except:
                e_i_gen2 = self.cross_attention(e_i_gen1[i], torch.zeros_like(e_i_gen1[i]))

            seq_e_i_gen1.append(e_i_gen1)
            seq_e_i_gen2.append(e_i_gen2)

            try:
                h_i, c_i = self.lstm(e_i, (seq_h[i - 1], seq_c[i - 1]))
            except:
                h_i, c_i = self.lstm(e_i, (torch.zeros_like(e_i), torch.zeros_like(e_i)))

            seq_h.append(h_i)
            seq_c.append(c_i)

            try:
                h_i_gen, c_i_gen = self.lstm(e_i_gen2, (seq_h_gen[i - 1], seq_c_gen[i - 1]))
            except:
                h_i_gen, c_i_gen = self.lstm(e_i_gen2, (torch.zeros_like(e_i_gen2), torch.zeros_like(e_i_gen2)))

            seq_h_gen.append(h_i_gen)
            seq_c_gen.append(c_i_gen)

            seq_h_prob[i] = self.classifyer(h_i)
            seq_h_gen_prob[i] = self.classifyer(h_i_gen)

        return seq_h_prob, seq_h_gen_prob, seq_noise_loss
