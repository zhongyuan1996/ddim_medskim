import torch
import torch.nn as nn
from utils.diffUtil import get_beta_schedule
from models.unet import *

class DDPM(nn.Module):

    def __init__(self, config, vocab_size, x_dim, h_dim, dropout, seq_len, code_len, device):
        super(DDPM,self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.dropout = dropout
        self.device = device
        self.config = config
        self.vocab_size = vocab_size
        self.code_len = code_len
        self.seq_len = seq_len
        self.initial_embedding = nn.Embedding(self.vocab_size+1, self.vocab_size+1, padding_idx=-1)
        self.y_embedding = nn.Embedding(2, self.vocab_size+1, padding_idx=-1)
        betas = get_beta_schedule(beta_schedule=self.config.diffusion.beta_schedule,
                                  beta_start=self.config.diffusion.beta_start,
                                  beta_end=self.config.diffusion.beta_end,
                                  num_diffusion_timesteps=self.config.diffusion.num_diffusion_timesteps)
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.diffusion_num_timesteps = betas.shape[0]
        self.softmax = nn.Softmax(dim=-1)

        self.diffusion = UNetModel(in_channels=self.code_len, model_channels=128,
                                   out_channels=self.code_len, num_res_blocks=2,
                                   attention_resolutions=[16, ])

        self.pos_embedding = nn.Embedding(1000+1, self.vocab_size+1, padding_idx=-1)

    def forward(self, x):
        bs, seq_len, code_len, emb_dim = x.size()
        x = x.view(bs * seq_len, code_len, emb_dim)

        diffusion_time_t = torch.randint(
            low=0, high=self.config.diffusion.num_diffusion_timesteps, size=[x.shape[0], ]).to(
            self.device)
        alpha = (1 - self.betas).cumprod(dim=0).index_select(0, diffusion_time_t).view(-1, 1, 1)
        normal_noise = torch.randn_like(x)
        x_with_noise = x * alpha.sqrt() + normal_noise * (1.0 - alpha).sqrt()
        predicted_noise = self.diffusion(x_with_noise, timesteps=diffusion_time_t)
        gen_x = x_with_noise - predicted_noise
        x = x.view(bs, seq_len, code_len, emb_dim)
        gen_x = gen_x.view(bs, seq_len, code_len, emb_dim)

        return x, gen_x

class LSTM_predictor(nn.Module):

    def __init__(self, vocab_size, x_dim, h_dim, dropout):
        super(LSTM_predictor, self).__init__()
        self.vocab_size = vocab_size
        self.x_dim = x_dim
        self.h_dim = h_dim

        self.initial_embedding = nn.Embedding(self.vocab_size + 1, x_dim, padding_idx=-1)
        self.lstm = nn.LSTM(x_dim, h_dim, dropout=dropout, batch_first=True)

        self.fc1 = nn.Linear(h_dim, int(h_dim / 2))
        self.bn1 = nn.BatchNorm1d(int(h_dim / 2))
        self.relu1 = nn.LeakyReLU()

        self.fc2 = nn.Linear(int(h_dim / 2), 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.initial_embedding(x).sum(-2)
        output, (h_n, c_n) = self.lstm(x)

        f = self.fc1(output[:, -1, :])
        f = self.bn1(f)
        f = self.relu1(f)
        f = self.dropout(f)

        output = self.fc2(f)
        return output


