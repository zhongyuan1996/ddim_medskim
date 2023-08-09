import torch
import torch.nn as nn
from utils.diffUtil import get_beta_schedule

class tabDDPM(nn.Module):

    def __init__(self, config, vocab_size, x_dim, h_dim, dropout, device):
        super(tabDDPM,self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.dropout = dropout
        self.device = device
        self.config = config
        self.vocab_size = vocab_size
        self.initial_embedding = nn.Embedding(self.vocab_size+1, x_dim, padding_idx=-1)
        betas = get_beta_schedule(beta_schedule=self.config.diffusion.beta_schedule,
                                  beta_start=self.config.diffusion.beta_start,
                                  beta_end=self.config.diffusion.beta_end,
                                  num_diffusion_timesteps=self.config.diffusion.num_diffusion_timesteps)
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.diffusion_num_timesteps = betas.shape[0]

        self.mlpEncoder = nn.Sequential(
            nn.Linear(self.x_dim, int(self.x_dim / 2)),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(int(self.x_dim / 2), self.h_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.mlpDecoder = nn.Sequential(
            nn.Linear(self.h_dim, int(self.x_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(self.x_dim / 2), self.x_dim),
            nn.ReLU()
        )

        self.pos_embedding = nn.Embedding(1000+1, x_dim, padding_idx=-1)

    def forward(self, x):
        x = self.initial_embedding(x)

        diffusion_time_t = torch.randint(0, 1000, (x.shape[0],), device=self.device)

        time_embedding = self.pos_embedding(diffusion_time_t)
        time_embedding = time_embedding.unsqueeze(1).unsqueeze(2)
        time_embedding = time_embedding.expand(-1, x.shape[1], x.shape[2], -1)


        alpha = (1 - self.betas).cumprod(dim=0).index_select(0, diffusion_time_t).view(-1, 1, 1, 1)

        normal_noise = torch.randn_like(x)

        x_with_noise = x * alpha.sqrt() + normal_noise * (1.0 - alpha).sqrt()

        x_diffinput = x_with_noise + time_embedding

        encoded = self.mlpEncoder(x_diffinput)
        predicted_noise = self.mlpDecoder(encoded)

        gen_x = x_diffinput - predicted_noise

        return x, gen_x

class LSTM_predictor(nn.Module):

    def __init__(self, vocab_size, x_dim, h_dim, dropout):
        super(LSTM_predictor, self).__init__()
        self.vocab_size = vocab_size
        self.initial_embedding = nn.Embedding(self.vocab_size + 1, x_dim, padding_idx=-1)
        self.lstm = nn.LSTM(x_dim, h_dim, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(h_dim, 2)

    def forward(self, x):
        x = self.initial_embedding(x).sum(-2)
        output, (h_n, c_n) = self.lstm(x)
        output = self.linear(output[:, -1, :])

        return output

