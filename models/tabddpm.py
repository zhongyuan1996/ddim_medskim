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
            nn.BatchNorm1d(int(self.x_dim / 2)),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(int(self.x_dim / 2), int(self.x_dim / 4)),
            nn.BatchNorm1d(int(self.x_dim / 4)),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(int(self.x_dim / 4), self.h_dim),
            nn.LeakyReLU()
        )

        self.mlpDecoder = nn.Sequential(
            nn.Linear(self.h_dim, int(self.x_dim / 4)),
            nn.BatchNorm1d(int(self.x_dim / 4)),
            nn.LeakyReLU(),
            nn.Linear(int(self.x_dim / 4), int(self.x_dim / 2)),
            nn.BatchNorm1d(int(self.x_dim / 2)),
            nn.LeakyReLU(),
            nn.Linear(int(self.x_dim / 2), self.x_dim),
            nn.LeakyReLU()
        )

        self.pos_embedding = nn.Embedding(1000+1, x_dim, padding_idx=-1)

    def forward(self, x):
        x = self.initial_embedding(x)
        bs, seq_len, code_len, x_dim = x.shape

        diffusion_time_t = torch.randint(0, 1000, (x.shape[0],), device=self.device)

        time_embedding = self.pos_embedding(diffusion_time_t)
        time_embedding = time_embedding.unsqueeze(1).unsqueeze(2)
        time_embedding = time_embedding.expand(-1, x.shape[1], x.shape[2], -1)


        alpha = (1 - self.betas).cumprod(dim=0).index_select(0, diffusion_time_t).view(-1, 1, 1, 1)

        normal_noise = torch.randn_like(x)

        x_with_noise = x * alpha.sqrt() + normal_noise * (1.0 - alpha).sqrt()

        x_diffinput = x_with_noise + time_embedding

        x_diffinput = x_diffinput.view(-1, x_dim)

        encoded = self.mlpEncoder(x_diffinput)
        predicted_noise = self.mlpDecoder(encoded)

        predicted_noise = predicted_noise.view(bs, seq_len, code_len, x_dim)
        x_diffinput = x_diffinput.view(bs, seq_len, code_len, x_dim)

        gen_x = x_diffinput - predicted_noise

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


