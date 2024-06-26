import torch
import torch.nn as nn
from utils.diffUtil import get_beta_schedule
from models.unet import *

class tabDDPM(nn.Module):

    def __init__(self, config, vocab_size, x_dim, h_dim, dropout, code_len, device):
        super(tabDDPM,self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.dropout = dropout
        self.device = device
        self.config = config
        self.vocab_size = vocab_size
        self.code_len = code_len
        self.initial_embedding = nn.Embedding(self.vocab_size+1, self.vocab_size, padding_idx=-1)
        self.y_embedding = nn.Embedding(2, self.vocab_size+1, padding_idx=-1)
        betas = get_beta_schedule(beta_schedule=self.config.diffusion.beta_schedule,
                                  beta_start=self.config.diffusion.beta_start,
                                  beta_end=self.config.diffusion.beta_end,
                                  num_diffusion_timesteps=self.config.diffusion.num_diffusion_timesteps)
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.diffusion_num_timesteps = betas.shape[0]
        self.softmax = nn.Softmax(dim=-1)

        self.mlpEncoder = nn.Sequential(
            nn.Linear(self.vocab_size, int((self.vocab_size) / 4)),
            nn.BatchNorm1d(int((self.vocab_size) / 4)),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(int((self.vocab_size) / 4), int((self.vocab_size) / 8)),
            nn.BatchNorm1d(int((self.vocab_size) / 8)),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(int((self.vocab_size) / 8), self.h_dim),
            nn.LeakyReLU()
        )

        self.mlpDecoder = nn.Sequential(
            nn.Linear(self.h_dim, int((self.vocab_size) / 8)),
            nn.BatchNorm1d(int((self.vocab_size) / 8)),
            nn.LeakyReLU(),
            nn.Linear(int((self.vocab_size) / 8), int((self.vocab_size) / 4)),
            nn.BatchNorm1d(int((self.vocab_size) / 4)),
            nn.LeakyReLU(),
            nn.Linear(int((self.vocab_size) / 4), self.vocab_size),
            nn.LeakyReLU()
        )

        self.pos_embedding = nn.Embedding(1000+1, self.vocab_size, padding_idx=-1)

    def forward(self, x, label = None):
        x = self.initial_embedding(x).sum(dim=-2)
        bs, seq_len, code_len = x.shape

        # Diffusion time
        diffusion_time_t = torch.randint(0, 1000, (bs,), device=self.device)
        time_embedding = self.pos_embedding(diffusion_time_t)
        time_embedding = time_embedding.unsqueeze(1).expand(-1, x.shape[1], -1)
        alpha = (1 - self.betas).cumprod(dim=0).index_select(0, diffusion_time_t).view(-1, 1, 1)

        # Add categorical noise
        uniform_noise = torch.ones_like(x) / self.vocab_size
        x_with_noise = alpha * x + (1-alpha) * uniform_noise
        if label:
            y_embedding = self.y_embedding(label.to(torch.long)).sum(dim=-2).unsqueeze(1).expand(-1, x.shape[1], -1)
            x_diffinput = x_with_noise + time_embedding + y_embedding
        else:
            x_diffinput = x_with_noise + time_embedding
        # Flatten the tensor for MLP processing
        x_diffinput = x_diffinput.view(-1, code_len)

        # Encoder and Decoder passes
        encoded = self.mlpEncoder(x_diffinput)
        predicted_noise = self.mlpDecoder(encoded)

        # Reshape to original dimensions
        predicted_noise = predicted_noise.view(bs, seq_len, code_len)
        x_diffinput = x_diffinput.view(bs, seq_len, code_len)

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


