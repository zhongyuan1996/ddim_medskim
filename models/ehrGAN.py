import torch
import torch.nn as nn


class ehrGAN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, code_length, dropout):
        super(ehrGAN, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.code_length = code_length
        self.dropout = nn.Dropout(dropout)

        self.encoder = nn.Sequential(
            nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, embedding_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x, mask):
        batch_size, seq_len, _ = x.size()
        x = x.transpose(1, 2)  # shape: batch_size x embedding_dim x seq_len
        h = self.encoder(x)  # shape: batch_size x hidden_dim x seq_len
        z = torch.randn_like(h)  # noise
        m = torch.where(mask > 0, 1, 0).to(h.dtype)  # apply mask to noise
        h_hat = h * (1 - m) + m * z  # shape: batch_size x hidden_dim x seq_len
        fake_x = self.decoder(h_hat)  # shape: batch_size x embedding_dim x seq_len
        fake_x = fake_x.transpose(1, 2)  # shape: batch_size x seq_len x embedding_dim
        return fake_x


