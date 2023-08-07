import torch
import torch.nn as nn


class ehrGAN(nn.Module):
    def __init__(self, x_dim, h_dim, dropout, device):
        super(ehrGAN, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.dropout = dropout
        self.device = device

        self.encoder = nn.Sequential(
            nn.Conv1d(self.x_dim, int(self.x_dim / 2), kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(int(self.x_dim / 2)),
            nn.Dropout(self.dropout),
            nn.Conv1d(int(self.x_dim / 2), self.h_dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.h_dim),
            nn.Dropout(self.dropout)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(self.h_dim, int(self.x_dim / 2), kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(int(self.x_dim / 2)),
            nn.ConvTranspose1d(int(self.x_dim / 2), self.x_dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU()
        )

    def forward(self, x, mask):
        bs, seq_len, code_len, emb_dim = x.size()  # x.shape: batch_size x seq_len x code_len x embedding_dim
        x = x.view(bs * seq_len, code_len, emb_dim)  # reshape: (batch_size * seq_len) x code_len x embedding_dim
        x = x.transpose(1, 2)  # transpose: (batch_size * seq_len) x embedding_dim x code_len

        h = self.encoder(x)  # h.shape: (batch_size * seq_len) x hidden_dim x code_len
        z = torch.randn_like(h)
        m = torch.randn_like(h)
        m = torch.where(m > 0, 1, 0).to(h.dtype)  # apply mask to noise
        h_hat = h * (1 - m) + m * z  # h_hat.shape: (batch_size * seq_len) x hidden_dim x code_len

        fake_x = self.decoder(h_hat)  # fake_x.shape: (batch_size * seq_len) x embedding_dim x code_len
        decode_x = self.decoder(h)  # decode_x.shape: (batch_size * seq_len) x embedding_dim x code_len

        fake_x = fake_x.transpose(1, 2).view(bs, seq_len, code_len,
                                             emb_dim)  # reshape: batch_size x seq_len x code_len x embedding_dim
        decode_x = decode_x.transpose(1, 2).view(bs, seq_len, code_len,
                                                 emb_dim)  # reshape: batch_size x seq_len x code_len x embedding_dim
        og_x = x.transpose(1, 2).view(bs, seq_len, code_len,
                                        emb_dim)  # reshape: batch_size x seq_len x code_len x embedding_dim

        return fake_x * mask, og_x * mask


class CNNPredictor(nn.Module):
    def __init__(self, emb_dim, output_dim, dropout):
        super(CNNPredictor, self).__init__()

        self.conv1 = nn.Conv1d(emb_dim, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc = nn.Linear(128, output_dim)

        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(dropout)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.adaptivepool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        bs, seq_len, code_len, emb_dim = x.size()
        x = x.view(bs, seq_len*code_len, emb_dim) # reshape to [bs, seq_len*code_len, emb_dim]
        x = x.transpose(1, 2)  # [bs, seq_len*code_len, emb_dim] -> [bs, emb_dim, seq_len*code_len]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leakyrelu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leakyrelu(x)
        x = self.maxpool(x)

        x = self.adaptivepool(x).squeeze(dim=2)  # squeeze to shape [bs, 128]

        x = self.fc(x)

        return x

class CNNDiscriminator(nn.Module):
    def __init__(self, emb_dim, dropout):
        super(CNNDiscriminator, self).__init__()

        self.conv1 = nn.Conv1d(emb_dim, 100, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(emb_dim, 100, kernel_size=4, padding=2)  # padding=2 to keep size same after convolution
        self.conv3 = nn.Conv1d(emb_dim, 100, kernel_size=5, padding=2)  # padding=2 to keep size same after convolution

        self.fc = nn.Linear(300, 1)  # 300 = 3 * 100

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.adaptivepool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        bs, seq_len, code_len, emb_dim = x.size()
        x = x.contiguous().view(bs, seq_len*code_len, emb_dim) # reshape to [bs, seq_len*code_len, emb_dim]
        x = x.transpose(1, 2)  # [bs, seq_len*code_len, emb_dim] -> [bs, emb_dim, seq_len*code_len]

        conv1 = self.conv1(x)
        conv1 = self.relu(conv1)
        conv1 = self.maxpool(conv1)

        conv2 = self.conv2(x)
        conv2 = self.relu(conv2)
        conv2 = self.maxpool(conv2)

        conv3 = self.conv3(x)
        conv3 = self.relu(conv3)
        conv3 = self.maxpool(conv3)

        conv1 = self.adaptivepool(conv1).squeeze(dim=2)  # squeeze to shape [bs, 100]
        conv2 = self.adaptivepool(conv2).squeeze(dim=2)  # squeeze to shape [bs, 100]
        conv3 = self.adaptivepool(conv3).squeeze(dim=2)  # squeeze to shape [bs, 100]

        out = torch.cat([conv1, conv2, conv3], dim=1)  # concatenating feature maps
        out = self.fc(out)

        return out


