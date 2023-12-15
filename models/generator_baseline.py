import torch
import torch.nn as nn

class Linear_Generator(nn.Module):
    def __init__(self, d_model=256, dropout=0.1):
        super(Linear_Generator, self).__init__()

        self.d_model = d_model
        self.dropout = dropout
        self.block1 = nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.BatchNorm1d(self.d_model), nn.ReLU(), nn.Dropout(self.dropout))
        self.block2 = nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.BatchNorm1d(self.d_model), nn.ReLU(), nn.Dropout(self.dropout))
        self.block3 = nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.BatchNorm1d(self.d_model), nn.ReLU(), nn.Dropout(self.dropout))
        self.out = nn.Linear(self.d_model, self.d_model)
    def forward(self, x):

        z = torch.randn_like(x).to(x.device)
        z = self.block1(z) + z
        z = self.block2(z) + z
        z = self.block3(z) + z
        z = self.out(z)
        return z

class Linear_Discriminator(nn.Module):
    def __init__(self, d_model=256, h_model = 128, dropout=0.1):
        super(Linear_Discriminator, self).__init__()
        self.d_model = d_model
        self.h_model = h_model
        self.dropout = dropout
        self.block1 = nn.Sequential(nn.Linear(self.d_model, self.h_model), nn.BatchNorm1d(self.h_model), nn.ReLU(), nn.Dropout(self.dropout))
        self.out = nn.Linear(self.h_model, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.out(x)
        return x


class medGAN(nn.Module):
    def __init__(self, d_model=256, h_model=128, dropout=0.1):
        super(medGAN, self).__init__()
        self.generator = Linear_Generator(d_model=d_model, dropout=dropout)
        self.discriminator = Linear_Discriminator(d_model=d_model, h_model=h_model, dropout=dropout)

    def generate(self, x):
        return self.generator(x)

    def discriminate(self, x):
        return self.discriminator(x)

    def forward(self, x):
        # Generate data
        generated_data = self.generate(x)

        # Discriminate on the real and generated data
        real_data_discrimination = self.discriminate(x)
        generated_data_discrimination = self.discriminate(generated_data)

        return generated_data, real_data_discrimination, generated_data_discrimination

