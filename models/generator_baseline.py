import torch
import torch.nn as nn

class Linear_Generator(nn.Module):
    def __init__(self, d_model=256, dropout=0.1):
        super(Linear_Generator, self).__init__()

        self.d_model = d_model
        self.dropout = dropout
        self.linear1 = nn.Linear(self.d_model, self.d_model)
        self.linear2 = nn.Linear(self.d_model, self.d_model)
        self.linear3 = nn.Linear(self.d_model, self.d_model)
        # self.batchnorm1 = nn.BatchNorm1d(self.d_model)
        # self.batchnorm2 = nn.BatchNorm1d(self.d_model)
        # self.batchnorm3 = nn.BatchNorm1d(self.d_model)
        self.seq = nn.Sequential(self.linear1, nn.ReLU(inplace=False), nn.Dropout(self.dropout,inplace=False), self.linear2, nn.ReLU(inplace=False), nn.Dropout(self.dropout, inplace=False), self.linear3, nn.ReLU(inplace=False), nn.Dropout(self.dropout,inplace=False))
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(self.dropout, inplace=False)
        # self.block1 = nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.BatchNorm1d(self.d_model), nn.ReLU(), nn.Dropout(self.dropout))
        # self.block2 = nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.BatchNorm1d(self.d_model), nn.ReLU(), nn.Dropout(self.dropout))
        # self.block3 = nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.BatchNorm1d(self.d_model), nn.ReLU(), nn.Dropout(self.dropout))
        self.out = nn.Linear(self.d_model, self.d_model)

    def forward(self, x):
        z = torch.randn_like(x).to(x.device)
        out = self.seq(z)
        return out


        # # First block
        # z1 = self.linear1(z).transpose(1, 2)
        # z1 = self.dropout(self.relu(self.batchnorm1(z1).transpose(1, 2)))
        # z = z1 + z  # Residual connection
        #
        # # Second block
        # z2 = self.linear2(z).transpose(1, 2)
        # z2 = self.dropout(self.relu(self.batchnorm2(z2).transpose(1, 2)))
        # z = z2 + z  # Residual connection
        #
        # # Third block
        # z3 = self.linear3(z).transpose(1, 2)
        # z3 = self.dropout(self.relu(self.batchnorm3(z3).transpose(1, 2)))
        # z = z3 + z  # Residual connection
        #
        # # Final output layer
        # out = self.out(z)
        # return out


class Linear_Discriminator(nn.Module):
    def __init__(self, d_model=256, h_model = 128, dropout=0.1):
        super(Linear_Discriminator, self).__init__()
        self.d_model = d_model
        self.h_model = h_model
        self.dropout = dropout
        self.linear1 = nn.Linear(self.d_model, self.h_model)
        self.batchnorm1 = nn.BatchNorm1d(self.h_model)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(self.dropout, inplace=False)
        self.out = nn.Linear(self.h_model, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(self.relu(x))
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

        # # Discriminate on the real and generated data
        # real_data_discrimination = self.discriminate(x)
        # generated_data_discrimination = self.discriminate(generated_data)

        return generated_data

