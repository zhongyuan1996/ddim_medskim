import torch
import torch.nn as nn
from utils.diffUtil import get_beta_schedule
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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
        self.initial_embedding = nn.Embedding(self.vocab_size+1, self.vocab_size+1, padding_idx=-1)
        self.y_embedding = nn.Embedding(2, self.vocab_size+1, padding_idx=-1)
        betas = get_beta_schedule(beta_schedule=self.config.diffusion.beta_schedule,
                                  beta_start=self.config.diffusion.beta_start,
                                  beta_end=self.config.diffusion.beta_end,
                                  num_diffusion_timesteps=self.config.diffusion.num_diffusion_timesteps)
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.diffusion_num_timesteps = betas.shape[0]
        self.softmax = nn.Softmax(dim=-1)

        self.mlpEncoder = nn.Sequential(
            nn.Linear(self.vocab_size+1, int((self.vocab_size+1) / 4)),
            nn.BatchNorm1d(int((self.vocab_size+1) / 4)),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(int((self.vocab_size+1) / 4), int((self.vocab_size+1) / 8)),
            nn.BatchNorm1d(int((self.vocab_size+1) / 8)),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(int((self.vocab_size+1) / 8), self.h_dim),
            nn.LeakyReLU()
        )

        self.mlpDecoder = nn.Sequential(
            nn.Linear(self.h_dim, int((self.vocab_size+1) / 8)),
            nn.BatchNorm1d(int((self.vocab_size+1) / 8)),
            nn.LeakyReLU(),
            nn.Linear(int((self.vocab_size+1) / 8), int((self.vocab_size+1) / 4)),
            nn.BatchNorm1d(int((self.vocab_size+1) / 4)),
            nn.LeakyReLU(),
            nn.Linear(int((self.vocab_size+1) / 4), self.vocab_size+1),
            nn.LeakyReLU()
        )

        self.pos_embedding = nn.Embedding(1000+1, self.vocab_size+1, padding_idx=-1)

    # def old_forward(self, x):
    #     x = self.initial_embedding(x)
    #     bs, seq_len, code_len, x_dim = x.shape
    #
    #     diffusion_time_t = torch.randint(0, 1000, (x.shape[0],), device=self.device)
    #
    #     time_embedding = self.pos_embedding(diffusion_time_t)
    #     time_embedding = time_embedding.unsqueeze(1).unsqueeze(2)
    #     time_embedding = time_embedding.expand(-1, x.shape[1], x.shape[2], -1)
    #
    #
    #     alpha = (1 - self.betas).cumprod(dim=0).index_select(0, diffusion_time_t).view(-1, 1, 1, 1)
    #
    #     normal_noise = torch.randn_like(x)
    #
    #     x_with_noise = x * alpha.sqrt() + normal_noise * (1.0 - alpha).sqrt()
    #
    #     x_diffinput = x_with_noise + time_embedding
    #
    #     x_diffinput = x_diffinput.view(-1, x_dim)
    #
    #     encoded = self.mlpEncoder(x_diffinput)
    #     predicted_noise = self.mlpDecoder(encoded)
    #
    #     predicted_noise = predicted_noise.view(bs, seq_len, code_len, x_dim)
    #     x_diffinput = x_diffinput.view(bs, seq_len, code_len, x_dim)
    #
    #     gen_x = x_diffinput - predicted_noise
    #
    #     return x, gen_x

    def forward(self, x, label):
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
        y_embedding = self.y_embedding(label.to(torch.long)).sum(dim=-2).unsqueeze(1).expand(-1, x.shape[1], -1)

        x_diffinput = x_with_noise + time_embedding + y_embedding

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

    def __init__(self, vocab_size, d_model, dropout, dropout_emb, num_layers, num_heads, max_pos):
        super().__init__()
        self.embbedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.dropout = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.output_mlp = nn.Sequential(nn.Linear(d_model, 2))
        self.pooler = MaxPoolLayer()
        self.rnns = nn.LSTM(d_model, d_model, num_layers, bidirectional=False, batch_first=True, dropout=dropout)

    def forward(self, input_seqs, lengths):
        batch_size, seq_len, num_cui_per_visit = input_seqs.size()
        x = self.embbedding(input_seqs).sum(dim=2)
        x = self.emb_dropout(x)
        rnn_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        rnn_output, _ = self.rnns(rnn_input)
        x, _ = pad_packed_sequence(rnn_output, batch_first=True, total_length=seq_len)
        x = self.pooler(x, lengths)
        x = self.output_mlp(x)
        return x

class MaxPoolLayer(nn.Module):
    """
    A layer that performs max pooling along the sequence dimension
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, mask_or_lengths=None):
        """
        inputs: tensor of shape (batch_size, seq_len, hidden_size)
        mask_or_lengths: tensor of shape (batch_size) or (batch_size, seq_len)

        returns: tensor of shape (batch_size, hidden_size)
        """
        bs, sl, _ = inputs.size()
        if mask_or_lengths is not None:
            if len(mask_or_lengths.size()) == 1:
                mask = (torch.arange(sl, device=inputs.device).unsqueeze(0).expand(bs,
                                                                                   sl) >= mask_or_lengths.unsqueeze(
                    1))
            else:
                mask = mask_or_lengths
            inputs = inputs.masked_fill(mask.unsqueeze(-1).expand_as(inputs), float('-inf'))
        max_pooled = inputs.max(1)[0]
        return max_pooled

    # def __init__(self, vocab_size, x_dim, h_dim, dropout):
    #     super(LSTM_predictor, self).__init__()
    #     self.vocab_size = vocab_size
    #     self.x_dim = x_dim
    #     self.h_dim = h_dim
    #
    #     self.initial_embedding = nn.Embedding(self.vocab_size + 1, x_dim, padding_idx=-1)
    #     self.lstm = nn.LSTM(x_dim, h_dim, dropout=dropout, batch_first=True)
    #
    #     self.fc1 = nn.Linear(h_dim, int(h_dim / 2))
    #     self.bn1 = nn.BatchNorm1d(int(h_dim / 2))
    #     self.relu1 = nn.LeakyReLU()
    #
    #     self.fc2 = nn.Linear(int(h_dim / 2), 2)
    #     self.dropout = nn.Dropout(dropout)
    #
    # def forward(self, x):
    #     x = self.initial_embedding(x).sum(-2)
    #     output, (h_n, c_n) = self.lstm(x)
    #
    #     f = self.fc1(output[:, -1, :])
    #     f = self.bn1(f)
    #     f = self.relu1(f)
    #     f = self.dropout(f)
    #
    #     output = self.fc2(f)
    #     return output


