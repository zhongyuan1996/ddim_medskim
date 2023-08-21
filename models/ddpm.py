import torch
import torch.nn as nn
from utils.diffUtil import get_beta_schedule
from models.unet import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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

    def __init__(self, vocab_size, d_model, dropout, dropout_emb, num_layers, num_heads, max_pos):
        super().__init__()
        self.embbedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=-1)
        self.dropout = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.output_mlp = nn.Sequential(nn.Linear(d_model, 2))
        self.pooler = MaxPoolLayer()
        self.rnns = nn.LSTM(d_model, d_model, num_layers, bidirectional=False, batch_first=True, dropout=dropout)

    def forward(self, input_seqs, masks, lengths, seq_time_step, code_masks):
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
                mask = (torch.arange(sl, device=inputs.device).unsqueeze(0).expand(bs, sl) >= mask_or_lengths.unsqueeze(
                    1))
            else:
                mask = mask_or_lengths
            inputs = inputs.masked_fill(mask.unsqueeze(-1).expand_as(inputs), float('-inf'))
        max_pooled = inputs.max(1)[0]
        return max_pooled


