import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.baseline import Attention
import torch.nn.functional as F
from utils.diffUtil import get_beta_schedule
from models.toy import VisitLevelPositionalEncoder

class Discriminator(nn.Module):
    def __init__(self, d_model):
        super(Discriminator, self).__init__()

        # Define additional layers to increase model complexity
        self.layer1 = nn.Linear(d_model, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, 1)

        # Non-linearity and dropout for regularization
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.1)

        # Final activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Forward pass through the network
        x = self.leaky_relu(self.layer1(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.layer2(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.layer3(x))
        x = self.dropout(x)
        x = self.output_layer(x)
        return self.sigmoid(x)

class Linear_generator(nn.Module):
    def __init__(self, d_model=256, h_model=64):
        super(Linear_generator, self).__init__()
        self.linear1 = nn.Linear(d_model, h_model)
        self.linear2 = nn.Linear(h_model, d_model)
        self.batchnorm1 = nn.BatchNorm1d(h_model)
        self.batchnorm2 = nn.BatchNorm1d(d_model)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(0.1, inplace=False)

    def forward(self, z, t=None):
        # Reshape for batch norm: merge batch and seq_len dimensions
        original_shape = z.shape  # [batch_size, seq_len, d_model]
        z = z.reshape(-1, z.shape[-1])  # [batch_size * seq_len, d_model]

        if t is not None:
            t = t.reshape(-1, t.shape[-1])
            z = z + t
        # First layer with skip connection
        identity = z
        z = self.linear1(z)
        z = self.batchnorm1(z)
        z = self.relu(z)
        z = self.dropout(z)
        z = z + identity  # Skip connection

        # Second layer with skip connection
        identity = z
        z = self.linear2(z)
        z = self.batchnorm2(z)
        z = self.relu(z)
        z = z + identity  # Skip connection

        # Reshape back to original shape
        z = z.reshape(original_shape)

        return z

class VAE_generator(nn.Module):
    def __init__(self, d_model=256, l_model = 128, h_model=64):
        super(VAE_generator, self).__init__()
        self.d_model = d_model
        self.h_model = h_model
        self.l_model = l_model
        self.fc1 = nn.Linear(d_model, l_model)
        self.fc2 = nn.Linear(l_model, h_model)
        self.fc1_reverse = nn.Linear(l_model, d_model)
        self.fc2_reverse = nn.Linear(h_model, l_model)
        self.mean_layer = nn.Linear(h_model, h_model)
        self.logvar_layer = nn.Linear(h_model, h_model)
        self.relu = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()

        self.encoder = nn.Sequential(self.fc1, self.relu, self.fc2, self.relu)
        self.decoder = nn.Sequential(nn.Linear(h_model, h_model), self.relu, self.fc2_reverse, self.relu, self.fc1_reverse)

    def encode(self, x):
        x = self.encoder(x)
        return self.mean_layer(x), self.logvar_layer(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std, device=std.device)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        miu, logvar = self.encode(x)
        z = self.reparameterize(miu, logvar)
        return self.decode(z), miu, logvar

class Diff_generator(nn.Module):
    def __init__(self, d_model=256, dropout=0.1, device = 'cpu'):
        super(Diff_generator, self).__init__()
        self.device = device
        self.d_model = d_model
        betas = get_beta_schedule(beta_schedule="linear",
                                  beta_start=0.0001,
                                  beta_end=0.02,
                                  num_diffusion_timesteps=1000)
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.diffusion_num_timesteps = betas.shape[0]
        self.encoder = nn.Sequential(nn.Linear(d_model, int(d_model/2)), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(int(d_model/2), int(d_model/4)), nn.ReLU(), nn.Dropout(dropout))
        self.decoder = nn.Sequential(nn.Linear(int(d_model/4), int(d_model/2)), nn.ReLU(), nn.Dropout(dropout),
                                        nn.Linear(int(d_model/2), d_model), nn.ReLU())
    def denoise(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

    def forward(self, x, lengths):
        diffusion_time_t = torch.randint(
            low=0, high=1000, size=[x.shape[0], ]).to(
            self.device)
        alpha = (1 - self.betas).cumprod(dim=0).index_select(0, diffusion_time_t).view(-1, 1, 1)
        z = torch.randn_like(x)
        added_z = z * (1.0 - alpha).sqrt()
        x_hat = x * alpha.sqrt() + added_z

        max_length = x.size(1)
        mask = torch.arange(max_length, device=x.device).expand(len(lengths), max_length) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1)
        x_masked = x_hat * mask.float()
        x_denoised = self.denoise(x_masked)
        x_denoised_masked = x_denoised * mask.float()
        learned_noise = x_masked - x_denoised_masked

        return x_denoised_masked, added_z * mask.float(), learned_noise
class EVA(nn.Module):
    def __init__(self, name, vocab_list, d_model, dropout, generator):
        super(EVA, self).__init__()
        self.name = name
        self.dropout = nn.Dropout(dropout)
        self.visit_generator = generator
        self.diag_embedding = nn.Embedding(vocab_list[0]+1, d_model)
        self.drug_embedding = nn.Embedding(vocab_list[1]+1, d_model)
        self.lab_embedding = nn.Embedding(vocab_list[2]+1, d_model)
        self.proc_embedding = nn.Embedding(vocab_list[3]+1, d_model)
        self.diag_output_mlp = nn.Sequential(nn.Linear(d_model,d_model), nn.ReLU(), nn.Dropout(0.5), nn.Linear(d_model, vocab_list[0]))
        self.drug_output_mlp = nn.Sequential(nn.Linear(d_model,d_model), nn.ReLU(), nn.Dropout(0.5), nn.Linear(d_model, vocab_list[1]))
        self.lab_output_mlp = nn.Sequential(nn.Linear(d_model,d_model), nn.ReLU(), nn.Dropout(0.5), nn.Linear(d_model, vocab_list[2]))
        self.proc_output_mlp = nn.Sequential(nn.Linear(d_model,d_model), nn.ReLU(), nn.Dropout(0.5), nn.Linear(d_model, vocab_list[3]))
        self.hidden_state_learner = CNN_hidden_state_learner(d_model, dropout)
    def forward(self, diag_seq, drug_seq, lab_seq, proc_seq, lengths=None):
        batch_size, seq_len, code_len = diag_seq.shape
        diag_v = self.dropout(self.diag_embedding(diag_seq)).sum(dim=-2)
        drug_v = self.dropout(self.drug_embedding(drug_seq)).sum(dim=-2)
        lab_v = self.dropout(self.lab_embedding(lab_seq)).sum(dim=-2)
        proc_v = self.dropout(self.proc_embedding(proc_seq)).sum(dim=-2)
        #each modalities have shape [batch_size, seq_len, d_model]

        v = torch.cat([diag_v, drug_v, lab_v, proc_v], dim=-1).view(batch_size, seq_len, 4, -1).sum(dim=-2)
        #after concatenation and summation, v has shape [batch_size, seq_len, d_model]

        #here we use CNN hidden state learner to learn the hidden state for each visit
        h = self.hidden_state_learner(v, lengths)

        h_miu, h_logvar = self.visit_generator.encode(h)

        z = self.visit_generator.reparameterize(h_miu, h_logvar)

        v_gen = self.visit_generator.decode(z)

        diag_logits = self.diag_output_mlp(v_gen)
        drug_logits = self.drug_output_mlp(v_gen)
        lab_logits = self.lab_output_mlp(v_gen)
        proc_logits = self.proc_output_mlp(v_gen)

        return diag_logits, drug_logits, lab_logits, proc_logits, h_miu, h_logvar
class TWIN(nn.Module):
    def __init__(self, name, vocab_list, d_model, dropout, K, generator):
        super(TWIN, self).__init__()
        self.name = name
        self.dropout = nn.Dropout(dropout)
        self.visit_generator = generator
        self.K = K
        self.diag_embedding = nn.Embedding(vocab_list[0]+1, d_model)
        self.drug_embedding = nn.Embedding(vocab_list[1]+1, d_model)
        self.lab_embedding = nn.Embedding(vocab_list[2]+1, d_model)
        self.proc_embedding = nn.Embedding(vocab_list[3]+1, d_model)
        self.diag_output_mlp = nn.Sequential(nn.Linear(d_model,d_model), nn.ReLU(), nn.Dropout(0.5), nn.Linear(d_model, vocab_list[0]))
        self.drug_output_mlp = nn.Sequential(nn.Linear(d_model,d_model), nn.ReLU(), nn.Dropout(0.5), nn.Linear(d_model, vocab_list[1]))
        self.lab_output_mlp = nn.Sequential(nn.Linear(d_model,d_model), nn.ReLU(), nn.Dropout(0.5), nn.Linear(d_model, vocab_list[2]))
        self.proc_output_mlp = nn.Sequential(nn.Linear(d_model,d_model), nn.ReLU(), nn.Dropout(0.5), nn.Linear(d_model, vocab_list[3]))
        #TWIN does not have hidden state learner
        #extra time embedding and prediction head for rebuttal
        self.time_embedding = VisitLevelPositionalEncoder(d_model)
        # self.time_output_mlp = nn.Sequential(nn.Linear(d_model * 4,d_model), nn.ReLU(), nn.Dropout(0.5), nn.Linear(d_model, 1))
        self.time_output_mlp1 = nn.Sequential(nn.Linear(d_model * 4,d_model), nn.ReLU(), nn.Dropout(0.5))
        self.time_output_mlp2 = nn.Linear(d_model, 1)

    def retrive_top_k_within_batch(self, h):

        batch_size, seq_len, d_model = h.shape
        h_reshaped = h.reshape(batch_size * seq_len, d_model)

        sim_matrix = torch.matmul(h_reshaped, h_reshaped.transpose(0,1))
        mask = torch.eye(batch_size * seq_len, device=h.device)
        sim_matrix.masked_fill_(mask.bool(), float('-inf'))
        top_k_values, top_k_indices = torch.topk(sim_matrix, self.K, dim=-1)

        top_k_values = top_k_values.reshape(batch_size, seq_len, self.K)
        top_k_indices = top_k_indices.reshape(batch_size, seq_len, self.K)

        return top_k_values, top_k_indices

    def calculate_attention(self, h, top_k_indices, top_k_values):
        batch_size, seq_len, d_model = h.shape
        h_reshaped = h.reshape(batch_size * seq_len, d_model)
        top_k_indices_reshaped = top_k_indices.reshape(batch_size * seq_len, self.K)
        top_k_values_softmax = F.softmax(top_k_values, dim=-1)

        selected_h = torch.zeros(batch_size * seq_len, self.K, d_model, device=h.device)
        for i in range(self.K):
            selected_h[:, i, :] = h_reshaped[top_k_indices_reshaped[:, i]]
        selected_h = selected_h.reshape(batch_size, seq_len, self.K, d_model)
        weighted_h = selected_h * top_k_values_softmax.unsqueeze(-1)
        weighted_h = weighted_h.sum(dim=-2)
        return weighted_h + h

    def forward(self, diag_seq, drug_seq, lab_seq, proc_seq, lengths=None, time=None):
        batch_size, seq_len, code_len = diag_seq.shape
        diag_h = self.dropout(self.diag_embedding(diag_seq)).sum(dim=-2)
        drug_h = self.dropout(self.drug_embedding(drug_seq)).sum(dim=-2)
        lab_h = self.dropout(self.lab_embedding(lab_seq)).sum(dim=-2)
        proc_h = self.dropout(self.proc_embedding(proc_seq)).sum(dim=-2)

        if time is not None:
            time = self.time_embedding(time)
            diag_h = diag_h + time
            drug_h = drug_h + time
            lab_h = lab_h + time
            proc_h = proc_h + time

        diag_top_k_h, diag_top_k_indices = self.retrive_top_k_within_batch(diag_h)
        drug_top_k_h, drug_top_k_indices = self.retrive_top_k_within_batch(drug_h)
        lab_top_k_h, lab_top_k_indices = self.retrive_top_k_within_batch(lab_h)
        proc_top_k_h, proc_top_k_indices = self.retrive_top_k_within_batch(proc_h)

        diag_h_bar = self.calculate_attention(diag_h, diag_top_k_indices, diag_top_k_h)
        drug_h_bar = self.calculate_attention(drug_h, drug_top_k_indices, drug_top_k_h)
        lab_h_bar = self.calculate_attention(lab_h, lab_top_k_indices, lab_top_k_h)
        proc_h_bar = self.calculate_attention(proc_h, proc_top_k_indices, proc_top_k_h)

        diag_h_miu, diag_h_logvar = self.visit_generator.encode(diag_h_bar)
        drug_h_miu, drug_h_logvar = self.visit_generator.encode(drug_h_bar)
        lab_h_miu, lab_h_logvar = self.visit_generator.encode(lab_h_bar)
        proc_h_miu, proc_h_logvar = self.visit_generator.encode(proc_h_bar)

        diag_z = self.visit_generator.reparameterize(diag_h_miu, diag_h_logvar)
        drug_z = self.visit_generator.reparameterize(drug_h_miu + diag_h_miu, drug_h_logvar + diag_h_logvar)
        lab_z = self.visit_generator.reparameterize(lab_h_miu + diag_h_miu, lab_h_logvar + diag_h_logvar)
        proc_z = self.visit_generator.reparameterize(proc_h_miu + diag_h_miu, proc_h_logvar + diag_h_logvar)

        diag_v_gen = self.visit_generator.decode(diag_z)
        drug_v_gen = self.visit_generator.decode(drug_z)
        lab_v_gen = self.visit_generator.decode(lab_z)
        proc_v_gen = self.visit_generator.decode(proc_z)

        diag_logits = self.diag_output_mlp(diag_v_gen)
        drug_logits = self.drug_output_mlp(drug_v_gen)
        lab_logits = self.lab_output_mlp(lab_v_gen)
        proc_logits = self.proc_output_mlp(proc_v_gen)

        if time is not None:
            time_pred = self.time_output_mlp1(torch.cat([diag_h, drug_h, lab_h, proc_h], dim=-1))
            time_pred = self.time_output_mlp2(time_pred).squeeze(-1)
        else:
            time_pred = None

        concat_miu = torch.cat([diag_h_miu, drug_h_miu, lab_h_miu, proc_h_miu], dim=-1)
        concat_logvar = torch.cat([diag_h_logvar, drug_h_logvar, lab_h_logvar, proc_h_logvar], dim=-1)

        return diag_logits, drug_logits, lab_logits, proc_logits, concat_miu, concat_logvar, time_pred

class synTEG(nn.Module):
    def __init__(self, name, vocab_list, d_model, dropout, generator):
        super(synTEG, self).__init__()
        self.name = name
        self.dropout = nn.Dropout(dropout)
        self.visit_generator = generator
        self.diag_embedding = nn.Embedding(vocab_list[0]+1, d_model)
        self.drug_embedding = nn.Embedding(vocab_list[1]+1, d_model)
        self.lab_embedding = nn.Embedding(vocab_list[2]+1, d_model)
        self.proc_embedding = nn.Embedding(vocab_list[3]+1, d_model)
        self.diag_output_mlp = nn.Sequential(nn.Linear(d_model,d_model), nn.ReLU(), nn.Dropout(0.5), nn.Linear(d_model, vocab_list[0]))
        self.drug_output_mlp = nn.Sequential(nn.Linear(d_model,d_model), nn.ReLU(), nn.Dropout(0.5), nn.Linear(d_model, vocab_list[1]))
        self.lab_output_mlp = nn.Sequential(nn.Linear(d_model,d_model), nn.ReLU(), nn.Dropout(0.5), nn.Linear(d_model, vocab_list[2]))
        self.proc_output_mlp = nn.Sequential(nn.Linear(d_model,d_model), nn.ReLU(), nn.Dropout(0.5), nn.Linear(d_model, vocab_list[3]))
        self.hiddne_state_learner = Transformer_hidden_state_learner(d_model, 2, dropout)

    def forward(self, diag_seq, drug_seq, lab_seq, proc_seq, lengths=None):
        batch_size, seq_len, code_len = diag_seq.shape
        diag_v = self.dropout(self.diag_embedding(diag_seq)).sum(dim=-2)
        drug_v = self.dropout(self.drug_embedding(drug_seq)).sum(dim=-2)
        lab_v = self.dropout(self.lab_embedding(lab_seq)).sum(dim=-2)
        proc_v = self.dropout(self.proc_embedding(proc_seq)).sum(dim=-2)

        v = torch.cat([diag_v, drug_v, lab_v, proc_v], dim=-1).view(batch_size, seq_len, 4, -1).sum(dim=-2)

        h = self.hiddne_state_learner(v, lengths)

        v_gen = self.visit_generator(h.clone())

        real_diag_logits = self.diag_output_mlp(h.clone())
        real_drug_logits = self.drug_output_mlp(h.clone())
        real_lab_logits = self.lab_output_mlp(h.clone())
        real_proc_logits = self.proc_output_mlp(h.clone())

        gen_diag_logits = self.diag_output_mlp(v_gen.clone())
        gen_drug_logits = self.drug_output_mlp(v_gen.clone())
        gen_lab_logits = self.lab_output_mlp(v_gen.clone())
        gen_proc_logits = self.proc_output_mlp(v_gen.clone())

        return real_diag_logits, real_drug_logits, real_lab_logits, real_proc_logits, gen_diag_logits, gen_drug_logits, gen_lab_logits, gen_proc_logits, h, v_gen

class LSTM_TabDDPM(nn.Module):

    def __init__(self, name, vocab_list, d_model, dropout, generator):
        super(LSTM_TabDDPM, self).__init__()
        self.name = name
        self.dropout = nn.Dropout(dropout)
        self.visit_generator = generator
        self.diag_embedding = nn.Embedding(vocab_list[0]+1, d_model)
        self.drug_embedding = nn.Embedding(vocab_list[1]+1, d_model)
        self.lab_embedding = nn.Embedding(vocab_list[2]+1, d_model)
        self.proc_embedding = nn.Embedding(vocab_list[3]+1, d_model)
        self.diag_output_mlp = nn.Sequential(nn.Linear(d_model,d_model), nn.ReLU(), nn.Dropout(0.5), nn.Linear(d_model, vocab_list[0]))
        self.drug_output_mlp = nn.Sequential(nn.Linear(d_model,d_model), nn.ReLU(), nn.Dropout(0.5), nn.Linear(d_model, vocab_list[1]))
        self.lab_output_mlp = nn.Sequential(nn.Linear(d_model,d_model), nn.ReLU(), nn.Dropout(0.5), nn.Linear(d_model, vocab_list[2]))
        self.proc_output_mlp = nn.Sequential(nn.Linear(d_model,d_model), nn.ReLU(), nn.Dropout(0.5), nn.Linear(d_model, vocab_list[3]))
        self.lstm = nn.LSTM(d_model, d_model, 1, bidirectional=False, batch_first=True, dropout=dropout)

    def hidden_state_learner(self, v, lengths):
        h, _ = self.lstm(v)
        return h

    def forward(self, diag_seq, drug_seq, lab_seq, proc_seq, lengths=None):
        batch_size, seq_len, code_len = diag_seq.shape
        diag_v = self.dropout(self.diag_embedding(diag_seq)).sum(dim=-2)
        drug_v = self.dropout(self.drug_embedding(drug_seq)).sum(dim=-2)
        lab_v = self.dropout(self.lab_embedding(lab_seq)).sum(dim=-2)
        proc_v = self.dropout(self.proc_embedding(proc_seq)).sum(dim=-2)

        v = torch.cat([diag_v, drug_v, lab_v, proc_v], dim=-1).view(batch_size, seq_len, 4, -1).sum(dim=-2)

        h = self.hidden_state_learner(v, lengths)

        v_gen, added_noise, learned_noise = self.visit_generator(h.clone(), lengths)

        real_diag_logits = self.diag_output_mlp(h.clone())
        real_drug_logits = self.drug_output_mlp(h.clone())
        real_lab_logits = self.lab_output_mlp(h.clone())
        real_proc_logits = self.proc_output_mlp(h.clone())

        gen_diag_logits = self.diag_output_mlp(v_gen.clone())
        gen_drug_logits = self.drug_output_mlp(v_gen.clone())
        gen_lab_logits = self.lab_output_mlp(v_gen.clone())
        gen_proc_logits = self.proc_output_mlp(v_gen.clone())


        return real_diag_logits, real_drug_logits, real_lab_logits, real_proc_logits, gen_diag_logits, gen_drug_logits, gen_lab_logits, gen_proc_logits, h, v_gen, added_noise, learned_noise

class LSTM_ScoEHR(nn.Module):

    def __init__(self, name, vocab_list, d_model, dropout, generator):
        super(LSTM_ScoEHR, self).__init__()
        self.name = name
        self.dropout = nn.Dropout(dropout)
        self.visit_generator = generator
        self.diag_embedding = nn.Embedding(vocab_list[0]+1, d_model)
        self.drug_embedding = nn.Embedding(vocab_list[1]+1, d_model)
        self.lab_embedding = nn.Embedding(vocab_list[2]+1, d_model)
        self.proc_embedding = nn.Embedding(vocab_list[3]+1, d_model)
        self.diag_output_mlp = nn.Sequential(nn.Linear(d_model,d_model), nn.ReLU(), nn.Dropout(0.1), nn.Linear(d_model, vocab_list[0]))
        self.drug_output_mlp = nn.Sequential(nn.Linear(d_model,d_model), nn.ReLU(), nn.Dropout(0.1), nn.Linear(d_model, vocab_list[1]))
        self.lab_output_mlp = nn.Sequential(nn.Linear(d_model,d_model), nn.ReLU(), nn.Dropout(0.1), nn.Linear(d_model, vocab_list[2]))
        self.proc_output_mlp = nn.Sequential(nn.Linear(d_model,d_model), nn.ReLU(), nn.Dropout(0.1), nn.Linear(d_model, vocab_list[3]))
        self.four_modal_fc = nn.Sequential(nn.Linear(4*d_model, d_model), nn.ReLU(), nn.Dropout(0.1), nn.Linear(d_model, d_model))
        self.lstm = nn.LSTM(d_model, d_model, 1, bidirectional=False, batch_first=True, dropout=dropout)

    def hidden_state_learner(self, v, lengths):
        h, _ = self.lstm(v)
        return h

    def forward(self, diag_seq, drug_seq, lab_seq, proc_seq, lengths=None):
        batch_size, seq_len, code_len = diag_seq.shape
        diag_v = self.dropout(self.diag_embedding(diag_seq)).sum(dim=-2)
        drug_v = self.dropout(self.drug_embedding(drug_seq)).sum(dim=-2)
        lab_v = self.dropout(self.lab_embedding(lab_seq)).sum(dim=-2)
        proc_v = self.dropout(self.proc_embedding(proc_seq)).sum(dim=-2)

        v = torch.cat([diag_v, drug_v, lab_v, proc_v], dim=-1)
        v = self.four_modal_fc(v)

        h = self.hidden_state_learner(v, lengths)

        v_gen, added_noise, learned_noise = self.visit_generator(h.clone(), lengths)

        real_diag_logits = self.diag_output_mlp(h.clone())
        real_drug_logits = self.drug_output_mlp(h.clone())
        real_lab_logits = self.lab_output_mlp(h.clone())
        real_proc_logits = self.proc_output_mlp(h.clone())

        gen_diag_logits = self.diag_output_mlp(v_gen.clone())
        gen_drug_logits = self.drug_output_mlp(v_gen.clone())
        gen_lab_logits = self.lab_output_mlp(v_gen.clone())
        gen_proc_logits = self.proc_output_mlp(v_gen.clone())


        return real_diag_logits, real_drug_logits, real_lab_logits, real_proc_logits, gen_diag_logits, gen_drug_logits, gen_lab_logits, gen_proc_logits, h, v_gen, added_noise, learned_noise

class LSTM_Meddiff(nn.Module):
    def __init__(self, name, vocab_list, d_model, dropout, generator):
        super(LSTM_Meddiff, self).__init__()
        self.d_model = d_model
        self.name = name
        self.dropout = nn.Dropout(dropout)
        self.visit_generator = generator
        self.diag_embedding = nn.Embedding(vocab_list[0]+1, d_model)
        self.drug_embedding = nn.Embedding(vocab_list[1]+1, d_model)
        self.lab_embedding = nn.Embedding(vocab_list[2]+1, d_model)
        self.proc_embedding = nn.Embedding(vocab_list[3]+1, d_model)
        self.diag_output_mlp = nn.Sequential(nn.Linear(d_model,d_model), nn.ReLU(), nn.Dropout(0.1), nn.Linear(d_model, vocab_list[0]))
        self.drug_output_mlp = nn.Sequential(nn.Linear(d_model,d_model), nn.ReLU(), nn.Dropout(0.1), nn.Linear(d_model, vocab_list[1]))
        self.lab_output_mlp = nn.Sequential(nn.Linear(d_model,d_model), nn.ReLU(), nn.Dropout(0.1), nn.Linear(d_model, vocab_list[2]))
        self.proc_output_mlp = nn.Sequential(nn.Linear(d_model,d_model), nn.ReLU(), nn.Dropout(0.1), nn.Linear(d_model, vocab_list[3]))
        self.lstm = nn.LSTM(d_model * 4, d_model* 4, 1, bidirectional=False, batch_first=True, dropout=dropout)

    def hidden_state_learner(self, v, lengths):
        h, _ = self.lstm(v)
        return h

    def forward(self, diag_seq, drug_seq, lab_seq, proc_seq, lengths=None):
        batch_size, seq_len, code_len = diag_seq.shape
        diag_v = self.dropout(self.diag_embedding(diag_seq)).sum(dim=-2)
        drug_v = self.dropout(self.drug_embedding(drug_seq)).sum(dim=-2)
        lab_v = self.dropout(self.lab_embedding(lab_seq)).sum(dim=-2)
        proc_v = self.dropout(self.proc_embedding(proc_seq)).sum(dim=-2)

        v = torch.cat([diag_v, drug_v, lab_v, proc_v], dim=-1)

        h = self.hidden_state_learner(v, lengths)

        v_gen, added_noise, learned_noise = self.visit_generator(h.clone(), lengths)

        real_diag_logits = self.diag_output_mlp(h[:, :, :self.d_model])
        real_drug_logits = self.drug_output_mlp(h[:, :, self.d_model:2*self.d_model])
        real_lab_logits = self.lab_output_mlp(h[:, :, 2*self.d_model:3*self.d_model])
        real_proc_logits = self.proc_output_mlp(h[:, :, 3*self.d_model:])

        gen_diag_logits = self.diag_output_mlp(v_gen[:, :, :self.d_model])
        gen_drug_logits = self.drug_output_mlp(v_gen[:, :, self.d_model:2*self.d_model])
        gen_lab_logits = self.lab_output_mlp(v_gen[:, :, 2*self.d_model:3*self.d_model])
        gen_proc_logits = self.proc_output_mlp(v_gen[:, :, 3*self.d_model:])

        return real_diag_logits, real_drug_logits, real_lab_logits, real_proc_logits, gen_diag_logits, gen_drug_logits, gen_lab_logits, gen_proc_logits, h, v_gen, added_noise, learned_noise


class LSTM_MLP(nn.Module):
    def __init__(self, name, vocab_list, d_model, dropout, generator=None):
        super(LSTM_MLP, self).__init__()
        self.name = name
        self.dropout = nn.Dropout(dropout)
        self.visit_generator = generator
        self.diag_embedding = nn.Embedding(vocab_list[0]+1, d_model)
        self.drug_embedding = nn.Embedding(vocab_list[1]+1, d_model)
        self.lab_embedding = nn.Embedding(vocab_list[2]+1, d_model)
        self.proc_embedding = nn.Embedding(vocab_list[3]+1, d_model)
        self.diag_output_mlp = nn.Sequential(nn.Linear(d_model,d_model), nn.ReLU(), nn.Dropout(0.5), nn.Linear(d_model, vocab_list[0]))
        self.drug_output_mlp = nn.Sequential(nn.Linear(d_model,d_model), nn.ReLU(), nn.Dropout(0.5), nn.Linear(d_model, vocab_list[1]))
        self.lab_output_mlp = nn.Sequential(nn.Linear(d_model,d_model), nn.ReLU(), nn.Dropout(0.5), nn.Linear(d_model, vocab_list[2]))
        self.proc_output_mlp = nn.Sequential(nn.Linear(d_model,d_model), nn.ReLU(), nn.Dropout(0.5), nn.Linear(d_model, vocab_list[3]))
        self.lstm = nn.LSTM(d_model, d_model, 1, bidirectional=False, batch_first=True, dropout=dropout)

    def hidden_state_learner(self, v, lengths):
        h, _ = self.lstm(v)
        return h

    def forward(self, diag_seq, drug_seq, lab_seq, proc_seq, lengths=None):
        batch_size, seq_len, code_len = diag_seq.shape
        diag_v = self.dropout(self.diag_embedding(diag_seq)).sum(dim=-2)
        drug_v = self.dropout(self.drug_embedding(drug_seq)).sum(dim=-2)
        lab_v = self.dropout(self.lab_embedding(lab_seq)).sum(dim=-2)
        proc_v = self.dropout(self.proc_embedding(proc_seq)).sum(dim=-2)

        v = torch.cat([diag_v, drug_v, lab_v, proc_v], dim=-1).view(batch_size, seq_len, 4, -1).sum(dim=-2)

        h = self.hidden_state_learner(v, lengths)

        real_diag_logits = self.diag_output_mlp(h.clone())
        real_drug_logits = self.drug_output_mlp(h.clone())
        real_lab_logits = self.lab_output_mlp(h.clone())
        real_proc_logits = self.proc_output_mlp(h.clone())

        return real_diag_logits, real_drug_logits, real_lab_logits, real_proc_logits, None, None, None, None, h, None

class LSTM_medGAN(nn.Module):
    def __init__(self, name, vocab_list, d_model, dropout, generator):
        super(LSTM_medGAN, self).__init__()
        self.name = name
        self.dropout = nn.Dropout(dropout)
        self.visit_generator = generator
        self.diag_embedding = nn.Embedding(vocab_list[0]+1, d_model)
        self.drug_embedding = nn.Embedding(vocab_list[1]+1, d_model)
        self.lab_embedding = nn.Embedding(vocab_list[2]+1, d_model)
        self.proc_embedding = nn.Embedding(vocab_list[3]+1, d_model)
        self.diag_output_mlp = nn.Sequential(nn.Linear(d_model,d_model), nn.ReLU(), nn.Dropout(0.5), nn.Linear(d_model, vocab_list[0]))
        self.drug_output_mlp = nn.Sequential(nn.Linear(d_model,d_model), nn.ReLU(), nn.Dropout(0.5), nn.Linear(d_model, vocab_list[1]))
        self.lab_output_mlp = nn.Sequential(nn.Linear(d_model,d_model), nn.ReLU(), nn.Dropout(0.5), nn.Linear(d_model, vocab_list[2]))
        self.proc_output_mlp = nn.Sequential(nn.Linear(d_model,d_model), nn.ReLU(), nn.Dropout(0.5), nn.Linear(d_model, vocab_list[3]))
        self.visit_time_embedding = VisitLevelPositionalEncoder(d_model)
        self.lstm = nn.LSTM(d_model, d_model, 1, bidirectional=False, batch_first=True, dropout=dropout)

    def hidden_state_learner(self, v, lengths):
        # batch_size, seq_len, d_model = v.shape
        # rnn_input = pack_padded_sequence(v, lengths.cpu(), batch_first=True, enforce_sorted=False)
        # rnn_output, _ = self.lstm(rnn_input)
        # h, _ = pad_packed_sequence(rnn_output, batch_first=True, total_length=seq_len)
        h, _ = self.lstm(v)
        return h

    def forward(self, diag_seq, drug_seq, lab_seq, proc_seq, lengths=None, time=None):

        #assuming diag_seq, drug_seq, lab_seq, proc_seq are all have shape [batch_size, seq_len, code_len]
        batch_size, seq_len, code_len = diag_seq.shape
        diag_v = self.diag_embedding(diag_seq)
        diag_v = self.dropout(diag_v)
        drug_v = self.drug_embedding(drug_seq)
        drug_v = self.dropout(drug_v)
        lab_v = self.lab_embedding(lab_seq)
        lab_v = self.dropout(lab_v)
        proc_v = self.proc_embedding(proc_seq)
        proc_v = self.dropout(proc_v)

        diag_v = diag_v.sum(dim=-2) #all modalities should have shape [batch_size, seq_len, d_model]
        drug_v = drug_v.sum(dim=-2)
        lab_v = lab_v.sum(dim=-2)
        proc_v = proc_v.sum(dim=-2)

        #concatenate all modalities and get visit representation
        v = torch.cat([diag_v, drug_v, lab_v, proc_v], dim=-1).view(batch_size, seq_len, 4, -1).sum(dim=-2) #shape [batch_size, seq_len, d_model]

        if time is not None:
            t = self.visit_time_embedding(time)


        #put in hidden state learner and get hidden state for each visit, which also has shape [batch_size, seq_len, d_model]
        h = self.hidden_state_learner(v, lengths)

        z = torch.rand_like(h)

        #assuming the generator will take in the hidden state and output a synthetic one.
        if time is not None:
            v_gen = self.visit_generator(z, t)
        else:
            v_gen = self.visit_generator(z, None)

        #both real v and gen_v predict the next visit

        real_diag_logits = self.diag_output_mlp(h.clone())
        real_drug_logits = self.drug_output_mlp(h.clone())
        real_lab_logits = self.lab_output_mlp(h.clone())
        real_proc_logits = self.proc_output_mlp(h.clone())

        gen_diag_logits = self.diag_output_mlp(v_gen.clone())
        gen_drug_logits = self.drug_output_mlp(v_gen.clone())
        gen_lab_logits = self.lab_output_mlp(v_gen.clone())
        gen_proc_logits = self.proc_output_mlp(v_gen.clone())


        return real_diag_logits, real_drug_logits, real_lab_logits, real_proc_logits, gen_diag_logits, gen_drug_logits, gen_lab_logits, gen_proc_logits, h, v_gen



class LSTM_hidden_state_learner(nn.Module):
    def __init__(self, d_model, dropout):
        super(LSTM_hidden_state_learner, self).__init__()
        self.lstm = nn.LSTM(d_model, d_model, 1, bidirectional=False, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, v, lengths):
        #assuming v is the visit representation with shape [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = v.shape
        rnn_input = pack_padded_sequence(v, lengths.cpu(), batch_first=True, enforce_sorted=False)
        rnn_output, _ = self.lstm(rnn_input)
        h, _ = pad_packed_sequence(rnn_output, batch_first=True, total_length=seq_len)
        return h


class CNN_hidden_state_learner(nn.Module):
    def __init__(self, d_model, dropout):
        super(CNN_hidden_state_learner, self).__init__()
        self.d_model = d_model

        # Define multiple convolutional layers with increasing dilation
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=4, dilation=4)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(d_model)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.bn3 = nn.BatchNorm1d(d_model)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, v, lengths):
        batch_size, seq_len, d_model = v.shape
        v = v.transpose(1, 2)  # Change to [batch_size, d_model, seq_len]

        # Apply convolutions
        v = self.conv1(v)
        v = self.relu(self.bn1(v))
        v = self.conv2(v)
        v = self.relu(self.bn2(v))
        v = self.conv3(v)
        v = self.relu(self.bn3(v))

        # Dropout
        v = self.dropout(v)

        # Masking
        mask = torch.arange(seq_len).to(v.device).expand(len(lengths), seq_len) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1)

        v = v.transpose(1, 2)
        v = v * mask.float()
        return v


class Transformer_hidden_state_learner(nn.Module):
    def __init__(self, d_model, num_head, dropout):
        super(Transformer_hidden_state_learner, self).__init__()
        self.attention = Attention(d_model, num_head, dropout)
    def forward(self, v, lengths):
        batch_size, seq_len, d_model = v.shape
        if lengths is not None:
            mask = torch.arange(seq_len).to(v.device).expand(len(lengths), seq_len) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(1).expand(-1, seq_len, -1)
            v, _ = self.attention(v, v, v, attn_mask=mask)
        else:
            v, _ = self.attention(v, v, v, None)
        return v