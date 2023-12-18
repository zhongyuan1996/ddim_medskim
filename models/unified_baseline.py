import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.baseline import Attention


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

    def forward(self, z):
        # Reshape for batch norm: merge batch and seq_len dimensions
        original_shape = z.shape  # [batch_size, seq_len, d_model]
        z = z.reshape(-1, z.shape[-1])  # [batch_size * seq_len, d_model]

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
        self.diag_output_mlp = nn.Sequential(nn.Linear(d_model, vocab_list[0]))
        self.drug_output_mlp = nn.Sequential(nn.Linear(d_model, vocab_list[1]))
        self.lab_output_mlp = nn.Sequential(nn.Linear(d_model, vocab_list[2]))
        self.proc_output_mlp = nn.Sequential(nn.Linear(d_model, vocab_list[3]))
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
        self.diag_output_mlp = nn.Sequential(nn.Linear(d_model, vocab_list[0]))
        self.drug_output_mlp = nn.Sequential(nn.Linear(d_model, vocab_list[1]))
        self.lab_output_mlp = nn.Sequential(nn.Linear(d_model, vocab_list[2]))
        self.proc_output_mlp = nn.Sequential(nn.Linear(d_model, vocab_list[3]))
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
        self.diag_output_mlp = nn.Sequential(nn.Linear(d_model, vocab_list[0]))
        self.drug_output_mlp = nn.Sequential(nn.Linear(d_model, vocab_list[1]))
        self.lab_output_mlp = nn.Sequential(nn.Linear(d_model, vocab_list[2]))
        self.proc_output_mlp = nn.Sequential(nn.Linear(d_model, vocab_list[3]))
        self.lstm = nn.LSTM(d_model, d_model, 1, bidirectional=False, batch_first=True, dropout=dropout)

    def hidden_state_learner(self, v, lengths):
        # batch_size, seq_len, d_model = v.shape
        # rnn_input = pack_padded_sequence(v, lengths.cpu(), batch_first=True, enforce_sorted=False)
        # rnn_output, _ = self.lstm(rnn_input)
        # h, _ = pad_packed_sequence(rnn_output, batch_first=True, total_length=seq_len)
        h, _ = self.lstm(v)
        return h

    def forward(self, diag_seq, drug_seq, lab_seq, proc_seq, lengths=None):

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

        #put in hidden state learner and get hidden state for each visit, which also has shape [batch_size, seq_len, d_model]
        h = self.hidden_state_learner(v, lengths)

        #assuming the generator will take in the hidden state and output a synthetic one.

        if self.name == 'GAN':
            v_gen = self.visit_generator(h.clone())
        elif self.name == 'DIFF':
            v_gen, added_noise, learned_noise = self.visit_generator(h.clone())

        #both real v and gen_v predict the next visit

        real_diag_logits = self.diag_output_mlp(h.clone())
        real_drug_logits = self.drug_output_mlp(h.clone())
        real_lab_logits = self.lab_output_mlp(h.clone())
        real_proc_logits = self.proc_output_mlp(h.clone())

        gen_diag_logits = self.diag_output_mlp(v_gen.clone())
        gen_drug_logits = self.drug_output_mlp(v_gen.clone())
        gen_lab_logits = self.lab_output_mlp(v_gen.clone())
        gen_proc_logits = self.proc_output_mlp(v_gen.clone())

        if self.name == 'GAN':
            return real_diag_logits, real_drug_logits, real_lab_logits, real_proc_logits, gen_diag_logits, gen_drug_logits, gen_lab_logits, gen_proc_logits, h, v_gen
        elif self.name == 'DIFF':
            return real_diag_logits, real_drug_logits, real_lab_logits, real_proc_logits, gen_diag_logits, gen_drug_logits, gen_lab_logits, gen_proc_logits, h, v_gen, added_noise, learned_noise


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
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, v, lengths):
        batch_size, seq_len, d_model = v.shape
        v = v.transpose(1, 2)
        v = self.conv1(v)
        v = self.dropout(v)
        mask = torch.arange(seq_len).expand(len(lengths), seq_len) < lengths.unsqueeze(1)
        mask = mask.to(v.device).transpose(1, 2)
        v = v * mask.float()
        v = v.transpose(1, 2)
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