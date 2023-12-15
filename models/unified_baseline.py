import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.baseline import Attention


class baseline_wrapper(nn.Module):
    def __init__(self, name, vocab_list, d_model, dropout, hidden_state_learner, visit_generator):
        super(baseline_wrapper, self).__init__()
        self.name = name
        self.dropout = nn.Dropout(dropout)
        self.hidden_state_learner = hidden_state_learner
        self.visit_generator = visit_generator
        self.diag_embedding = nn.Embedding(vocab_list[0]+1, d_model)
        self.drug_embedding = nn.Embedding(vocab_list[1]+1, d_model)
        self.lab_embedding = nn.Embedding(vocab_list[2]+1, d_model)
        self.proc_embedding = nn.Embedding(vocab_list[3]+1, d_model)
        self.diag_output_mlp = nn.Sequential(nn.Linear(d_model, vocab_list[0]))
        self.drug_output_mlp = nn.Sequential(nn.Linear(d_model, vocab_list[1]))
        self.lab_output_mlp = nn.Sequential(nn.Linear(d_model, vocab_list[2]))
        self.proc_output_mlp = nn.Sequential(nn.Linear(d_model, vocab_list[3]))

    # def forward(self, diag_seq, drug_seq, lab_seq, proc_seq, time_step, visit_timegap, diag_timegaps, drug_timegaps,
    #             lab_timegaps, proc_timegaps, \
    #             diag_mask, drug_mask, lab_mask, proc_mask, diag_length, drug_length, lab_length, proc_length, demo):
    def forward(self, diag_seq, drug_seq, lab_seq, proc_seq):

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
        h = self.hidden_state_learner(v)

        #assuming the generator will take in the hidden state and output a synthetic one.

        if self.name == 'GAN':
            v_gen, real_discrimination, gen_discrimination = self.visit_generator(h)
        elif self.name == 'DIFF':
            v_gen, added_noise, learned_noise = self.visit_generator(h)

        #both real v and gen_v predict the next visit

        real_diag_logits = self.diag_output_mlp(h)
        real_drug_logits = self.drug_output_mlp(h)
        real_lab_logits = self.lab_output_mlp(h)
        real_proc_logits = self.proc_output_mlp(h)

        gen_diag_logits = self.diag_output_mlp(v_gen)
        gen_drug_logits = self.drug_output_mlp(v_gen)
        gen_lab_logits = self.lab_output_mlp(v_gen)
        gen_proc_logits = self.proc_output_mlp(v_gen)

        if self.name == 'GAN':
            return real_diag_logits, real_drug_logits, real_lab_logits, real_proc_logits, gen_diag_logits, gen_drug_logits, gen_lab_logits, gen_proc_logits, h, v_gen, real_discrimination, gen_discrimination
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
        mask = torch.arange(seq_len).expand(len(lengths), seq_len) < lengths.unsqueeze(1)
        mask = mask.to(v.device)
        mask = mask.unsqueeze(1).expand(-1, seq_len, -1)
        v, _ = self.attention(v, v, v, attn_mask=mask)
        return v