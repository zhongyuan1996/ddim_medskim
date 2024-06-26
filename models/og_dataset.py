import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def padMatrix(input_data, max_num_pervisit, maxlen, pad_id):
    pad_seq = [pad_id] * max_num_pervisit
    output = []
    lengths = []
    for seq in input_data:
        record_ids = []
        for visit in seq:
            visit_ids = visit[0: max_num_pervisit]
            for i in range(0, (max_num_pervisit - len(visit_ids))):
                visit_ids.append(pad_id)
            record_ids.append(visit_ids)
        record_ids = record_ids[-maxlen:]
        lengths.append(len(record_ids))
        for j in range(0, (maxlen - len(record_ids))):
            record_ids.append(pad_seq)
        output.append(record_ids)
    masks = []
    for l in lengths:
        mask = np.tril(np.ones((maxlen, maxlen)))
        # mask[:l, :l] = np.tril(np.ones((l, l)))
        masks.append(mask)
    return output, masks, lengths


def padMatrix2(input_data, max_num_pervisit, maxlen, pad_id):
    pad_seq = [pad_id] * max_num_pervisit
    output = []
    masks = []
    for seq in input_data:
        record_ids = []
        mask = []
        for visit in seq:
            visit_ids = visit[0: max_num_pervisit]
            mask_v = [1] * len(visit_ids)
            for i in range(0, (max_num_pervisit - len(visit_ids))):
                visit_ids.append(pad_id)
                mask_v.append(0)
            record_ids.append(visit_ids)
            mask.append(mask_v)
        record_ids = record_ids[-maxlen:]
        mask = mask[-maxlen:]
        for j in range(0, (maxlen - len(record_ids))):
            record_ids.append(pad_seq)
            mask.append([0] * max_num_pervisit)
        output.append(record_ids)
        masks.append(mask)
    return output, masks


def padTime(time_step, maxlen, pad_id):
    for k in range(len(time_step)):
        time_step[k] = time_step[k][-maxlen:]
        while len(time_step[k]) < maxlen:
            time_step[k].append(pad_id)
    return time_step

def padTime3(time_step, maxlen, pad_id):

    return time_step

def codeMask(input_data, max_num_pervisit, maxlen):
    batch_mask = np.zeros((len(input_data), maxlen, max_num_pervisit), dtype=np.float32) + 1e+20
    output = []
    for seq in input_data:
        record_ids = []
        for visit in seq:
            visit_ids = visit[0: max_num_pervisit]
            record_ids.append(visit_ids)
        record_ids = record_ids[-maxlen:]
        output.append(record_ids)

    for bid, seq in enumerate(output):
        for pid, subseq in enumerate(seq):
            for tid, code in enumerate(subseq):
                batch_mask[bid, pid, tid] = 0
    return batch_mask

class MyDataset(Dataset):
    def __init__(self, dir_ehr, dir_txt, max_len, max_numcode_pervisit, max_numblk_pervisit, ehr_pad_id, txt_pad_id,
                 device):
        ehr, self.labels, time_step = pickle.load(open(dir_ehr, 'rb'))
        txt = pickle.load(open(dir_txt, 'rb'))
        self.ehr, self.mask_ehr, self.lengths = padMatrix(ehr, max_numcode_pervisit, max_len, ehr_pad_id)
        # time_step = time_step_to_deltatime(time_step)
        self.txt, self.mask_txt = padMatrix2(txt, max_numblk_pervisit, max_len, txt_pad_id)
        self.time_step = padTime(time_step, max_len, 100000)
        self.code_mask = codeMask(ehr, max_numcode_pervisit, max_len)
        self.device = device

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        assert torch.LongTensor(self.mask_txt[idx]).size() == torch.LongTensor(self.txt[idx]).size()

        return torch.tensor(self.labels[idx], dtype=torch.long).to(self.device), torch.LongTensor(self.ehr[idx]).to(
            self.device), \
               torch.LongTensor(self.mask_ehr[idx]).to(self.device), torch.LongTensor(self.txt[idx]).to(self.device), \
               torch.LongTensor(self.mask_txt[idx]).to(self.device), torch.tensor(self.lengths[idx],
                                                                                  dtype=torch.long).to(self.device), \
               torch.Tensor(self.time_step[idx]).to(self.device), torch.FloatTensor(self.code_mask[idx]).to(self.device)

def time_step_to_deltatime(time_step):
    time_gaps = [[0] + [t[i] - t[i+1] for i in range(len(t)-1)] for t in time_step]

    return time_gaps

class MyDataset2(Dataset):
    def __init__(self, dir_ehr, max_len, max_numcode_pervisit, max_numblk_pervisit, ehr_pad_id,
                 device):
        ehr, self.labels, time_step = pickle.load(open(dir_ehr, 'rb'))
        self.ehr, self.mask_ehr, self.lengths = padMatrix(ehr, max_numcode_pervisit, max_len, ehr_pad_id)
        # time_step = time_step_to_deltatime(time_step)
        self.time_step = padTime(time_step, max_len, 10000)
        self.code_mask = codeMask(ehr, max_numcode_pervisit, max_len)
        self.device = device

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        _ = None
        return torch.tensor(self.labels[idx], dtype=torch.long).to(self.device), torch.LongTensor(self.ehr[idx]).to(
            self.device), \
               torch.LongTensor(self.mask_ehr[idx]).to(self.device), _, \
               _, torch.tensor(self.lengths[idx], dtype=torch.long).to(self.device), \
               torch.Tensor(self.time_step[idx]).to(self.device), torch.FloatTensor(self.code_mask[idx]).to(self.device)

class BalancedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, ratio):
        self.dataset = dataset
        self.batch_size = batch_size
        self.ratio = ratio

        # Split indices by label
        self.positive_indices = []
        self.negative_indices = []
        for idx in range(len(self.dataset)):
            if self.dataset[idx][0].item() == 1:  # Assuming the label is the first element of the dataset
                self.positive_indices.append(idx)
            else:
                self.negative_indices.append(idx)

        # Calculate the number of positive and negative samples per batch
        self.num_positives_per_batch = int(batch_size * ratio)
        self.num_negatives_per_batch = batch_size - self.num_positives_per_batch

    def __iter__(self):
        np.random.shuffle(self.positive_indices)
        np.random.shuffle(self.negative_indices)

        num_batches = min(len(self.positive_indices) // self.num_positives_per_batch,
                          len(self.negative_indices) // self.num_negatives_per_batch)

        for batch_idx in range(num_batches):
            # Select positive indices
            start_idx = batch_idx * self.num_positives_per_batch
            end_idx = start_idx + self.num_positives_per_batch
            positive_indices_in_batch = self.positive_indices[start_idx:end_idx]

            # Select negative indices
            start_idx = batch_idx * self.num_negatives_per_batch
            end_idx = start_idx + self.num_negatives_per_batch
            negative_indices_in_batch = self.negative_indices[start_idx:end_idx]

            # Yield combined indices
            yield positive_indices_in_batch + negative_indices_in_batch

    def __len__(self):
        return min(len(self.positive_indices) // self.num_positives_per_batch,
                   len(self.negative_indices) // self.num_negatives_per_batch)
class MyDataset3(Dataset):
    def __init__(self, dir_ehr, max_len, max_numcode_pervisit, max_numblk_pervisit, ehr_pad_id,
                 device):
        data = np.load(dir_ehr, allow_pickle=True)
        self.ehr, self.labels, self.time_step = data['x'], data['y'], data['timeseq']
        # self.labels = [[0, 1] if label == 1 else [1, 0] for label in labels]
        # self.ehr, self.mask_ehr, self.lengths = padMatrix(ehr, max_numcode_pervisit, max_len, ehr_pad_id)
        # self.time_step = padTime3(time_step, max_len, 100000)
        # self.code_mask = codeMask(ehr, max_numcode_pervisit, max_len)
        self.device = device

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        _ = None
        # return torch.tensor(self.labels[idx], dtype=torch.long).to(self.device), torch.FloatTensor(self.ehr[idx]).to(
        #     self.device), \
        #        torch.LongTensor(self.mask_ehr[idx]).to(self.device), _, \
        #        _, torch.tensor(self.lengths[idx], dtype=torch.long).to(self.device), \
        #        torch.Tensor(self.time_step[idx]).to(self.device), torch.FloatTensor(self.code_mask[idx]).to(self.device)

        return torch.tensor(self.labels[idx], dtype=torch.long).to(self.device), torch.FloatTensor(self.ehr[idx]).to(
            self.device), \
               _, _, \
               _, _, \
               torch.Tensor(self.time_step[idx]).to(self.device), _
def collate_fn(batch):
    label, ehr, mask, txt, mask_txt, length, time_step, code_mask = [], [], [], [], [], [], [], []
    for data in batch:
        label.append(data[0])
        ehr.append(data[1])
        mask.append(data[2])
        # txt.append(data[3])
        # mask_txt.append(data[4])
        length.append(data[5])
        time_step.append(data[6])
        code_mask.append(data[7])
    _=None
    # return torch.stack(label, 0), torch.stack(ehr, 0), torch.stack(mask, 0), torch.stack(txt, 0), \
    #        torch.stack(mask_txt, 0), torch.stack(length, 0), torch.stack(time_step, 0), torch.stack(code_mask, 0)
    return torch.stack(label, 0), torch.stack(ehr, 0), torch.stack(mask, 0), _, \
           _, torch.stack(length, 0), torch.stack(time_step, 0), torch.stack(code_mask, 0)

def collate_fn3(batch):
    label, ehr, mask, txt, mask_txt, length, time_step, code_mask = [], [], [], [], [], [], [], []
    for data in batch:
        label.append(data[0])
        ehr.append(data[1])
        # mask.append(data[2])
        # txt.append(data[3])
        # mask_txt.append(data[4])
        # length.append(data[5])
        time_step.append(data[6])
        # code_mask.append(data[7])
    _=None
    # return torch.stack(label, 0), torch.stack(ehr, 0), torch.stack(mask, 0), torch.stack(txt, 0), \
    #        torch.stack(mask_txt, 0), torch.stack(length, 0), torch.stack(time_step, 0), torch.stack(code_mask, 0)
    return torch.stack(label, 0), torch.stack(ehr, 0), _, _, \
           _, _, torch.stack(time_step, 0), _
class MyDataset_EEG(Dataset):
    def __init__(self, dir_ehr, dir_txt, max_len, max_numcode_pervisit, max_numblk_pervisit, ehr_pad_id, txt_pad_id,
                 device):
        ehr, self.labels = pickle.load(open(dir_ehr, 'rb'))

        self.ehr, self.mask_ehr, self.lengths = padMatrix(ehr, max_numcode_pervisit, max_len, ehr_pad_id)
        self.time_step = torch.ones(len(self.ehr), len(self.ehr[0]))
        self.code_mask = codeMask(ehr, max_numcode_pervisit, max_len)
        self.device = device

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        _ = None
        return torch.tensor(self.labels[idx], dtype=torch.long).to(self.device), torch.tensor(self.ehr[idx], dtype=torch.float32).to(
            self.device), \
               torch.LongTensor(self.mask_ehr[idx]).to(self.device), _, \
               _, torch.tensor(self.lengths[idx], dtype=torch.long).to(self.device), \
               torch.Tensor(self.time_step[idx]).to(self.device), torch.FloatTensor(self.code_mask[idx]).to(self.device)

