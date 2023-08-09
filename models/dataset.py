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
    lengths = []
    for seq in input_data:
        record_ids = []
        for visit in seq:
            visit_ids = [visit] if visit != pad_id else []  # Changed this line
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


def padTime(time_step, maxlen, pad_id):
    for k in range(len(time_step)):
        time_step[k] = time_step[k][-maxlen:]
        while len(time_step[k]) < maxlen:
            time_step[k].append(pad_id)
    return time_step
def padTime3(time_step, maxlen, pad_id):

    return time_step
def codeMask(input_data, max_num_pervisit, maxlen):
    batch_mask = np.zeros((len(input_data), maxlen, max_num_pervisit), dtype=np.float32) + 1
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

def codeMask2(input_data, max_num_pervisit, maxlen):
    batch_mask = np.zeros((len(input_data), maxlen, max_num_pervisit), dtype=np.float32) + 1
    output = []
    for seq in input_data:
        record_ids = []
        for visit in seq:
            visit_ids = [visit]  # Changed this line
            record_ids.append(visit_ids)
        record_ids = record_ids[-maxlen:]
        output.append(record_ids)

    for bid, seq in enumerate(output):
        for pid, subseq in enumerate(seq):
            for tid, code in enumerate(subseq):
                batch_mask[bid, pid, tid] = 0
    return batch_mask

class MyDataset(Dataset):
    def __init__(self, dir_ehr, max_len, max_numcode_pervisit, ehr_pad_id,
                 device):
        ehr, labels, time_step = pickle.load(open(dir_ehr, 'rb'))
        self.labels = [[0,1] if label == 1 else [1,0] for label in labels]

        self.ehr, _, _ = padMatrix(ehr, max_numcode_pervisit, max_len, ehr_pad_id)
        self.time_step = padTime(time_step, max_len, 100000)
        self.code_mask = codeMask(ehr, max_numcode_pervisit, max_len)
        self.device = device

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return torch.tensor(self.ehr[idx], dtype=torch.long).to(self.device),\
               torch.tensor(self.time_step[idx], dtype=torch.long).to(self.device),\
               torch.tensor(self.labels[idx], dtype=torch.float).to(self.device),\
                  torch.tensor(self.code_mask[idx], dtype=torch.float).to(self.device)


class MyDataset_mapping(Dataset):
    def __init__(self, dir_ehr, max_len, max_numcode_pervisit, ehr_pad_id,
                 device):
        ehr, labels, time_step = pickle.load(open(dir_ehr, 'rb'))

        reshaped_ehr = []
        reshaped_time_step = []
        expanded_labels = []
        for b, batch in enumerate(ehr):
            for v, visit in enumerate(batch):
                for c, code in enumerate(visit):
                    reshaped_ehr.append([code])
                    reshaped_time_step.append([time_step[b][v]])
                    expanded_labels.append(labels[b])

        self.labels = [[0, 1] if label == 1 else [1, 0] for label in expanded_labels]
        self.ehr, _, _ = padMatrix2(reshaped_ehr, max_numcode_pervisit, max_len, ehr_pad_id)
        self.time_step = padTime(reshaped_time_step, max_len, 100000)
        self.code_mask = codeMask2(reshaped_ehr, max_numcode_pervisit, max_len)
        self.device = device

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return torch.tensor(self.ehr[idx], dtype=torch.long).to(self.device), \
            torch.tensor(self.time_step[idx], dtype=torch.long).to(self.device), \
            torch.tensor(self.labels[idx], dtype=torch.float).to(self.device), \
            torch.tensor(self.code_mask[idx], dtype=torch.float).to(self.device)

class MyDataset3(Dataset):
    def __init__(self, dir_ehr, max_len, max_numcode_pervisit, max_numblk_pervisit, ehr_pad_id,
                 device):
        data = np.load(dir_ehr, allow_pickle=True)
        self.ehr, labels, self.time_step = data['x'], data['y'], data['timeseq']
        self.labels = [[0,1] if label == 1 else [1,0] for label in labels]
        # self.ehr, _, _ = padMatrix(ehr, max_numcode_pervisit, max_len, ehr_pad_id)
        # self.time_step = padTime3(time_step, max_len, 100000)
        self.device = device

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        _ = None
        return torch.tensor(self.ehr[idx], dtype=torch.float32).to(self.device),\
               torch.tensor(self.time_step[idx], dtype=torch.long).to(self.device),\
               torch.tensor(self.labels[idx], dtype=torch.float).to(self.device)
class MyDataset_with_single_label(Dataset):
    def __init__(self, dir_ehr, max_len, max_numcode_pervisit, ehr_pad_id,
                 device):
        ehr, self.labels, time_step = pickle.load(open(dir_ehr, 'rb'))
        self.ehr, _, _ = padMatrix(ehr, max_numcode_pervisit, max_len, ehr_pad_id)
        self.time_step = padTime(time_step, max_len, 100000)
        self.device = device

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return torch.tensor(self.ehr[idx], dtype=torch.long).to(self.device),\
               torch.tensor(self.time_step[idx], dtype=torch.long).to(self.device),\
               torch.tensor(self.labels[idx], dtype=torch.long).to(self.device)

# class _MyDataset(Dataset):
#     def __init__(self, dir_ehr, dir_txt, max_len, max_numcode_pervisit, max_numblk_pervisit, ehr_pad_id, txt_pad_id,
#                  device):
#         ehr, self.labels, time_step = pickle.load(open(dir_ehr, 'rb'))
#         txt = pickle.load(open(dir_txt, 'rb'))
#         self.ehr, self.mask_ehr, self.lengths = padMatrix(ehr, max_numcode_pervisit, max_len, ehr_pad_id)
#         self.txt, self.mask_txt = padMatrix2(txt, max_numblk_pervisit, max_len, txt_pad_id)
#         self.time_step = padTime(time_step, max_len, 100000)
#         self.code_mask = codeMask(ehr, max_numcode_pervisit, max_len)
#         self.device = device
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         assert torch.LongTensor(self.mask_txt[idx]).size() == torch.LongTensor(self.txt[idx]).size()
#
#         return torch.tensor(self.labels[idx], dtype=torch.long).to(self.device), torch.LongTensor(self.ehr[idx]).to(
#             self.device), \
#                torch.LongTensor(self.mask_ehr[idx]).to(self.device), torch.LongTensor(self.txt[idx]).to(self.device), \
#                torch.LongTensor(self.mask_txt[idx]).to(self.device), torch.tensor(self.lengths[idx],
#                                                                                   dtype=torch.long).to(self.device), \
#                torch.Tensor(self.time_step[idx]).to(self.device), torch.FloatTensor(self.code_mask[idx]).to(self.device)


def collate_fn(batch):
    ehr, time_step, label, code_mask = [], [], [], []
    # label, ehr, mask, txt, mask_txt, length, time_step, code_mask = [], [], [], [], [], [], [], []
    for data in batch:
        ehr.append(data[0])
        time_step.append(data[1])
        label.append(data[2])
        code_mask.append(data[3])

    return torch.stack(ehr, 0), torch.stack(time_step, 0), torch.stack(label, 0), torch.stack(code_mask, 0)
        #
        # label.append(data[0])
        # ehr.append(data[1])
        # mask.append(data[2])
        # txt.append(data[3])
        # mask_txt.append(data[4])
        # length.append(data[5])
        # time_step.append(data[6])
        # code_mask.append(data[7])
    # return torch.stack(label, 0), torch.stack(ehr, 0), torch.stack(mask, 0), torch.stack(txt, 0), \
    #        torch.stack(mask_txt, 0), torch.stack(length, 0), torch.stack(time_step, 0), torch.stack(code_mask, 0)

class w2vecDataset(Dataset):
    def __init__(self, dir_ehr,device):
        self.ehr, _, _ = pickle.load(open(dir_ehr, 'rb'))
        self.device = device
    def __len__(self):
        return len(self.ehr)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return torch.tensor(self.ehr[idx], dtype=torch.long).to(self.device)

def collate_fn_w2vec(batch):
    ehr = []
    for data in batch:
        ehr.append(data)
    return torch.stack(ehr, 0)


class ehrGANDataset(Dataset):
    def __init__(self, dir_ehr, max_len, max_numcode_pervisit, ehr_pad_id, w2v, device):
        ehr, labels, time_step = pickle.load(open(dir_ehr, 'rb'))
        self.labels = [[0,1] if label == 1 else [1,0] for label in labels]

        # aggregate ehr and time_step into 90-day windows
        ehr, time_step = self.segment_time_windows(ehr, time_step)
        self.ehr, _, _ = padMatrix(ehr, max_numcode_pervisit, max_len, ehr_pad_id)
        self.time_step = padTime(time_step, max_len, 100000)
        self.device = device
        self.w2v = w2v
        self.pad_id = ehr_pad_id

    def ehr_to_embedding(self, ehr):
        embedding_sequence = []
        mask = []
        for visit in ehr:  # Loop over each visit in the EHR data
            visit_embedding = []
            visit_mask = []
            for code in visit:  # Loop over each code in the visit
                if code == self.pad_id:  # If the code is a padding token, return a zero vector.
                    visit_embedding.append(np.zeros(self.w2v.vector_size))
                    visit_mask.append(np.zeros(self.w2v.vector_size))  # Add a zero vector to the mask if it is a padding token.
                elif code in self.w2v.wv:  # If the code is in the Word2Vec model, add its vector.
                    visit_embedding.append(self.w2v.wv[code])
                    visit_mask.append(np.ones(self.w2v.vector_size))  # Add a one vector to the mask if it is a real code.
            embedding_sequence.append(visit_embedding)
            mask.append(visit_mask)
        return np.array(embedding_sequence), np.array(mask)

    def segment_time_windows(self, ehr, time_step):
        ehr_segmented, time_step_segmented = [], []
        for patient_idx in range(len(ehr)):
            patient_ehr = ehr[patient_idx]
            patient_time_step = time_step[patient_idx]
            patient_ehr_segmented, patient_time_step_segmented = [], []
            aggregated_codes, end_time = [], patient_time_step[0]
            for visit_idx in range(len(patient_ehr)):
                visit = patient_ehr[visit_idx]
                if end_time - patient_time_step[visit_idx] < 90:
                    aggregated_codes.extend(visit)
                else:
                    # Aggregate and append codes of the current window
                    patient_ehr_segmented.append(list(set(aggregated_codes)))
                    patient_time_step_segmented.append(end_time)
                    # Start a new window
                    aggregated_codes = visit
                    end_time = patient_time_step[visit_idx]
            # Don't forget to append the last window
            patient_ehr_segmented.append(list(set(aggregated_codes)))
            patient_time_step_segmented.append(end_time)

            # Append the segmented ehr and time_step of each patient
            ehr_segmented.append(patient_ehr_segmented)
            time_step_segmented.append(patient_time_step_segmented)

        return ehr_segmented, time_step_segmented

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        temp_ehr, temp_code_mask = self.ehr_to_embedding(self.ehr[idx])

        ehr = torch.tensor(temp_ehr, dtype=torch.float).to(self.device)
        time_step = torch.tensor(self.time_step[idx], dtype=torch.long).to(self.device)
        label = torch.tensor(self.labels[idx], dtype=torch.float).to(self.device)
        code_mask = torch.tensor(temp_code_mask, dtype=torch.float).to(self.device)

        return ehr, time_step, label, code_mask
