import argparse
import os
import time
import random
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, cohen_kappa_score
from torch.optim import *
from sklearn.metrics import precision_recall_curve, auc
from models.dataset import *
from models.health_risk_baselines import *
from utils.utils import check_path, export_config, bool_flag
import csv
import os
from datetime import datetime
from models.adacare import AdaCare
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
class LSTM_for_AE(nn.Module):
    def __init__(self, name, vocab_list, d_model, dropout, generator=None):
        super(LSTM_for_AE, self).__init__()
        self.name = name
        self.dropout = nn.Dropout(dropout)
        self.visit_generator = generator
        self.diag_embedding = nn.Embedding(vocab_list[0] + 1, d_model)
        self.drug_embedding = nn.Embedding(vocab_list[1] + 1, d_model)
        self.lab_embedding = nn.Embedding(vocab_list[2] + 1, d_model)
        self.proc_embedding = nn.Embedding(vocab_list[3] + 1, d_model)
        self.diag_output_mlp = nn.Sequential(nn.Linear(int(d_model*4), int(d_model*2)), nn.ReLU(), nn.Dropout(0.1),
                                             nn.Linear(int(d_model*2), vocab_list[0]))
        # self.lstm = nn.LSTM(d_model, d_model, 1, bidirectional=False, batch_first=True, dropout=dropout)


    def hidden_state_learner(self, v, lengths):
        h, _ = self.lstm(v)
        return h
    def forward(self, diag_seq, drug_seq, lab_seq, proc_seq, lengths=None):
        batch_size, seq_len, code_len = diag_seq.shape
        diag_v = self.dropout(self.diag_embedding(diag_seq)).sum(dim=-2)
        drug_v = self.dropout(self.drug_embedding(drug_seq)).sum(dim=-2)
        lab_v = self.dropout(self.lab_embedding(lab_seq)).sum(dim=-2)
        proc_v = self.dropout(self.proc_embedding(proc_seq)).sum(dim=-2)

        # v = torch.cat([diag_v, drug_v, lab_v, proc_v], dim=-1).view(batch_size, seq_len, 4, -1).sum(dim=-2)
        v = torch.cat([diag_v, drug_v, lab_v, proc_v], dim=-1)

        # h = self.hidden_state_learner(v, lengths)

        real_diag_logits = self.diag_output_mlp(v)

        return real_diag_logits


def eval_metric(eval_set, model, diag_pad_id, num_classes):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    all_y_true = []
    all_y_pred = []

    with torch.no_grad():
        for data in eval_set:
            _, diag, drug, lab, proc, demo = [d.to(device) for d in data]
            logits = model(diag, drug, lab, proc, demo)

            logits = logits[:, :-1, :]
            diag = diag[:, 1:, :]

            target = F.one_hot(diag, num_classes+1).sum(dim=-2)
            target = target.to(torch.float32)
            target = target[:,:,:-1]

            probabilities = F.softmax(logits, dim=-1)

            all_y_true.append(target.view(-1, num_classes).cpu().numpy())
            all_y_pred.append(probabilities.view(-1, num_classes).cpu().numpy())

    all_y_true = np.concatenate(all_y_true, axis=0)
    all_y_pred = np.concatenate(all_y_pred, axis=0)

    # Compute AUROC for each class
    auroc_per_class = []
    for i in range(num_classes):
        y_true = all_y_true[:, i]
        y_pred = all_y_pred[:, i]

        # Check if only one class is present
        if len(np.unique(y_true)) > 1:
            auroc = roc_auc_score(y_true, y_pred)
            auroc_per_class.append(auroc)
        else:
            # Handle the single-class case here, e.g., by continuing or assigning a default value
            continue

    # Handle the case where auroc_per_class is empty
    if not auroc_per_class:
        return None  # or some default value

    # Average AUROC
    macro_auroc = np.mean(auroc_per_class)

    return macro_auroc
def main(seed, name, model, data, task, max_len, max_num, sav_dir, mode, short_ICD, toy):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=True, type=bool_flag, nargs='?', const=True, help='use GPU')
    parser.add_argument('--seed', default=seed, type=int, help='seed')
    parser.add_argument('-bs', '--batch_size', default=32, type=int)
    parser.add_argument('-me', '--max_epochs_before_stop', default=5, type=int)
    parser.add_argument('--d_model', default=1024, type=int, help='dimension of hidden layers')
    parser.add_argument('--h_model', default=128, type=int, help='dimension of hidden layers')
    parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate of hidden layers')
    parser.add_argument('--dropout_emb', default=0.1, type=float, help='dropout rate of embedding layers')
    parser.add_argument('--num_layers', default=1, type=int, help='number of transformer layers of EHR encoder')
    parser.add_argument('--num_heads', default=4, type=int, help='number of attention heads')
    parser.add_argument('--max_len', default=max_len, type=int, help='max visits of EHR')
    parser.add_argument('--max_num_codes', default=max_num, type=int, help='max number of ICD codes in each visit')
    parser.add_argument('--max_num_blks', default=100, type=int, help='max number of blocks in each visit')
    parser.add_argument('--blk_emb_path', default='./data/processed/block_embedding.npy',
                        help='embedding path of blocks')
    parser.add_argument('--target_disease', default=data, choices=['Heartfailure', 'COPD', 'Kidney', 'Dementia', 'Amnesia', 'mimic'])
    parser.add_argument('--target_att_heads', default=4, type=int, help='target disease attention heads number')
    parser.add_argument('--mem_size', default=15, type=int, help='memory size')
    parser.add_argument('--mem_update_size', default=15, type=int, help='memory update size')
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=0.001, type=float)
    parser.add_argument('--target_rate', default=0.3, type=float)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='max grad norm (0 to disable)')
    parser.add_argument('--warmup_steps', default=200, type=int)
    parser.add_argument('--n_epochs', default=50, type=int)
    parser.add_argument('--log_interval', default=100, type=int)
    parser.add_argument('--mode', default=mode)
    parser.add_argument('--model', default=model)
    parser.add_argument('--name', default=name)
    parser.add_argument('--save_dir', default=sav_dir)
    parser.add_argument('--lambda_gen', default=0.1, type=float)
    parser.add_argument('--lambda_ce', default=1, type=float)
    parser.add_argument('--short_ICD', default=short_ICD, type=bool_flag, nargs='?', const=True, help='use short ICD codes')
    parser.add_argument('--lambda_gp', default=10, type=float)
    parser.add_argument('--critic_iterations', default=5, type=int)
    parser.add_argument('--toy', default=toy, type=bool_flag, nargs='?', const=True, help='use toy dataset')
    parser.add_argument('--task', default=task)
    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    else:
        raise ValueError('Invalid mode')

def train(args):
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    files = os.listdir(str(args.save_dir))
    results_file_name = str(args.save_dir) + str(args.name)+ '_' + str(args.model) + '_' + str(args.target_disease) + '_' + str(args.task)+ '_' + str(args.seed) + '.csv'

    if results_file_name in files:
        print('Already trained')
        return

    if args.target_disease == 'pancreas':
        code2id = pd.read_csv('./data/pancreas/code_to_int_mapping.csv', header=None)
        demo_len = 15
        pad_id = len(code2id)
        data_path = './data/pancreas/'
    elif args.target_disease == 'mimic' and args.short_ICD:
        diag2id = pd.read_csv('./data/mimic/diagnosis_to_int_mapping_3dig.csv', header=None)
        drug2id = pd.read_csv('./data/mimic/drug_to_int_mapping_3dig.csv', header=None)
        lab2id = pd.read_csv('./data/mimic/lab_to_int_mapping_3dig.csv', header=None)
        proc2id = pd.read_csv('./data/mimic/proc_to_int_mapping_3dig.csv', header=None)
        id2diag = dict(zip(diag2id[1], diag2id[0]))
        id2drug = dict(zip(drug2id[1], drug2id[0]))
        id2lab = dict(zip(lab2id[1], lab2id[0]))
        id2proc = dict(zip(proc2id[1], proc2id[0]))
        demo_len = 76
        diag_pad_id = len(diag2id)
        drug_pad_id = len(drug2id)
        lab_pad_id = len(lab2id)
        proc_pad_id = len(proc2id)
        drug_nan_id = 146
        lab_nan_id = 206
        proc_nan_id = 24
        pad_id_list = [diag_pad_id, drug_pad_id, lab_pad_id, proc_pad_id]
        nan_id_list = [diag_pad_id, drug_nan_id, lab_nan_id, proc_nan_id]
        data_path = './data/mimic/'
    elif args.target_disease == 'mimic' and not args.short_ICD:
        diag2id = pd.read_csv('./data/mimic/diagnosis_to_int_mapping_5dig.csv', header=None)
        drug2id = pd.read_csv('./data/mimic/drug_to_int_mapping_5dig.csv', header=None)
        lab2id = pd.read_csv('./data/mimic/lab_to_int_mapping_5dig.csv', header=None)
        proc2id = pd.read_csv('./data/mimic/proc_to_int_mapping_5dig.csv', header=None)
        id2diag = dict(zip(diag2id[1], diag2id[0]))
        id2drug = dict(zip(drug2id[1], drug2id[0]))
        id2lab = dict(zip(lab2id[1], lab2id[0]))
        id2proc = dict(zip(proc2id[1], proc2id[0]))
        demo_len = 76
        diag_pad_id = len(diag2id)
        drug_pad_id = len(drug2id)
        lab_pad_id = len(lab2id)
        proc_pad_id = len(proc2id)
        drug_nan_id = 146
        lab_nan_id = 206
        proc_nan_id = 24
        pad_id_list = [diag_pad_id, drug_pad_id, lab_pad_id, proc_pad_id]
        nan_id_list = [diag_pad_id, drug_nan_id, lab_nan_id, proc_nan_id]
        data_path = './data/mimic/'
    elif args.target_disease == 'breast':
        diag2id = pd.read_csv('./data/breast/ae_to_int_mapping.csv', header=None)
        drug2id = pd.read_csv('./data/breast/drug_to_int_mapping.csv', header=None)
        lab2id = pd.read_csv('./data/breast/lab_to_int_mapping.csv', header=None)
        proc2id = pd.read_csv('./data/breast/proc_to_int_mapping.csv', header=None)
        id2diag = dict(zip(diag2id[1], diag2id[0]))
        id2drug = dict(zip(drug2id[1], drug2id[0]))
        id2lab = dict(zip(lab2id[1], lab2id[0]))
        id2proc = dict(zip(proc2id[1], proc2id[0]))
        demo_len = 10
        diag_pad_id = len(diag2id)
        drug_pad_id = len(drug2id)
        lab_pad_id = len(lab2id)
        proc_pad_id = len(proc2id)
        diag_nan_id = 51
        drug_nan_id = 101
        lab_nan_id = 10
        proc_nan_id = 4
        pad_id_list = [diag_pad_id, drug_pad_id, lab_pad_id, proc_pad_id]
        nan_id_list = [diag_nan_id, drug_nan_id, lab_nan_id, proc_nan_id]
        data_path = './data/breast/'
    else:
        raise ValueError('Invalid disease')
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

    if args.model != 'none':
        synthetic_dataset = pancreas_Gendataset_multimodal_for_baselines(
            data_path + str(args.model) + '_synthetic_3dig' + str(args.target_disease) + '.csv',
            args.max_len, args.max_num_codes, pad_id_list, nan_id_list, args.task, device)
    else:
        synthetic_dataset = None

    train_dataset = pancreas_Gendataset_multimodal(
        data_path + 'train_3dig' + str(args.target_disease) + '.csv', args.max_len,
        args.max_num_codes, pad_id_list, nan_id_list, args.task, device)
    val_dataset = pancreas_Gendataset_multimodal(
        data_path + 'val_3dig' + str(args.target_disease) + '.csv', args.max_len,
        args.max_num_codes, pad_id_list, nan_id_list, args.task, device)
    test_dataset = pancreas_Gendataset_multimodal(
        data_path + 'test_3dig' + str(args.target_disease) + '.csv', args.max_len,
        args.max_num_codes, pad_id_list, nan_id_list, args.task, device)

    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=gen_collate_fn_multimodal)
    dev_dataloader = DataLoader(val_dataset, args.batch_size, shuffle=False, collate_fn=gen_collate_fn_multimodal)
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=gen_collate_fn_multimodal)
    if args.model != 'none':
        synthetic_dataloader = DataLoader(synthetic_dataset, args.batch_size, shuffle=True, collate_fn=gen_collate_fn_multimodal)
    else:
        synthetic_dataloader = None

    model = LSTM_for_AE('LSTM_for_AE', [diag_pad_id, drug_pad_id, lab_pad_id, proc_pad_id], args.d_model, args.dropout).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    global_step, best_dev_epoch = 0, 0

    best_dev_auc, total_loss = 0.0, 0.0
    best_test_auc = 0.0
    model.train()


    for epoch in range(args.n_epochs):
        if synthetic_dataloader:
            for i, data in enumerate(synthetic_dataloader):
                _, diag, drug, lab, proc, demo = data
                logits = model(diag, drug, lab, proc, demo)
                logits = logits[:, :-1, :]
                diag = diag[:, 1:, :]

                # One-hot encode target_diag
                target = torch.nn.functional.one_hot(diag, diag_pad_id + 1).sum(dim=-2)
                target = target.to(torch.float32)
                target = target[:, :, :-1]

                # Reshape logits and target for BCEWithLogitsLoss
                logits = logits.reshape(-1, diag_pad_id)  # Flatten logits
                target = target.reshape(-1, diag_pad_id)  # Flatten target

                # Calculate loss
                loss = criterion(logits, target)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                global_step += 1
            print('Epoch: ', epoch, 'Loss: ', total_loss / global_step)
            total_loss = 0.0


        else:
            for i, data in enumerate(train_dataloader):
                _, diag, drug, lab, proc, demo = data
                logits = model(diag, drug, lab, proc, demo)
                logits = logits[:, :-1, :]
                diag = diag[:, 1:, :]

                # One-hot encode target_diag
                target = torch.nn.functional.one_hot(diag, diag_pad_id + 1).sum(dim=-2)
                target = target.to(torch.float32)
                target = target[:, :, :-1]

                # Reshape logits and target for BCEWithLogitsLoss
                logits = logits.reshape(-1, diag_pad_id)  # Flatten logits
                target = target.reshape(-1, diag_pad_id)  # Flatten target

                # Calculate loss
                loss = criterion(logits, target)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                global_step += 1
            print('Epoch: ', epoch, 'Loss: ', total_loss / global_step)
            total_loss = 0.0

    # for epoch in range(args.n_epochs):
    #     for i, data in enumerate(synthetic_dataloader):
    #         _, diag, drug, lab, proc, demo = data
    #         logits = model(diag, drug, lab, proc, demo)
    #         logits = logits[:, :-1, :]
    #         diag = diag[:, 1:, :]
    #
    #         # One-hot encode target_diag
    #         target = torch.nn.functional.one_hot(diag, diag_pad_id + 1).sum(dim=-2)
    #         target = target.to(torch.float32)
    #         target = target[:, :, :-1]
    #
    #         # Reshape logits and target for BCEWithLogitsLoss
    #         logits = logits.reshape(-1, diag_pad_id)  # Flatten logits
    #         target = target.reshape(-1, diag_pad_id)  # Flatten target
    #
    #         # Calculate loss
    #         loss = criterion(logits, target)
    #
    #         # Backward pass
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #     for i, data in enumerate(train_dataloader):
    #         _, diag, drug, lab, proc, demo = data
    #         logits = model(diag, drug, lab, proc, demo)
    #         logits = logits[:, :-1, :]
    #         diag = diag[:, 1:, :]
    #
    #         # One-hot encode target_diag
    #         target = torch.nn.functional.one_hot(diag, diag_pad_id + 1).sum(dim=-2)
    #         target = target.to(torch.float32)
    #         target = target[:, :, :-1]
    #
    #         # Reshape logits and target for BCEWithLogitsLoss
    #         logits = logits.reshape(-1, diag_pad_id)  # Flatten logits
    #         target = target.reshape(-1, diag_pad_id)  # Flatten target
    #
    #         # Calculate loss
    #         loss = criterion(logits, target)
    #
    #         # Backward pass
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         total_loss += loss.item()
    #         global_step += 1
    #     print('Epoch: ', epoch, 'Loss: ', total_loss / global_step)
    #     total_loss = 0.0


        model.eval()
        dev_au_auc = eval_metric(dev_dataloader, model, None, diag_pad_id)
        test_au_auc = eval_metric(test_dataloader, model, None, diag_pad_id)

        print('-' * 71)
        print('Epoch: ', epoch)
        print('Dev AUROC: ', dev_au_auc)
        print('Test AUROC: ', test_au_auc)
        print('-' * 71)
        if dev_au_auc > best_dev_auc:
            best_dev_auc = dev_au_auc
            best_test_auc = test_au_auc
            print('Find new best model at Epoch: ', epoch)
        model.train()
    print('Best Epoch: ', best_dev_epoch)
    print('Best Dev AUROC: ', best_dev_auc)
    print('Best Test AUROC: ', best_test_auc)
    #write results
    with open(results_file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Best Epoch', 'Best Dev AUROC', 'Best Test AUROC'])
        writer.writerow([best_dev_epoch, best_dev_auc, best_test_auc])

if __name__ == '__main__':
    # main(10, 'void', 'none', 'breast', '', 10, 10, './saved_adverseEvents/', 'train', True, False)
    # main(10, 'void', 'EVA', 'breast', '', 10, 10, './saved_adverseEvents/', 'train', True, False)
    # main(10, 'void', 'LSTM-TabDDPM', 'breast', '', 10, 10, './saved_adverseEvents/', 'train', True, True)
    # main(10, 'void', 'LSTM-MLP', 'breast', '', 10, 10, './saved_adverseEvents/', 'train', True, False)
    # main(10, 'void', 'LSTM-medGAN', 'breast', '', 10, 10, './saved_adverseEvents/', 'train', True, True)
    # main(10, 'void', 'synTEG', 'breast', '', 10, 10, './saved_adverseEvents/', 'train', True, False)
    # main(10, 'void', 'TWIN', 'breast', '', 10, 10, './saved_adverseEvents/', 'train', True, False)
    # main(10, 'void', 'MedDiffGa', 'breast', '', 10, 10, './saved_adverseEvents/', 'train', True, False)
    # main(10, 'void', 'promptEHR', 'breast', '', 10, 10, './saved_adverseEvents/', 'train', True, False)

    # main(10, 'void', 'HALO', 'breast', '', 10, 10, './saved_adverseEvents/', 'train', True, False)
    main(10, 'void', 'LSTM-Meddiff', 'breast', '', 10, 10, './saved_adverseEvents/', 'train', True, False)
    # main(10, 'void', 'LSTM-ScoEHR', 'breast', '', 10, 10, './saved_adverseEvents/', 'train', True, False)














