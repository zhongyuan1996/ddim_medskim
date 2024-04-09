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
class LSTM_for_SE(nn.Module):
    def __init__(self, name, vocab_list, d_model, dropout, generator=None):
        super(LSTM_for_SE, self).__init__()
        self.name = name
        self.dropout = nn.Dropout(dropout)
        self.visit_generator = generator
        self.diag_embedding = nn.Embedding(vocab_list[0] + 1, d_model)
        self.drug_embedding = nn.Embedding(vocab_list[1] + 1, d_model)
        self.lab_embedding = nn.Embedding(vocab_list[2] + 1, d_model)
        self.proc_embedding = nn.Embedding(vocab_list[3] + 1, d_model)
        self.output_mlp = nn.Sequential(nn.Linear(int(d_model), int(d_model/2)), nn.ReLU(), nn.Dropout(0.1),
                                             nn.Linear(int(d_model/2), 2))
        self.lstm = nn.LSTM(int(d_model*4), d_model, 1, bidirectional=False, batch_first=True, dropout=dropout)
        self.pooler = MaxPoolLayer()


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

        pooled_h = self.pooler(h, lengths)

        out = self.output_mlp(pooled_h)

        return out


def eval_metric(eval_set, model, encoder):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        y_true = np.array([])
        y_pred = np.array([])
        y_score = np.array([])
        for i, data in enumerate(eval_set):
            labels, diag, drug, lab, proc, demo = data
            logits = model(diag, drug, lab, proc)
            scores = torch.softmax(logits, dim=-1)
            scores = scores.data.cpu().numpy()
            labels = labels.data.cpu().numpy()
            score = scores[:, 1]
            pred = scores.argmax(1)
            y_true = np.concatenate((y_true, labels))
            y_pred = np.concatenate((y_pred, pred))
            y_score = np.concatenate((y_score, score))
        accuary = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_score)
        lr_precision, lr_recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(lr_recall, lr_precision)
        kappa = cohen_kappa_score(y_true, y_pred)

    return accuary, precision, recall, f1, roc_auc, pr_auc, kappa
def main(seed, name, model, data, task, max_len, max_num, sav_dir, mode, short_ICD, toy):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=True, type=bool_flag, nargs='?', const=True, help='use GPU')
    parser.add_argument('--seed', default=seed, type=int, help='seed')
    parser.add_argument('-bs', '--batch_size', default=32, type=int)
    parser.add_argument('-me', '--max_epochs_before_stop', default=5, type=int)
    parser.add_argument('--d_model', default=16, type=int, help='dimension of hidden layers')
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

    model = LSTM_for_SE('LSTM_for_SE', [diag_pad_id, drug_pad_id, lab_pad_id, proc_pad_id], args.d_model, args.dropout).to(device)
    optim = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_func = nn.CrossEntropyLoss()

    global_step, best_dev_epoch = 0, 0
    best_dev_auc, total_loss = 0.0, 0.0
    best_test_acc, best_t_precision, best_t_recall, best_t_f1, best_t_roc_auc, best_t_pr_auc, best_t_kappa = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    model.train()


    for epoch_id in range(args.n_epochs):
        print('epoch: {:5} '.format(epoch_id))
        model.train()
        start_time = time.time()
        if args.model == 'none':
            for i, data in enumerate(tqdm(train_dataloader, desc="Training with real data")):
                labels, diag, drug, lab, proc, demo = data
                optim.zero_grad()
                outputs = model(diag, drug, lab, proc)
                loss = loss_func(outputs, labels)
                loss.backward()
                total_loss += (loss.item() / labels.size(0)) * args.batch_size
                if args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optim.step()
                if (global_step + 1) % args.log_interval == 0:
                    total_loss /= args.log_interval
                    ms_per_batch = 1000 * (time.time() - start_time) / args.log_interval
                    print('| step {:5} | loss {:7.4f} | ms/batch {:7.2f} |'.format(global_step,
                                                                                   total_loss,
                                                                                   ms_per_batch))
                    total_loss = 0.0
                    start_time = time.time()
                global_step += 1
        elif args.name == 'fakeOnly':
            for i, synthetic_data in enumerate(tqdm(synthetic_dataloader, desc="Training with synthetic data")):
                synthetic_labels, synthetic_diag, synthetic_drug, synthetic_lab, synthetic_proc, synthetic_demo = synthetic_data
                optim.zero_grad()
                synthetic_outputs = model(synthetic_diag, synthetic_drug, synthetic_lab, synthetic_proc)
                loss = loss_func(synthetic_outputs, synthetic_labels)
                loss.backward()
                total_loss += (loss.item() / synthetic_labels.size(0)) * args.batch_size
                if args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optim.step()
                if (global_step + 1) % args.log_interval == 0:
                    total_loss /= args.log_interval
                    ms_per_batch = 1000 * (time.time() - start_time) / args.log_interval
                    print('| step {:5} | loss {:7.4f} | ms/batch {:7.2f} |'.format(global_step,
                                                                                   total_loss,
                                                                                   ms_per_batch))
                    total_loss = 0.0
                    start_time = time.time()
                global_step += 1

        elif args.model != 'none' and args.name != 'fakeOnly':
            for i, (data, synthetic_data) in enumerate(tqdm(zip(train_dataloader, synthetic_dataloader), desc="Training with real and synthetic data")):
                labels, diag, drug, lab, proc, demo = data
                synthetic_labels, synthetic_diag, synthetic_drug, synthetic_lab, synthetic_proc, synthetic_demo = synthetic_data
                optim.zero_grad()
                outputs = model(diag, drug, lab, proc)
                synthetic_outputs = model(synthetic_diag, synthetic_drug, synthetic_lab, synthetic_proc)
                loss = loss_func(outputs, labels) + args.lambda_gen * loss_func(synthetic_outputs, synthetic_labels)
                loss.backward()
                total_loss += (loss.item() / labels.size(0)) * args.batch_size
                if args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optim.step()
                if (global_step + 1) % args.log_interval == 0:
                    total_loss /= args.log_interval
                    ms_per_batch = 1000 * (time.time() - start_time) / args.log_interval
                    print('| step {:5} | loss {:7.4f} | ms/batch {:7.2f} |'.format(global_step,
                                                                                   total_loss,
                                                                                   ms_per_batch))
                    total_loss = 0.0
                    start_time = time.time()
                global_step += 1



        model.eval()
        train_acc, tr_precision, tr_recall, tr_f1, tr_roc_auc, tr_pr_auc, tr_kappa = eval_metric(train_dataloader,
                                                                                                 model,
                                                                                                 args.name)
        dev_acc, d_precision, d_recall, d_f1, d_roc_auc, d_pr_auc, d_kappa = eval_metric(dev_dataloader, model,
                                                                                         args.name)
        test_acc, t_precision, t_recall, t_f1, t_roc_auc, t_pr_auc, t_kappa = eval_metric(test_dataloader, model,
                                                                                          args.name)
        print('-' * 71)
        print('| step {:5} | train_acc {:7.4f} | dev_acc {:7.4f} | test_acc {:7.4f} '.format(global_step,
                                                                                             train_acc,
                                                                                             dev_acc,
                                                                                             test_acc))
        print(
            '| step {:5} | train_precision {:7.4f} | dev_precision {:7.4f} | test_precision {:7.4f} '.format(
                global_step,
                tr_precision,
                d_precision,
                t_precision))
        print('| step {:5} | train_recall {:7.4f} | dev_recall {:7.4f} | test_recall {:7.4f} '.format(
            global_step,
            tr_recall,
            d_recall,
            t_recall))
        print('| step {:5} | train_f1 {:7.4f} | dev_f1 {:7.4f} | test_f1 {:7.4f} '.format(global_step,
                                                                                          tr_f1,
                                                                                          d_f1,
                                                                                          t_f1))
        print('| step {:5} | train_auc {:7.4f} | dev_auc {:7.4f} | test_auc {:7.4f} '.format(global_step,
                                                                                             tr_roc_auc,
                                                                                             d_roc_auc,
                                                                                             t_roc_auc))
        print('| step {:5} | train_pr {:7.4f} | dev_pr {:7.4f} | test_pr {:7.4f} '.format(global_step,
                                                                                          tr_pr_auc,
                                                                                          d_pr_auc,
                                                                                          t_pr_auc))
        print('| step {:5} | train_kappa {:7.4f} | dev_kappa {:7.4f} | test_kappa {:7.4f} '.format(global_step,
                                                                                                   tr_kappa,
                                                                                                   d_kappa,
                                                                                                   t_kappa))
        print('-' * 71)

        if d_roc_auc >= best_dev_auc:
            best_dev_auc = d_roc_auc
            best_dev_epoch = epoch_id
            best_test_acc = test_acc
            best_t_precision = t_precision
            best_t_recall = t_recall
            best_t_f1 = t_f1
            best_t_roc_auc = t_roc_auc
            best_t_pr_auc = t_pr_auc
            best_t_kappa = t_kappa

            print('Getting better performance on dev set at epoch {}'.format(epoch_id))
        if epoch_id - best_dev_epoch >= args.max_epochs_before_stop:
            break
    print()
    print('Training ends at epoch {}'.format(epoch_id))
    print('Best dev auc: {:.4f}'.format(best_dev_auc))
    print('Test auc: {:.4f}'.format(best_t_roc_auc))
    print('Test precision: {:.4f}'.format(best_t_precision))
    print('Test recall: {:.4f}'.format(best_t_recall))
    print('Test f1: {:.4f}'.format(best_t_f1))
    print('Test kappa: {:.4f}'.format(best_t_kappa))
    print('Test pr_auc: {:.4f}'.format(best_t_pr_auc))
    print()
    results_file = open(results_file_name, 'w')
    csvwriter = csv.writer(results_file)
    csvwriter.writerow([best_test_acc, best_t_precision, best_t_recall, best_t_f1, best_t_roc_auc, best_t_pr_auc, best_t_kappa])

if __name__ == '__main__':
    # main(10, 'void', 'none', 'breast', '', 10, 10, './saved_severeOutcomePred/', 'train', True, False)
    # main(10, 'void', 'EVA', 'breast', '', 10, 10, './saved_severeOutcomePred/', 'train', True, False)
    # main(10, 'void', 'LSTM-TabDDPM', 'breast', '', 10, 10, './saved_severeOutcomePred/', 'train', True, True)
    # main(10, 'void', 'LSTM-MLP', 'breast', '', 10, 10, './saved_severeOutcomePred/', 'train', True, False)
    # main(10, 'void', 'LSTM-medGAN', 'breast', '', 10, 10, './saved_severeOutcomePred/', 'train', True, True)
    # main(10, 'void', 'synTEG', 'breast', '', 10, 10, './saved_severeOutcomePred/', 'train', True, False)
    # main(10, 'void', 'TWIN', 'breast', '', 10, 10, './saved_severeOutcomePred/', 'train', True, False)
    # main(10, 'void', 'MedDiffGa', 'breast', '', 10, 10, './saved_severeOutcomePred/', 'train', True, False)
    # main(10, 'void', 'promptEHR', 'breast', '', 10, 10, './saved_severeOutcomePred/', 'train', True, False)
    # #
    # main(10, 'fakeOnly', 'EVA', 'breast', '', 10, 10, './saved_severeOutcomePred/', 'train', True, False)
    # main(10, 'fakeOnly', 'LSTM-TabDDPM', 'breast', '', 10, 10, './saved_severeOutcomePred/', 'train', True, True)
    # main(10, 'fakeOnly', 'LSTM-MLP', 'breast', '', 10, 10, './saved_severeOutcomePred/', 'train', True, False)
    # main(10, 'fakeOnly', 'LSTM-medGAN', 'breast', '', 10, 10, './saved_severeOutcomePred/', 'train', True, True)
    # main(10, 'fakeOnly', 'synTEG', 'breast', '', 10, 10, './saved_severeOutcomePred/', 'train', True, False)
    # main(10, 'fakeOnly', 'TWIN', 'breast', '', 10, 10, './saved_severeOutcomePred/', 'train', True, False)
    # main(10, 'fakeOnly', 'MedDiffGa', 'breast', '', 10, 10, './saved_severeOutcomePred/', 'train', True, False)
    # main(10, 'fakeOnly', 'promptEHR', 'breast', '', 10, 10, './saved_severeOutcomePred/', 'train', True, False)
    # #

    # main(10, 'void', 'HALO', 'breast', '', 10, 10, './saved_severeOutcomePred/', 'train', True, False)
    main(10, 'fakeOnly', 'HALO', 'breast', '', 10, 10, './saved_severeOutcomePred/', 'train', True, False)










