import argparse
import os
import time
import csv
import ast
import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
from models.dataset import pancreas_Gendataset, gen_collate_fn
from torch.utils.data import Dataset, DataLoader
from models.toy import *
from utils.utils import check_path, export_config, bool_flag
from utils.icd_rel import *
from models.evaluators import Evaluator
import warnings
import random
torch.autograd.set_detect_anomaly(True)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


# def eval_metric(dataloader, model, pad_id, args):
#     CE = nn.BCEWithLogitsLoss(reduction='mean')
#     MSE = nn.MSELoss(reduction='mean')
#     total_loss = 0
#     total_samples = 0
#
#     with torch.no_grad():
#         model.eval()
#         for data in tqdm(dataloader, desc="Evaluating"):
#             ehr, time_step, code_timegaps, code_mask, lengths, visit_timegaps, demo = data
#             logits, Delta_ts, added_z, learned_z = model(ehr, None, lengths, time_step, code_mask, code_timegaps, visit_timegaps, demo)
#
#             # Create multi-hot representation for ehr
#             multihot_ehr = torch.zeros_like(logits, dtype=torch.float32)
#             for batch_idx in range(ehr.size(0)):
#                 for seq_idx in range(ehr.size(1)):
#                     for label in ehr[batch_idx, seq_idx]:
#                         if label != pad_id:
#                             multihot_ehr[batch_idx, seq_idx, label] = 1.0
#
#             # Calculate losses
#             a = CE(logits, multihot_ehr)
#             b = torch.log(MSE(Delta_ts, visit_timegaps) + 1e-10)
#             c = MSE(added_z, learned_z)
#             loss = a * args.lambda_ce + args.lambda_timegap * b + args.lambda_diff * c
#
#             # Accumulate total loss and count the number of samples
#             total_loss += loss.item() * ehr.size(0)
#             total_samples += ehr.size(0)
#
#     # Compute mean loss
#     mean_loss = total_loss / total_samples
#     return mean_loss

def main(seed, name, data, max_len, max_num, sav_dir, mode, focal_alpha, focal_gamma, short_ICD):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=True, type=bool_flag, nargs='?', const=True, help='use GPU')
    parser.add_argument('--seed', default=seed, type=int, help='seed')
    parser.add_argument('-bs', '--batch_size', default=32, type=int)
    parser.add_argument('-me', '--max_epochs_before_stop', default=15, type=int)
    parser.add_argument('--d_model', default=256, type=int, help='dimension of hidden layers')
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
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=0.001, type=float)
    parser.add_argument('--target_rate', default=0.3, type=float)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='max grad norm (0 to disable)')
    parser.add_argument('--warmup_steps', default=200, type=int)
    parser.add_argument('--n_epochs', default=5, type=int)
    parser.add_argument('--log_interval', default=100, type=int)
    parser.add_argument('--mode', default=mode)
    parser.add_argument('--model', default=name)
    parser.add_argument('--save_dir', default=sav_dir)
    parser.add_argument('--lambda_timegap', default=0.11, type=float)
    parser.add_argument('--lambda_diff', default=0.5, type=float)
    parser.add_argument('--lambda_ce', default=10, type=float)
    parser.add_argument('--focal_alpha', default=focal_alpha, type=float)
    parser.add_argument('--focal_gamma', default=focal_gamma, type=float)
    parser.add_argument('--short_ICD', default=short_ICD, type=bool_flag, nargs='?', const=True, help='use short ICD codes')
    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'gen':
        gen(args)
    else:
        raise ValueError('Invalid mode')

def train(args):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    files = os.listdir(str(args.save_dir))
    csv_filename = str(args.model) + '_' + str(args.target_disease) + '_' + str(args.seed) + '_' + str(args.focal_alpha) + '_' + str(args.focal_gamma) + '.csv'
    if csv_filename in files:
        print("conducted_experiments")
    else:
        config_path = os.path.join(args.save_dir, 'config.json')
        model_path = os.path.join(args.save_dir, str(args.model) + '_' + str(args.target_disease) + '_' + str(args.seed) + '_' + str(args.focal_alpha) + '_' + str(args.focal_gamma) + '.pt')
        log_path = os.path.join(args.save_dir, 'log.csv')
        # export_config(args, config_path)
        check_path(model_path)
        # with open(log_path, 'w') as fout:
        #     fout.write('step,train_auc,dev_auc,test_auc\n')

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
        else:
            raise ValueError('Invalid disease')
        device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

        if args.model in ('MedDiffGa'):

            if args.short_ICD:

                train_dataset = pancreas_Gendataset(data_path + 'train_3dig' + str(args.target_disease) + '.csv',
                                          args.max_len, args.max_num_codes, pad_id_list, nan_id_list, device)
                dev_dataset = pancreas_Gendataset(data_path + 'val_3dig' + str(args.target_disease) + '.csv',
                                        args.max_len,args.max_num_codes, pad_id_list, nan_id_list, device)
                test_dataset = pancreas_Gendataset(data_path + 'test_3dig' + str(args.target_disease) + '.csv',  args.max_len,
                                         args.max_num_codes, pad_id_list, nan_id_list, device)

                # train_dataset = pancreas_Gendataset(data_path + 'toy_3dig' + str(args.target_disease) + '.csv',
                #                           args.max_len, args.max_num_codes, pad_id_list, nan_id_list, device)
                # dev_dataset = pancreas_Gendataset(data_path + 'toy_3dig' + str(args.target_disease) + '.csv',
                #                         args.max_len,args.max_num_codes, pad_id_list, nan_id_list, device)
                # test_dataset = pancreas_Gendataset(data_path + 'toy_3dig' + str(args.target_disease) + '.csv',  args.max_len,
                #                          args.max_num_codes, pad_id_list, nan_id_list, device)
            else:
                train_dataset = pancreas_Gendataset(data_path + 'train_5dig' + str(args.target_disease) + '.csv',
                                          args.max_len, args.max_num_codes, pad_id_list, nan_id_list, device)
                dev_dataset = pancreas_Gendataset(data_path + 'val_5dig' + str(args.target_disease) + '.csv',
                                        args.max_len,args.max_num_codes, pad_id_list, nan_id_list, device)
                test_dataset = pancreas_Gendataset(data_path + 'test_5dig' + str(args.target_disease) + '.csv',  args.max_len,
                                         args.max_num_codes, pad_id_list, nan_id_list, device)
            train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=gen_collate_fn)
            dev_dataloader = DataLoader(dev_dataset, args.batch_size, shuffle=False, collate_fn=gen_collate_fn)
            test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=gen_collate_fn)
        if args.model == 'MedDiffGa':
                model = MedDiffGa(pad_id_list, args.d_model, args.dropout, args.dropout_emb, args.num_layers, demo_len, device)
        else:
                raise ValueError('Invalid model')
        model.to(device)

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.learning_rate},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': args.learning_rate}
        ]
        optim = Adam(grouped_parameters)
        # CE = nn.BCEWithLogitsLoss(reduction='mean')
        MSE = nn.MSELoss(reduction='mean')

        print('parameters:')
        for name, param in model.named_parameters():
            if param.requires_grad:
                print('\t{:45}\ttrainable\t{}'.format(name, param.size()))
            else:
                print('\t{:45}\tfixed\t{}'.format(name, param.size()))
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('\ttotal:', num_params)
        print()
        print('-' * 71)
        global_step, best_dev_epoch, total_loss = 0, 0, 0.0
        CE_loss, Time_loss, Diff_loss = 0.0, 0.0, 0.0
        best_diag_lpl, best_drug_lpl, best_lab_lpl, best_proc_lpl = 1e10, 1e10, 1e10, 1e10
        best_diag_mpl, best_drug_mpl, best_lab_mpl, best_proc_mpl = 1e10, 1e10, 1e10, 1e10
        best_choosing_statistic = 1e10
        eva = Evaluator(model, device)
        focal = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, reduction='mean')
        model.train()
        for epoch_id in range(args.n_epochs):
            print('epoch: {:5} '.format(epoch_id))
            model.train()
            start_time = time.time()

            train_dataloader = tqdm(train_dataloader)
            for i, data in enumerate(train_dataloader):
                optim.zero_grad()
                # if args.model in ('toy', 'Hitatime'):
                #     ehr, time_step, code_timegaps, labels, code_mask, lengths, visit_timegaps = data
                #     outputs = model(ehr, None, lengths, time_step, code_mask, code_timegaps, visit_timegaps)
                if args.model in ('MedDiffGa'):
                    # ehr, time_step, code_timegaps, code_mask, lengths, visit_timegaps, demo = data
                    # logits, Delta_ts, added_z, learned_z = model(ehr, None, lengths, time_step, code_mask, code_timegaps, visit_timegaps, demo)
                    diag_seq, drug_seq, lab_seq, proc_seq, time_step, visit_timegaps, diag_timegaps, drug_timegaps, lab_timegaps, proc_timegaps,\
                        diag_mask, drug_mask, lab_mask, proc_mask, diag_length, drug_length, lab_length, proc_length, demo = data
                    diag_logits, drug_logits, lab_logits, proc_logits, Delta_ts, added_z, learned_z = model(diag_seq, drug_seq, lab_seq, proc_seq, time_step, visit_timegaps, diag_timegaps, drug_timegaps, lab_timegaps, proc_timegaps,\
                        diag_mask, drug_mask, lab_mask, proc_mask, diag_length, drug_length, lab_length, proc_length, demo)
                else:
                    raise ValueError('Invalid model')
                multihot_diag = torch.zeros_like(diag_logits, dtype=torch.float32)
                multihot_drug = torch.zeros_like(drug_logits, dtype=torch.float32)
                multihot_lab = torch.zeros_like(lab_logits, dtype=torch.float32)
                multihot_proc = torch.zeros_like(proc_logits, dtype=torch.float32)

                # valid_diag_mask = diag_seq != diag_pad_id
                # valid_drug_mask = (drug_seq != drug_pad_id) & (drug_seq != drug_nan_id)
                # valid_lab_mask = (lab_seq != lab_pad_id) & (lab_seq != lab_nan_id)
                # valid_proc_mask = (proc_seq != proc_pad_id) & (proc_seq != proc_nan_id)

                valid_diag_mask = diag_mask
                valid_drug_mask = drug_mask
                valid_lab_mask = lab_mask
                valid_proc_mask = proc_mask

                valid_diag_batch_indices, valid_diag_seq_indices, valid_diag_code_indices = torch.where(valid_diag_mask)
                valid_drug_batch_indices, valid_drug_seq_indices, valid_drug_code_indices = torch.where(valid_drug_mask)
                valid_lab_batch_indices, valid_lab_seq_indices, valid_lab_code_indices = torch.where(valid_lab_mask)
                valid_proc_batch_indices, valid_proc_seq_indices, valid_proc_code_indices = torch.where(valid_proc_mask)

                multihot_diag[valid_diag_batch_indices, valid_diag_seq_indices, valid_diag_code_indices] = 1.0
                multihot_drug[valid_drug_batch_indices, valid_drug_seq_indices, valid_drug_code_indices] = 1.0
                multihot_lab[valid_lab_batch_indices, valid_lab_seq_indices, valid_lab_code_indices] = 1.0
                multihot_proc[valid_proc_batch_indices, valid_proc_seq_indices, valid_proc_code_indices] = 1.0

                sequence_range = torch.arange(args.max_len, device=diag_seq.device).expand(diag_seq.size(0), args.max_len)
                expanded_lengths = diag_length.unsqueeze(-1).expand_as(sequence_range)
                length_mask = (sequence_range < expanded_lengths).float()

                # for batch_idx in range(diag_seq.size(0)):
                #     for seq_idx in range(diag_seq.size(1)):
                #         for label in diag_seq[batch_idx, seq_idx]:
                #             if label != pad_id:
                #                 multihot_diag[batch_idx, seq_idx, label] = 1.0



                # length_mask = torch.zeros(ehr.size(0), args.max_len, dtype=torch.float32).to(device)
                # #mask timestep based on length
                # for i in range(ehr.size(0)):
                #     length_mask[i, :lengths[i]] = 1.0

                # a = focal(logits, multihot_ehr) * args.lambda_ce
                a = (focal(diag_logits, multihot_diag) + focal(drug_logits, multihot_drug) + focal(lab_logits, multihot_lab) + focal(proc_logits, multihot_proc)) * args.lambda_ce
                b = torch.log(MSE(Delta_ts * length_mask, visit_timegaps * length_mask) + 1e-10) * args.lambda_timegap
                c = MSE(added_z, learned_z) * args.lambda_diff
                loss = a + b + c
                # loss = CE(logits, multihot_ehr) + MSE(Delta_ts, visit_timegaps) + MSE(added_z, learned_z)
                loss.backward()
                total_loss += (loss.item() / visit_timegaps.size(0)) * args.batch_size
                CE_loss += (a.item() / visit_timegaps.size(0)) * args.batch_size
                Time_loss += (b.item() / visit_timegaps.size(0)) * args.batch_size
                Diff_loss += (c.item() / visit_timegaps.size(0)) * args.batch_size

                if args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optim.step()
                if (global_step + 1) % args.log_interval == 0:
                    total_loss /= args.log_interval
                    CE_loss /= args.log_interval
                    Time_loss /= args.log_interval
                    Diff_loss /= args.log_interval
                    ms_per_batch = 1000 * (time.time() - start_time) / args.log_interval
                    print('| step {:5} | loss {:7.4f} | ms/batch {:7.2f} |'.format(global_step,
                                                                                   total_loss,
                                                                                   ms_per_batch))
                    print('| CE loss {:7.4f} | Time loss {:7.4f} | Diff loss {:7.4f} |'.format(CE_loss, Time_loss, Diff_loss))
                    total_loss, CE_loss, Time_loss, Diff_loss = 0.0, 0.0, 0.0, 0.0
                    start_time = time.time()
                global_step += 1
            drug_nan_id = 146
            lab_nan_id = 206
            proc_nan_id = 24
            model.eval()
            if args.model in ('MedDiffGa'):
                train_res = eva.eval(train_dataloader, [diag_pad_id, drug_pad_id, lab_pad_id, proc_pad_id], [diag_pad_id, drug_nan_id, lab_nan_id, proc_nan_id], ['diag', 'drug', 'lab', 'proc'], ['lpl', 'mpl'])
                val_res = eva.eval(dev_dataloader, [diag_pad_id, drug_pad_id, lab_pad_id, proc_pad_id], [diag_pad_id, drug_nan_id, lab_nan_id, proc_nan_id], ['diag', 'drug', 'lab', 'proc'], ['lpl', 'mpl'])
                test_res = eva.eval(test_dataloader, [diag_pad_id, drug_pad_id, lab_pad_id, proc_pad_id], [diag_pad_id, drug_nan_id, lab_nan_id, proc_nan_id], ['diag', 'drug', 'lab', 'proc'], ['lpl', 'mpl'])
                train_diag_lpl, tran_drug_lpl, train_lab_lpl, train_proc_lpl = train_res['lpl_diag'], train_res['lpl_drug'], train_res['lpl_lab'], train_res['lpl_proc']
                val_diag_lpl, val_drug_lpl, val_lab_lpl, val_proc_lpl = val_res['lpl_diag'], val_res['lpl_drug'], val_res['lpl_lab'], val_res['lpl_proc']
                test_diag_lpl, test_drug_lpl, test_lab_lpl, test_proc_lpl = test_res['lpl_diag'], test_res['lpl_drug'], test_res['lpl_lab'], test_res['lpl_proc']

                train_diag_mpl, tran_drug_mpl, train_lab_mpl, train_proc_mpl = train_res['mpl_diag'], train_res['mpl_drug'], train_res['mpl_lab'], train_res['mpl_proc']
                val_diag_mpl, val_drug_mpl, val_lab_mpl, val_proc_mpl = val_res['mpl_diag'], val_res['mpl_drug'], val_res['mpl_lab'], val_res['mpl_proc']
                test_diag_mpl, test_drug_mpl, test_lab_mpl, test_proc_mpl = test_res['mpl_diag'], test_res['mpl_drug'], test_res['mpl_lab'], test_res['mpl_proc']

                choosing_statistic = np.median([val_diag_lpl, val_drug_lpl, val_lab_lpl, val_proc_lpl, val_diag_mpl, val_drug_mpl, val_lab_mpl, val_proc_mpl])

            else:
                raise ValueError('Invalid model')
            print('-' * 71)
            # print('train lpl {:7.4f} | dev lpl {:7.4f} | test lpl {:7.4f}'.format(train_lpl, val_lpl, test_lpl))
            print('Epoch: {:5}'.format(epoch_id))
            print('Diagnosis:')
            print('train lpl {:7.4f} | dev lpl {:7.4f} | test lpl {:7.4f}'.format(train_diag_lpl, val_diag_lpl, test_diag_lpl))
            print('train mpl {:7.4f} | dev mpl {:7.4f} | test mpl {:7.4f}'.format(train_diag_mpl, val_diag_mpl, test_diag_mpl))
            print('Drug:')
            print('train lpl {:7.4f} | dev lpl {:7.4f} | test lpl {:7.4f}'.format(tran_drug_lpl, val_drug_lpl, test_drug_lpl))
            print('train mpl {:7.4f} | dev mpl {:7.4f} | test mpl {:7.4f}'.format(tran_drug_mpl, val_drug_mpl, test_drug_mpl))
            print('Lab:')
            print('train lpl {:7.4f} | dev lpl {:7.4f} | test lpl {:7.4f}'.format(train_lab_lpl, val_lab_lpl, test_lab_lpl))
            print('train mpl {:7.4f} | dev mpl {:7.4f} | test mpl {:7.4f}'.format(train_lab_mpl, val_lab_mpl, test_lab_mpl))
            print('Procedure:')
            print('train lpl {:7.4f} | dev lpl {:7.4f} | test lpl {:7.4f}'.format(train_proc_lpl, val_proc_lpl, test_proc_lpl))
            print('train mpl {:7.4f} | dev mpl {:7.4f} | test mpl {:7.4f}'.format(train_proc_mpl, val_proc_mpl, test_proc_mpl))
            print('-' * 71)

            if choosing_statistic <= best_choosing_statistic:
                best_diag_lpl, best_drug_lpl, best_lab_lpl, best_proc_lpl = test_diag_lpl, test_drug_lpl, test_lab_lpl, test_proc_lpl
                best_diag_mpl, best_drug_mpl, best_lab_mpl, best_proc_mpl = test_diag_mpl, test_drug_mpl, test_lab_mpl, test_proc_mpl
                best_dev_epoch = epoch_id
                best_choosing_statistic = choosing_statistic
                torch.save([model, args], model_path)
                # torch.save([model, args], model_path)
                # with open(log_path, 'a') as fout:
                #     fout.write('{},{},{},{}\n'.format(global_step, tr_pr_auc, d_pr_auc, t_pr_auc))
                print(f'model saved to {model_path}')
            if epoch_id - best_dev_epoch >= args.max_epochs_before_stop:
                break

        print()
        print('training ends at {} epoch'.format(epoch_id))
        print('Final statistics:')
        print('lpl: diag {:7.4f} | drug {:7.4f} | lab {:7.4f} | proc {:7.4f}'.format(best_diag_lpl, best_drug_lpl, best_lab_lpl, best_proc_lpl))
        print('mpl: diag {:7.4f} | drug {:7.4f} | lab {:7.4f} | proc {:7.4f}'.format(best_diag_mpl, best_drug_mpl, best_lab_mpl, best_proc_mpl))
        with open(args.save_dir + csv_filename, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([best_diag_lpl, best_drug_lpl, best_lab_lpl, best_proc_lpl, best_diag_mpl, best_drug_mpl, best_lab_mpl, best_proc_mpl])
        print()

def gen(args):
    print(args)

    model_path = os.path.join(args.save_dir, str(args.model) + '_' + str(args.target_disease) + '_' + str(args.seed) + '.pt')
    check_path(model_path)

    if args.target_disease == 'pancreas':
        code2id = pd.read_csv('./data/pancreas/code_to_int_mapping.csv', header=None)
        demo_len = 15
        pad_id = len(code2id)
        data_path = './data/pancreas/'
        id2code = dict(zip(code2id[1], code2id[0]))
    elif args.target_disease == 'mimic':
        code2id = pd.read_csv('./data/mimic/code_to_int_mapping.csv', header=None)
        demo_len = 70
        pad_id = len(code2id)
        data_path = './data/mimic/'
        id2code = dict(zip(code2id[1], code2id[0]))
    else:
        raise ValueError('Invalid disease')
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

    if args.model in ('MedDiffGa'):

        train_dataset = pancreas_Gendataset(data_path + 'train_' + str(args.target_disease) + '.csv',
                                  args.max_len, args.max_num_codes, pad_id, device)
        dev_dataset = pancreas_Gendataset(data_path + 'val_' + str(args.target_disease) + '.csv',
                                args.max_len,args.max_num_codes, pad_id, device)
        test_dataset = pancreas_Gendataset(data_path + 'test_' + str(args.target_disease) + '.csv',  args.max_len,
                                 args.max_num_codes, pad_id, device)
        train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=gen_collate_fn)
        dev_dataloader = DataLoader(dev_dataset, args.batch_size, shuffle=False, collate_fn=gen_collate_fn)
        test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=gen_collate_fn)
    if args.model == 'MedDiffGa':
            model = MedDiffGa(pad_id, args.d_model, args.dropout, args.dropout_emb, args.num_layers, demo_len, device)
    else:
            raise ValueError('Invalid model')
    model.to(device)
    v_len = 20
    c_len = 10
    # total_patient = 2000
    # demo = torch.randint(0, 2, (total_patient, demo_len), dtype=torch.float).to(device)
    demo_csv = pd.read_csv(data_path + 'toy_' + str(args.target_disease) + '.csv')
    demo = demo_csv['Demographic'].apply(lambda x: ast.literal_eval(x)).tolist()
    demo = torch.tensor(demo, dtype=torch.float).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)[0].state_dict())
    model.eval()
    with torch.no_grad():

        logits, ehr, time_gap = model.inference(demo, v_len, c_len)

    def map_indices_to_codes(id2code, code_selection_list):
        # Convert indices to codes
        mapped_data = [[[id2code.get(code, 'Unknown') for code in visit] for visit in patient] for patient in
                       code_selection_list]
        return mapped_data

    # Map the generated indices in EHR to actual codes
    mapped_ehr = map_indices_to_codes(id2code, ehr.cpu().numpy())

    # Flatten EHR and Time Gaps for saving to CSV
    formatted_ehr = ['; '.join([', '.join(map(str, visit)) for visit in patient]) for patient in mapped_ehr]
    formatted_time_gap = ['; '.join(map(str, patient)) for patient in time_gap.cpu().numpy()]

    # Create a DataFrame and save to CSV
    ehr_df = pd.DataFrame({
        'Patient_EHR': formatted_ehr,
        'Time_Gaps': formatted_time_gap
    })
    ehr_df.to_csv(args.save_dir + 'generated_ehr' + args.seed + '.csv', index=False)




if __name__ == '__main__':

    modes = ['train']
    short_ICD = True
    seeds = [10, 11, 12]
    focal_alphas = [0.5]
    focal_gammas = [1.0]
    # seeds = [11]
    # names = ['toy','LSTM','Hitatime', 'Hita']
    save_path = './saved_'
    names = ['MedDiffGa']
    save_dirs = [save_path+name+'/' for name in names]
    datas = ['mimic']
    max_lens = [20]
    max_nums = [10]
    for mode in modes:
        for seed in seeds:
            for focal_alpha in focal_alphas:
                for focal_gamma in focal_gammas:
                    for name, save_dir in zip(names, save_dirs):
                        for data, max_len, max_num in zip(datas, max_lens, max_nums):
                            main(seed, name, data, max_len, max_num, save_dir, mode, focal_alpha, focal_gamma, short_ICD)

    # csv_path = './saved_models/'
    #
    # # List all files with the specified pattern
    # csv_files = [f for f in os.listdir(csv_path) if f.endswith('.csv')]
    #
    # # Container to hold results
    # results = {}
    #
    # # Process each CSV file
    # for csv_file in csv_files:
    #     # Extract model name, dataset, and seed from the file name
    #     parts = csv_file.split('_')
    #     model_name, dataset = parts[0], parts[1]
    #     seed = parts[2].split('.')[0]  # remove .csv from the end and get seed
    #
    #     # Verify that the seed is a number
    #     if not seed.isdigit():
    #         continue
    #
    #     # Read the CSV
    #     data = pd.read_csv(os.path.join(csv_path, csv_file), header=None).values
    #
    #     # Store the results
    #     if (model_name, dataset) not in results:
    #         results[(model_name, dataset)] = []
    #     results[(model_name, dataset)].append(data)
    #
    # # Compute means
    # mean_results = {}
    # for key, values in results.items():
    #     mean_values = sum(values) / len(values)
    #     mean_results[key] = mean_values.mean(axis=0)  # mean across seeds for each metric
    #
    # # Print results
    # for key, value in mean_results.items():
    #     print(f'Model: {key[0]}, Dataset: {key[1]}, PR Mean: {value[0]:.4f}, F1 Mean: {value[1]:.4f}, Kappa Mean: {value[2]:.4f}')