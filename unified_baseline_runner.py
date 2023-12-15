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
from models.unified_baseline import *
from models.generator_baseline import *
from utils.utils import check_path, export_config, bool_flag
from utils.icd_rel import *
from models.evaluators import Evaluator
import warnings
import random
torch.autograd.set_detect_anomaly(True)

def main(seed, name, model, data, max_len, max_num, sav_dir, mode, short_ICD):
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
    parser.add_argument('--model', default=model)
    parser.add_argument('--name', default=name)
    parser.add_argument('--save_dir', default=sav_dir)
    parser.add_argument('--lambda_timegap', default=0.11, type=float)
    parser.add_argument('--lambda_diff', default=0.5, type=float)
    parser.add_argument('--lambda_ce', default=10, type=float)
    parser.add_argument('--short_ICD', default=short_ICD, type=bool_flag, nargs='?', const=True, help='use short ICD codes')
    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'gen':
        gen(args)
    else:
        raise ValueError('Invalid mode')

def gen(args):
    pass

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
    csv_filename = str(args.model) + '_' + str(args.target_disease) + '_' + str(args.seed) + '.csv'
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

        if args.name == 'GAN':
            if args.short_ICD:
                train_dataset = pancreas_Gendataset(data_path + 'train_3dig' + str(args.target_disease) + '.csv',
                                                    args.max_len, args.max_num_codes, pad_id_list, nan_id_list, device)
                dev_dataset = pancreas_Gendataset(data_path + 'val_3dig' + str(args.target_disease) + '.csv',
                                                  args.max_len, args.max_num_codes, pad_id_list, nan_id_list, device)
                test_dataset = pancreas_Gendataset(data_path + 'test_3dig' + str(args.target_disease) + '.csv',
                                                   args.max_len,
                                                   args.max_num_codes, pad_id_list, nan_id_list, device)
            else:

                train_dataset = pancreas_Gendataset(data_path + 'train_5dig' + str(args.target_disease) + '.csv',
                                                    args.max_len, args.max_num_codes, pad_id_list, nan_id_list, device)
                dev_dataset = pancreas_Gendataset(data_path + 'val_5dig' + str(args.target_disease) + '.csv',
                                                  args.max_len, args.max_num_codes, pad_id_list, nan_id_list, device)
                test_dataset = pancreas_Gendataset(data_path + 'test_5dig' + str(args.target_disease) + '.csv',
                                                   args.max_len,
                                                   args.max_num_codes, pad_id_list, nan_id_list, device)
            train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=gen_collate_fn)
            dev_dataloader = DataLoader(dev_dataset, args.batch_size, shuffle=False, collate_fn=gen_collate_fn)
            test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=gen_collate_fn)

            if args.model == 'medGAN':
                GAN = medGAN(d_model=args.d_model, h_model=args.h_model, dropout=args.dropout).to(device)
                h_learner = LSTM_hidden_state_learner(d_model=args.d_model, dropout=args.dropout).to(device)
                model = baseline_wrapper(args.name, pad_id_list, args.d_model, args.dropout, h_learner, GAN)
            else:
                raise ValueError('Invalid model')
            model.to(device)
            evaluator = Evaluator(model, device)

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': args.weight_decay, 'lr': args.learning_rate},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0, 'lr': args.learning_rate}
            ]
            optim = Adam(grouped_parameters)
            if args.name == 'name':
                generator_parameters = GAN.generator.parameters()
                discriminator_parameters = GAN.discriminator.parameters()
                optim_generator = Adam(generator_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
                optim_discriminator = Adam(discriminator_parameters, lr=args.learning_rate,
                                           weight_decay=args.weight_decay)

            MSE = nn.MSELoss(reduction='mean')
            CE = nn.BCEWithLogitsLoss(reduction='mean')

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

            for epoch_id in range(args.n_epochs):
                print('epoch: {:5} '.format(epoch_id))
                model.train()
                start_time = time.time()

                for i, data in enumerate(tqdm(train_dataloader, desc="Training", leave=False)):
                    optim.zero_grad()
                    diag_seq, drug_seq, lab_seq, proc_seq, time_step, visit_timegaps, diag_timegaps, drug_timegaps, lab_timegaps, proc_timegaps,\
                        diag_mask, drug_mask, lab_mask, proc_mask, diag_length, drug_length, lab_length, proc_length, demo = data
                    if args.model == 'medGAN':
                        real_diag_logits, real_drug_logits, real_lab_logits, real_proc_logits, gen_diag_logits, gen_drug_logits, gen_lab_logits, gen_proc_logits, h, v_gen, real_discrimination, gen_discrimination = model(diag_seq, drug_seq, lab_seq, proc_seq)
                    else:
                        raise ValueError('Invalid model')

                    if args.name == 'GAN':
                        optim_discriminator.zero_grad()

                        real_labels = torch.ones(h.size(0), 1).to(device)
                        fake_labels = torch.zeros(v_gen.size(0), 1).to(device)

                        loss_real = CE(real_discrimination, real_labels)
                        loss_fake = CE(gen_discrimination, fake_labels)

                        d_loss = (loss_real + loss_fake)/2
                        d_loss.backward()
                        optim_discriminator.step()

                        g_loss = CE(gen_discrimination, real_labels)

                        optim_generator.zero_grad()

                        g_loss.backward()
                        optim_generator.step()
