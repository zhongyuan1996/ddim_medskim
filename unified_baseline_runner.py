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
        model_path = os.path.join(args.save_dir, str(args.model) + '_' + str(args.target_disease) + '_' + str(args.seed) + '.pt')
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
            generation_module_loss, prediction_module_loss = 0.0, 0.0
            choosing_statistic = 1e10


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
                        loss_fake = CE(gen_discrimination.detach(), fake_labels)

                        d_loss = (loss_real + loss_fake)/2
                        d_loss.backward()
                        optim_discriminator.step()
                        g_loss = CE(gen_discrimination, real_labels)
                        optim_generator.zero_grad()

                        g_loss.backward()
                        optim_generator.step()

                        multihot_diag = torch.zeros_like(real_diag_logits, dtype=torch.float32)
                        multihot_drug = torch.zeros_like(real_drug_logits, dtype=torch.float32)
                        multihot_lab = torch.zeros_like(real_lab_logits, dtype=torch.float32)
                        multihot_proc = torch.zeros_like(real_proc_logits, dtype=torch.float32)

                        valid_diag_batch_indices, valid_diag_seq_indices, valid_diag_code_indices = torch.where(
                            diag_mask)
                        valid_drug_batch_indices, valid_drug_seq_indices, valid_drug_code_indices = torch.where(
                            drug_mask)
                        valid_lab_batch_indices, valid_lab_seq_indices, valid_lab_code_indices = torch.where(
                            lab_mask)
                        valid_proc_batch_indices, valid_proc_seq_indices, valid_proc_code_indices = torch.where(
                            proc_mask)
                        multihot_diag[valid_diag_batch_indices, valid_diag_seq_indices, valid_diag_code_indices] = 1.0
                        multihot_drug[valid_drug_batch_indices, valid_drug_seq_indices, valid_drug_code_indices] = 1.0
                        multihot_lab[valid_lab_batch_indices, valid_lab_seq_indices, valid_lab_code_indices] = 1.0
                        multihot_proc[valid_proc_batch_indices, valid_proc_seq_indices, valid_proc_code_indices] = 1.0

                        loss = (CE(real_diag_logits, multihot_diag) + CE(real_drug_logits, multihot_drug) + CE(real_lab_logits, multihot_lab) + CE(real_proc_logits, multihot_proc))/4 + (CE(gen_diag_logits, multihot_diag) + CE(gen_drug_logits, multihot_drug) + CE(gen_lab_logits, multihot_lab) + CE(gen_proc_logits, multihot_proc))/4
                        loss.backward()

                        prediction_module_loss += (loss.item() / visit_timegaps.size(0)) * args.batch_size
                        generation_module_loss += (g_loss.item() + d_loss.item() / visit_timegaps.size(0)) * args.batch_size

                        if args.max_grad_norm > 0:
                            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        optim.step()

                        if (global_step + 1) % args.log_interval == 0:
                            avg_prediction_module_loss = prediction_module_loss / args.log_interval
                            avg_generation_module_loss = generation_module_loss / args.log_interval

                            print(
                                '| step {:5} | prediction module loss {:7.4f} | generation module loss {:7.4f} |'.format(
                                    global_step, avg_prediction_module_loss, avg_generation_module_loss))

                            prediction_module_loss, generation_module_loss = 0.0, 0.0
                            start_time = time.time()
                        global_step += 1
                    model.eval()

                    train_res = evaluator.eval(train_dataloader, [diag_pad_id, drug_pad_id, lab_pad_id, proc_pad_id],
                                         [diag_pad_id, drug_nan_id, lab_nan_id, proc_nan_id],
                                         ['diag', 'drug', 'lab', 'proc'], ['lpl', 'mpl'])
                    val_res = evaluator.eval(dev_dataloader, [diag_pad_id, drug_pad_id, lab_pad_id, proc_pad_id],
                                       [diag_pad_id, drug_nan_id, lab_nan_id, proc_nan_id],
                                       ['diag', 'drug', 'lab', 'proc'], ['lpl', 'mpl'])
                    test_res = evaluator.eval(test_dataloader, [diag_pad_id, drug_pad_id, lab_pad_id, proc_pad_id],
                                        [diag_pad_id, drug_nan_id, lab_nan_id, proc_nan_id],
                                        ['diag', 'drug', 'lab', 'proc'], ['lpl', 'mpl'])
                    train_diag_lpl, tran_drug_lpl, train_lab_lpl, train_proc_lpl = train_res['lpl_diag'], train_res[
                        'lpl_drug'], train_res['lpl_lab'], train_res['lpl_proc']
                    val_diag_lpl, val_drug_lpl, val_lab_lpl, val_proc_lpl = val_res['lpl_diag'], val_res['lpl_drug'], \
                    val_res['lpl_lab'], val_res['lpl_proc']
                    test_diag_lpl, test_drug_lpl, test_lab_lpl, test_proc_lpl = test_res['lpl_diag'], test_res[
                        'lpl_drug'], test_res['lpl_lab'], test_res['lpl_proc']

                    train_diag_mpl, tran_drug_mpl, train_lab_mpl, train_proc_mpl = train_res['mpl_diag'], train_res[
                        'mpl_drug'], train_res['mpl_lab'], train_res['mpl_proc']
                    val_diag_mpl, val_drug_mpl, val_lab_mpl, val_proc_mpl = val_res['mpl_diag'], val_res['mpl_drug'], \
                    val_res['mpl_lab'], val_res['mpl_proc']
                    test_diag_mpl, test_drug_mpl, test_lab_mpl, test_proc_mpl = test_res['mpl_diag'], test_res[
                        'mpl_drug'], test_res['mpl_lab'], test_res['mpl_proc']

                    choosing_statistic = np.median(
                        [val_diag_lpl, val_drug_lpl, val_lab_lpl, val_proc_lpl, val_diag_mpl, val_drug_mpl, val_lab_mpl,
                         val_proc_mpl])

                    print('-' * 71)
                    print('Epoch: {:5}'.format(epoch_id))
                    print('Diagnosis:')
                    print('train lpl {:7.4f} | dev lpl {:7.4f} | test lpl {:7.4f}'.format(train_diag_lpl, val_diag_lpl,
                                                                                          test_diag_lpl))
                    print('train mpl {:7.4f} | dev mpl {:7.4f} | test mpl {:7.4f}'.format(train_diag_mpl, val_diag_mpl,
                                                                                          test_diag_mpl))
                    print('Drug:')
                    print('train lpl {:7.4f} | dev lpl {:7.4f} | test lpl {:7.4f}'.format(tran_drug_lpl, val_drug_lpl,
                                                                                          test_drug_lpl))
                    print('train mpl {:7.4f} | dev mpl {:7.4f} | test mpl {:7.4f}'.format(tran_drug_mpl, val_drug_mpl,
                                                                                          test_drug_mpl))
                    print('Lab:')
                    print('train lpl {:7.4f} | dev lpl {:7.4f} | test lpl {:7.4f}'.format(train_lab_lpl, val_lab_lpl,
                                                                                          test_lab_lpl))
                    print('train mpl {:7.4f} | dev mpl {:7.4f} | test mpl {:7.4f}'.format(train_lab_mpl, val_lab_mpl,
                                                                                          test_lab_mpl))
                    print('Procedure:')
                    print('train lpl {:7.4f} | dev lpl {:7.4f} | test lpl {:7.4f}'.format(train_proc_lpl, val_proc_lpl,
                                                                                          test_proc_lpl))
                    print('train mpl {:7.4f} | dev mpl {:7.4f} | test mpl {:7.4f}'.format(train_proc_mpl, val_proc_mpl,
                                                                                          test_proc_mpl))
                    print('-' * 71)

                    if choosing_statistic <= best_choosing_statistic:
                        best_diag_lpl, best_drug_lpl, best_lab_lpl, best_proc_lpl = test_diag_lpl, test_drug_lpl, test_lab_lpl, test_proc_lpl
                        best_diag_mpl, best_drug_mpl, best_lab_mpl, best_proc_mpl = test_diag_mpl, test_drug_mpl, test_lab_mpl, test_proc_mpl
                        best_dev_epoch = epoch_id
                        best_choosing_statistic = choosing_statistic
                        torch.save([model, args], model_path)
                        print(f'model saved to {model_path}')
                    if epoch_id - best_dev_epoch >= args.max_epochs_before_stop:
                        break

                print()
                print('training ends at {} epoch'.format(epoch_id))
                print('Final statistics:')
                print(
                    'lpl: diag {:7.4f} | drug {:7.4f} | lab {:7.4f} | proc {:7.4f}'.format(best_diag_lpl, best_drug_lpl,
                                                                                           best_lab_lpl, best_proc_lpl))
                print(
                    'mpl: diag {:7.4f} | drug {:7.4f} | lab {:7.4f} | proc {:7.4f}'.format(best_diag_mpl, best_drug_mpl,
                                                                                           best_lab_mpl, best_proc_mpl))
                with open(args.save_dir + csv_filename, 'w') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(
                        [best_diag_lpl, best_drug_lpl, best_lab_lpl, best_proc_lpl, best_diag_mpl, best_drug_mpl,
                         best_lab_mpl, best_proc_mpl])
                print()

if __name__ == '__main__':

    modes = ['train']
    short_ICD = True
    seeds = [10, 11, 12]
    save_path = './saved_'
    model_names = ['medGAN']
    save_dirs = [save_path+name+'/' for name in model_names]
    datas = ['mimic']
    max_lens = [20]
    max_nums = [10]
    for mode in modes:
        for seed in seeds:
            for model_name, save_dir in zip(model_names, save_dirs):
                for data, max_len, max_num in zip(datas, max_lens, max_nums):
                    if model_name in ['medGAN']:
                        model_type = 'GAN'
                        main(seed, model_type, model_name, data, max_len, max_num, save_dir, mode, short_ICD)
