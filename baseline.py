import argparse
import os
import time
import csv

import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score, log_loss
from torch.optim import *
from sklearn.metrics import precision_recall_curve, auc
from models.og_dataset import *
from models.baseline import *
from models.leap_lstm import LeapLSTM
from models.skim_rnn import SkimRNN
from models.skiplstm import SkipLSTM
from utils.utils import check_path, export_config, bool_flag

def eval_metric(eval_set, model):
    model.eval()
    with torch.no_grad():
        y_true = np.array([])
        y_pred = np.array([])
        y_score = np.array([])
        for i, data in enumerate(eval_set):
            labels, ehr, mask, txt, mask_txt, lengths, time_step, code_mask = data
            logits = model(ehr, mask, lengths, time_step, code_mask)
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

def eval_metric_arf(eval_set, model, model_name = None):
    model.eval()
    with torch.no_grad():
        y_true = np.array([])
        y_pred = np.array([])
        y_score = np.array([])

        for i, data in enumerate(eval_set):
            labels, ehr, mask, txt, mask_txt, lengths, time_step, code_mask = data
            fp_og, fp_fake, _, _ = model(ehr, mask, lengths, time_step, code_mask)

            scores = fp_og.data.cpu().numpy()

            labels = labels.data.cpu().numpy()
            # if model_name != 'LSTM_actGAN' and model_name != 'LSTM_medGAN':
            # labels = labels.argmax(1)
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
        loss = log_loss(y_true, y_pred)

    return accuary, precision, recall, f1, roc_auc, pr_auc, kappa, loss

def main(name, seed, data, max_len, max_num):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=True, type=bool_flag, nargs='?', const=True, help='use GPU')
    parser.add_argument('--seed', default=seed, type=int, help='seed')
    parser.add_argument('-bs', '--batch_size', default=64, type=int)
    parser.add_argument('-me', '--max_epochs_before_stop', default=15, type=int)
    parser.add_argument('--encoder', default=name, choices=['hita', 'lsan', 'lstm', 'sand', 'gruself', 'timeline', 'retain', 'retainex', 'LeapLSTM', 'skimrnn', 'skiprnn','TLSTM', 'LSTM_ehrGAN'])
    parser.add_argument('--d_model', default=256, type=int, help='dimension of hidden layers')
    parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate of hidden layers')
    parser.add_argument('--dropout_emb', default=0.1, type=float, help='dropout rate of embedding layers')
    parser.add_argument('--num_layers', default=2, type=int, help='number of transformer layers of EHR encoder')
    parser.add_argument('--num_heads', default=4, type=int, help='number of attention heads')
    parser.add_argument('--max_len', default=max_len, type=int, help='max visits of EHR')
    parser.add_argument('--max_num_codes', default=max_num, type=int, help='max number of ICD codes in each visit')
    parser.add_argument('--max_num_blks', default=120, type=int, help='max number of blocks in each visit')
    parser.add_argument('--blk_emb_path', default='./data/processed/block_embedding.npy',
                        help='embedding path of blocks')
    parser.add_argument('--blk_vocab_path', default='./data/processed/block_vocab.txt')
    parser.add_argument('--target_disease', default=data, choices=['Heart_failure', 'COPD', 'Kidney', 'Dementia', 'Amnesia','mimic','ARF','Shock','mortality'])
    parser.add_argument('--target_att_heads', default=4, type=int, help='target disease attention heads number')
    parser.add_argument('--mem_size', default=20, type=int, help='memory size')
    parser.add_argument('--mem_update_size', default=15, type=int, help='memory update size')
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=0.001, type=float)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='max grad norm (0 to disable)')
    parser.add_argument('--warmup_steps', default=200, type=int)
    parser.add_argument('--n_epochs', default=30, type=int)
    parser.add_argument('--log_interval', default=20, type=int)
    parser.add_argument('--mode', default='train', choices=['train', 'eval', 'pred','gen'], help='run training or evaluation')
    parser.add_argument('--save_dir', default='./saved_models/', help='model output directory')
    parser.add_argument("--lambda_DF_loss", type=float, default=0.1, help="scale of diffusion model loss")
    parser.add_argument("--lambda_CE_gen_loss", type=float, default=0.5, help="scale of generated sample loss")
    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'pred':
        pred(args)
    else:
        raise ValueError('Invalid mode')


def train(args):
    print(args)
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # if torch.cuda.is_available() and args.cuda:
    #     torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    files = os.listdir(str(args.save_dir))
    if str(args.encoder) + '_' + str(args.target_disease) + '_' + str(args.seed) + '.csv' in files:
        print("conducted_experiments")
    else:

        config_path = os.path.join(args.save_dir, 'config.json')
        model_path = os.path.join(args.save_dir,
                                  str(args.encoder) + '_' + str(args.target_disease) + '_' + str(args.seed) + '.pt')
        # log_path = os.path.join(args.save_dir, 'log.csv')
        # export_config(args, config_path)
        check_path(model_path)
        # with open(log_path, 'w') as fout:
        #     fout.write('step,train_auc,dev_auc,test_auc\n')
        initial_d = 0
        blk_emb = np.load(args.blk_emb_path)
        blk_pad_id = len(blk_emb) - 1
        if args.target_disease == 'Heart_failure':
            code2id = pickle.load(open('./data/hf/hf_code2idx_new.pickle', 'rb'))
            pad_id = len(code2id)
            data_path = './data/hf/hf'
            emb_path = './data/processed/heart_failure.npy'
        elif args.target_disease == 'COPD':
            code2id = pickle.load(open('./data/copd/copd_code2idx_new.pickle', 'rb'))
            pad_id = len(code2id)
            data_path = './data/copd/copd'
            emb_path = './data/processed/COPD.npy'
        elif args.target_disease == 'Kidney':
            code2id = pickle.load(open('./data/kidney/kidney_code2idx_new.pickle', 'rb'))
            pad_id = len(code2id)
            data_path = './data/kidney/kidney'
            emb_path = './data/processed/kidney_disease.npy'
        elif args.target_disease == 'Dementia':
            code2id = pickle.load(open('./data/dementia/dementia_code2idx_new.pickle', 'rb'))
            pad_id = len(code2id)
            data_path = './data/dementia/dementia'
            emb_path = './data/processed/dementia.npy'
        elif args.target_disease == 'Amnesia':
            code2id = pickle.load(open('./data/amnesia/amnesia_code2idx_new.pickle', 'rb'))
            pad_id = len(code2id)
            data_path = './data/amnesia/amnesia'
            emb_path = './data/processed/amnesia.npy'
        elif args.target_disease == 'mimic':
            pad_id = 4894
            data_path = './data/mimic/mimic'
        elif args.target_disease == 'ARF':
            pad_id = 5132
            initial_d = 5132
            data_path = './data/ARF/ARF'
        elif args.target_disease == 'mortality':
            pad_id = 7727
            initial_d = 7727
            data_path = './data/mortality/mortality'
        elif args.target_disease == 'Shock':
            pad_id = 5795
            initial_d = 5795
            data_path = './data/Shock/Shock'
        else:
            raise ValueError('Invalid disease')
        device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
        if args.target_disease == 'ARF' or args.target_disease == 'mortality' or args.target_disease == 'Shock':
            train_dataset = MyDataset3(data_path + '_training_new.npz',
                                      args.max_len, args.max_num_codes, args.max_num_blks, pad_id, device)
            dev_dataset = MyDataset3(data_path + '_validation_new.npz', args.max_len,
                                    args.max_num_codes, args.max_num_blks, pad_id, device)
            test_dataset = MyDataset3(data_path + '_testing_new.npz', args.max_len,
                                     args.max_num_codes, args.max_num_blks, pad_id, device)
            train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=collate_fn3)
            dev_dataloader = DataLoader(dev_dataset, args.batch_size, shuffle=False, collate_fn=collate_fn3)
            test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=collate_fn3)
        elif args.target_disease == 'mimic':
            train_dataset = MyDataset2(data_path + '_train.pickle',
                                      args.max_len, args.max_num_codes, args.max_num_blks, pad_id, device)
            dev_dataset = MyDataset2(data_path + '_val.pickle',
                                    args.max_len,
                                    args.max_num_codes, args.max_num_blks, pad_id, device)
            test_dataset = MyDataset2(data_path + '_test.pickle', args.max_len,
                                     args.max_num_codes, args.max_num_blks, pad_id, device)
            train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=collate_fn)
            dev_dataloader = DataLoader(dev_dataset, args.batch_size, shuffle=False, collate_fn=collate_fn)
            test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=collate_fn)
        else:

            train_dataset = MyDataset(data_path + '_training_new.pickle', data_path + '_training_txt.pickle',
                                      args.max_len, args.max_num_codes, args.max_num_blks, pad_id, blk_pad_id, device)
            dev_dataset = MyDataset(data_path + '_validation_new.pickle', data_path + '_validation_txt.pickle', args.max_len,
                                    args.max_num_codes, args.max_num_blks, pad_id, blk_pad_id, device)
            test_dataset = MyDataset(data_path + '_testing_new.pickle', data_path + '_testing_txt.pickle', args.max_len,
                                 args.max_num_codes, args.max_num_blks, pad_id, blk_pad_id, device)
            train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=collate_fn)
            dev_dataloader = DataLoader(dev_dataset, args.batch_size, shuffle=False, collate_fn=collate_fn)
            test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=collate_fn)

        if args.encoder == 'hita':
            model = HitaNet(pad_id, args.d_model, args.dropout, args.dropout_emb, args.num_layers, args.num_heads,
                                     args.max_len)
        elif args.encoder == 'lstm':
            model = LSTM_encoder(pad_id, args.d_model, args.dropout, args.dropout_emb, args.num_layers, args.num_heads,
                                     args.max_len)
        elif args.encoder == 'lsan':
            model = LSAN(pad_id, args.d_model, args.dropout, args.dropout_emb, args.num_layers, args.num_heads,
                                     args.max_len)
        elif args.encoder == 'gruself':
            model = GRUSelf(pad_id, args.d_model, args.dropout, args.dropout_emb, args.num_layers, args.num_heads,
                                args.max_len)
        elif args.encoder == 'timeline':
            model = Timeline(pad_id, args.d_model, args.dropout, args.dropout_emb, args.num_layers, args.num_heads,
                                args.max_len)
        elif args.encoder == 'sand':
            model = SAND(pad_id, args.d_model, args.dropout, args.dropout_emb, args.num_layers, args.num_heads,
                                args.max_len)
        elif args.encoder == 'retain':
            model = Retain(pad_id, args.d_model, args.dropout, args.dropout_emb, args.num_layers, args.num_heads,
                                args.max_len)
        elif args.encoder == 'retainex':
            model = RetainEx(pad_id, args.d_model, args.dropout, args.dropout_emb, args.num_layers, args.num_heads,
                                args.max_len)
        elif args.encoder == 'LeapLSTM':
            model = LeapLSTM(pad_id, args.d_model, args.dropout, args.dropout_emb, args.num_layers, args.num_heads,
                                args.max_len)
        elif args.encoder == 'skimrnn':
            model = SkimRNN(pad_id, args.d_model, args.dropout, args.dropout_emb, args.num_layers, args.num_heads,
                                args.max_len, device)
        elif args.encoder == 'TLSTM':
            model = TLSTM(pad_id, args.d_model, args.dropout, args.dropout_emb, args.num_layers, args.num_heads,
                                args.max_len)
        elif args.encoder == 'skiprnn':
            model = SkipLSTM(pad_id, args.d_model, args.d_model, 1)
        else:
            raise ValueError('Invalid encoder')
        model.to(device)

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.learning_rate},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': args.learning_rate}
        ]
        optim = Adam(grouped_parameters)
        loss_func = nn.CrossEntropyLoss(reduction='mean')

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
        global_step, best_dev_epoch = 0, 0
        best_dev_auc, final_test_auc, total_loss = 0.0, 0.0, 0.0
        best_epoch_pr, best_epoch_f1, best_epoch_kappa = 0.0, 0.0, 0.0
        model.train()
        for epoch_id in range(args.n_epochs):
            print('epoch: {:5} '.format(epoch_id))
            model.train()
            start_time = time.time()
            for i, data in enumerate(train_dataloader):
                optim.zero_grad()
                labels, ehr, mask, txt, mask_txt, lengths, time_step, code_mask = data
                outputs = model(ehr, mask, lengths, time_step, code_mask)
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

            model.eval()
            train_acc, tr_precision, tr_recall, tr_f1, tr_roc_auc, tr_pr_auc, tr_kappa = eval_metric(train_dataloader, model)
            dev_acc, d_precision, d_recall, d_f1, d_roc_auc, d_pr_auc, d_kappa = eval_metric(dev_dataloader, model)
            test_acc, t_precision, t_recall, t_f1, t_roc_auc, t_pr_auc, t_kappa = eval_metric(test_dataloader, model)
            # train_acc, tr_precision, tr_recall, tr_f1, tr_roc_auc, tr_pr_auc, tr_kappa, tr_loss = eval_metric_arf(train_dataloader, model)
            # dev_acc, d_precision, d_recall, d_f1, d_roc_auc, d_pr_auc, d_kappa, d_loss  = eval_metric_arf(dev_dataloader, model)
            # test_acc, t_precision, t_recall, t_f1, t_roc_auc, t_pr_auc, t_kappa, t_loss  = eval_metric_arf(test_dataloader, model)
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

            if d_f1 >= best_dev_auc:
                best_dev_auc = d_f1
                final_test_auc = t_f1
                best_dev_epoch = epoch_id
                best_epoch_pr = t_pr_auc
                best_epoch_f1 = t_f1
                best_epoch_kappa = t_kappa
                # torch.save([model, args], model_path)
                # with open(log_path, 'a') as fout:
                #     fout.write('{},{},{},{}\n'.format(global_step, tr_pr_auc, d_pr_auc, t_pr_auc))
                print(f'model saved to {model_path}')
            if epoch_id - best_dev_epoch >= args.max_epochs_before_stop:

                break

        print()
        print('training ends in {} steps'.format(global_step))
        print('best dev auc: {:.4f} (at epoch {})'.format(best_dev_auc, best_dev_epoch))
        print('final test auc: {:.4f}'.format(final_test_auc))
        results_file = open(str(args.save_dir) + str(args.encoder) + '_' + str(args.target_disease) + '_' + str(args.seed) + '.csv', 'w', encoding='gbk')
        csv_w = csv.writer(results_file)
        csv_w.writerow([best_epoch_pr, best_epoch_f1, best_epoch_kappa])
        print('best test pr: {:.4f}'.format(best_epoch_pr))
        print('best test f1: {:.4f}'.format(best_epoch_f1))
        print('best test kappa: {:.4f}'.format(best_epoch_kappa))
        print()


def pred(args):
    model_path = os.path.join(args.save_dir, 'model.pt')
    model, old_args = torch.load(model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    model.to(device)
    model.eval()
    blk_emb = np.load(old_args.blk_emb_path)
    blk_pad_id = len(blk_emb) - 1
    if old_args.target_disease == 'Heart_failure':
        code2id = pickle.load(open('./data/hf/hf_code2idx_new.pickle', 'rb'))
        id2code = {int(v): k for k, v in code2id.items()}
        code2topic = pickle.load(open('./data/hf/hf_code2topic.pickle', 'rb'))
        pad_id = len(code2id)
        data_path = './data/hf/hf'
    elif old_args.target_disease == 'COPD':
        code2id = pickle.load(open('./data/copd/copd_code2idx_new.pickle', 'rb'))
        id2code = {int(v): k for k, v in code2id.items()}
        code2topic = pickle.load(open('./data/copd/copd_code2topic.pickle', 'rb'))
        pad_id = len(code2id)
        data_path = './data/copd/copd'
    elif old_args.target_disease == 'Kidney':
        code2id = pickle.load(open('./data/kidney/kidney_code2idx_new.pickle', 'rb'))
        id2code = {int(v): k for k, v in code2id.items()}
        code2topic = pickle.load(open('./data/kidney/kidney_code2topic.pickle', 'rb'))
        pad_id = len(code2id)
        data_path = './data/kidney/kidney'
    elif old_args.target_disease == 'Amnesia':
        code2id = pickle.load(open('./data/amnesia/amnesia_code2idx_new.pickle', 'rb'))
        id2code = {int(v): k for k, v in code2id.items()}
        code2topic = pickle.load(open('./data/amnesia/amnesia_code2topic.pickle', 'rb'))
        pad_id = len(code2id)
        data_path = './data/amnesia/amnesia'
    elif old_args.target_disease == 'Dementia':
        code2id = pickle.load(open('./data/dementia/dementia_code2idx_new.pickle', 'rb'))
        id2code = {int(v): k for k, v in code2id.items()}
        code2topic = pickle.load(open('./data/dementia/dementia_code2topic.pickle', 'rb'))
        pad_id = len(code2id)
        data_path = './data/dementia/dementia'
    else:
        raise ValueError('Invalid disease')
    dev_dataset = MyDataset(data_path + '_validation_new.pickle', data_path + '_validation_txt.pickle',
                            old_args.max_len, old_args.max_num_codes, old_args.max_num_blks, pad_id, blk_pad_id, device)
    test_dataset = MyDataset(data_path + '_testing_new.pickle', data_path + '_testing_txt.pickle', old_args.max_len,
                             old_args.max_num_codes, old_args.max_num_blks, pad_id, blk_pad_id, device)
    dev_dataloader = DataLoader(dev_dataset, args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=collate_fn)
    # dev_acc, d_precision, d_recall, d_f1, d_roc_auc, d_pr_auc = eval_metric(dev_dataloader, model)
    test_acc, t_precision, t_recall, t_f1, t_roc_auc, t_pr_auc, t_kappa = eval_metric(test_dataloader, model)
    log_path = os.path.join(args.save_dir, 'result.csv')
    with open(log_path, 'w') as fout:
        fout.write('test_auc,test_f1,test_pre,test_recall,test_pr_auc,test_kappa\n')
        fout.write(
            '{},{},{},{},{},{}\n'.format(t_roc_auc, t_f1, t_precision, t_recall, t_pr_auc, t_kappa))
    with torch.no_grad():
        y_true = np.array([])
        y_pred = np.array([])
        y_score = np.array([])
        for i, data in enumerate(test_dataloader):
            labels, ehr, mask, txt, mask_txt, lengths, time_step, code_mask = data
            logits = model(ehr, mask, lengths, time_step, code_mask)
            scores = torch.softmax(logits, dim=-1)
            scores = scores.data.cpu().numpy()
            labels = labels.data.cpu().numpy()
            score = scores[:, 1]
            pred = scores.argmax(1)
            y_true = np.concatenate((y_true, labels))
            y_pred = np.concatenate((y_pred, pred))
            y_score = np.concatenate((y_score, score))
        with open(os.path.join(args.save_dir, 'prediction.csv'), 'w') as fout2:
            fout2.write('prediciton,score,label\n')
            for i in range(len(y_true)):
                fout2.write('{},{},{}\n'.format(y_pred[i], y_score[i], y_true[i]))


if __name__ == '__main__':
    model_name = ['lstm', 'gruself', 'retain', 'retainex', 'TLSTM', 'hita', 'sand', 'timeline']
    seeds = [1234, 2345, 3456, 4567, 5678, 6789, 7890, 8901, 9012, 6666, 7777]
    dataset = ["Kidney", "Amnesia", "mimic"]
    max_lens = [50,50,15]
    max_nums = [20,20,20]
    for seed in seeds:
        for name in model_name:
            for data, max_len, max_num in zip(dataset, max_lens, max_nums):

                main(name, seed, data, max_len, max_num)