import argparse
import os
import time
import random
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score
from torch.optim import *
from sklearn.metrics import precision_recall_curve, auc
from models.dataset import *
from models.arf_shock_mort_baselines import *
from utils.utils import check_path, export_config, bool_flag
import csv
import os
from datetime import datetime
from models.adacare import AdaCare


def eval_metric(eval_set, model, encoder):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        y_true = np.array([])
        y_pred = np.array([])
        y_score = np.array([])
        for i, data in enumerate(eval_set):
            labels, diag, drug, lab, proc, demo = data
            logits = model(diag, drug, lab, proc, demo)
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
    parser.add_argument('--d_model', default=256, type=int, help='dimension of hidden layers')
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
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=0.001, type=float)
    parser.add_argument('--target_rate', default=0.3, type=float)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='max grad norm (0 to disable)')
    parser.add_argument('--warmup_steps', default=200, type=int)
    parser.add_argument('--n_epochs', default=30, type=int)
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
    elif args.mode == 'gen':
        gen(args)
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
    else:
        raise ValueError('Invalid disease')
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

    if args.short_ICD:

        train_dataset = pancreas_Gendataset_multimodal(data_path + 'train_3dig' + str(args.target_disease)+ '_' +str(args.task) + '.csv', args.max_len, args.max_num_codes, pad_id_list, nan_id_list, args.task, device)
        val_dataset = pancreas_Gendataset_multimodal(data_path + 'val_3dig' + str(args.target_disease)+ '_' +str(args.task) + '.csv', args.max_len, args.max_num_codes, pad_id_list, nan_id_list, args.task, device)
        test_dataset = pancreas_Gendataset_multimodal(data_path + 'test_3dig' + str(args.target_disease)+ '_' +str(args.task) + '.csv', args.max_len, args.max_num_codes, pad_id_list, nan_id_list, args.task, device)
        if args.model != 'none':
            synthetic_dataset = pancreas_Gendataset_multimodal_for_baselines(data_path + str(args.model) + '_synthetic_3dig' + str(args.target_disease) + '_' + str(args.task) + '.csv', args.max_len, args.max_num_codes, pad_id_list, nan_id_list, args.task, device)
    else:
        train_dataset = pancreas_Gendataset_multimodal(data_path + 'train_5dig' + str(args.target_disease)+ '_' +str(args.task) + '.csv', args.max_len, args.max_num_codes, pad_id_list, nan_id_list, args.task, device)
        val_dataset = pancreas_Gendataset_multimodal(data_path + 'val_5dig' + str(args.target_disease)+ '_' +str(args.task) + '.csv', args.max_len, args.max_num_codes, pad_id_list, nan_id_list, args.task, device)
        test_dataset = pancreas_Gendataset_multimodal(data_path + 'test_5dig' + str(args.target_disease)+ '_' +str(args.task) + '.csv', args.max_len, args.max_num_codes, pad_id_list, nan_id_list, args.task, device)
        if args.model != 'none':
            synthetic_dataset = pancreas_Gendataset_multimodal_for_baselines(data_path + str(args.model) + '_synthetic_5dig' + str(args.target_disease) + '_' + str(args.task) + '.csv', args.max_len, args.max_num_codes, pad_id_list, nan_id_list, args.task, device)

    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=gen_collate_fn_multimodal)
    dev_dataloader = DataLoader(val_dataset, args.batch_size, shuffle=False, collate_fn=gen_collate_fn_multimodal)
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=gen_collate_fn_multimodal)
    if args.model != 'none':
        synthetic_dataloader = DataLoader(synthetic_dataset, args.batch_size, shuffle=True, collate_fn=gen_collate_fn_multimodal)

    if args.name == 'F-LSTM':
        model = F_LSTM(pad_id_list, demo_len, args.d_model, args.dropout, args.max_len)
    elif args.name == 'F-CNN':
        model = F_CNN(pad_id_list, demo_len, args.d_model, args.dropout, args.max_len)
    elif args.name == 'Raim':
        model = Raim(pad_id_list, demo_len, args.d_model, args.dropout, args.max_len)
    elif args.name == 'DCMN':
        model = DCMN(pad_id_list, demo_len, args.d_model, args.dropout, args.max_len)
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
    best_kappa, best_f1 = 0, 0
    model.train()

    if args.model != 'none':
        for epoch in range(3):
            for i, data in enumerate(tqdm(synthetic_dataloader, desc="Training with synthetic data")):
                labels, diag, drug, lab, proc, demo = data
                optim.zero_grad()
                outputs = model(diag, drug, lab, proc, demo)
                synthetic_loss = loss_func(outputs, labels)
                synthetic_loss.backward()


    for epoch_id in range(args.n_epochs):
        print('epoch: {:5} '.format(epoch_id))
        model.train()
        start_time = time.time()


        for i, data in enumerate(tqdm(train_dataloader, desc="Training with real data")):
            labels, diag, drug, lab, proc, demo = data
            optim.zero_grad()
            outputs = model(diag, drug, lab, proc, demo)
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

        if d_f1 >= best_dev_auc:
            best_dev_auc = d_f1
            final_test_auc = t_pr_auc
            best_dev_epoch = epoch_id
            best_f1 = t_f1
            best_kappa = t_kappa
            print('Getting better performance on dev set at epoch {}'.format(epoch_id))
        if epoch_id - best_dev_epoch >= args.max_epochs_before_stop:
            break

    results_file = open(results_file_name, 'w',
        encoding='gbk')
    csv_w = csv.writer(results_file)
    csv_w.writerow([final_test_auc, best_f1, best_kappa])
    print()
    print('training ends in {} steps'.format(global_step))
    print('best dev auc: {:.4f} (at epoch {})'.format(best_dev_auc, best_dev_epoch))
    print('final test auc: {:.4f}'.format(final_test_auc))
    print(final_test_auc, best_f1, best_kappa)
    print()
    results_file.close()

def gen(args):
    pass

# def pred(args):
#     model_path = os.path.join(args.save_dir, 'model.pt')
#     model, old_args = torch.load(model_path)
#     device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
#     model.to(device)
#     model.eval()
#     blk_emb = np.load(old_args.blk_emb_path)
#     blk_pad_id = len(blk_emb) - 1
#     if old_args.target_disease == 'Heart_failure':
#         code2id = pickle.load(open('./data/hf/hf_code2idx.pickle', 'rb'))
#         id2code = {int(v): k for k, v in code2id.items()}
#         code2topic = pickle.load(open('./data/hf/hf_code2topic.pickle', 'rb'))
#         pad_id = len(code2id)
#         data_path = './data/hf/hf'
#     elif old_args.target_disease == 'COPD':
#         code2id = pickle.load(open('./data/copd/copd_code2idx.pickle', 'rb'))
#         id2code = {int(v): k for k, v in code2id.items()}
#         code2topic = pickle.load(open('./data/copd/copd_code2topic.pickle', 'rb'))
#         pad_id = len(code2id)
#         data_path = './data/copd/copd'
#     elif old_args.target_disease == 'Kidney':
#         code2id = pickle.load(open('./data/kidney/kidney_code2idx.pickle', 'rb'))
#         id2code = {int(v): k for k, v in code2id.items()}
#         code2topic = pickle.load(open('./data/kidney/kidney_code2topic.pickle', 'rb'))
#         pad_id = len(code2id)
#         data_path = './data/kidney/kidney'
#     elif old_args.target_disease == 'Amnesia':
#         code2id = pickle.load(open('./data/amnesia/amnesia_code2idx.pickle', 'rb'))
#         id2code = {int(v): k for k, v in code2id.items()}
#         code2topic = pickle.load(open('./data/amnesia/amnesia_code2topic.pickle', 'rb'))
#         pad_id = len(code2id)
#         data_path = './data/amnesia/amnesia'
#     elif old_args.target_disease == 'Dementia':
#         code2id = pickle.load(open('./data/dementia/dementia_code2idx.pickle', 'rb'))
#         id2code = {int(v): k for k, v in code2id.items()}
#         code2topic = pickle.load(open('./data/dementia/dementia_code2topic.pickle', 'rb'))
#         pad_id = len(code2id)
#         data_path = './data/dementia/dementia'
#     else:
#         raise ValueError('Invalid disease')
#     dev_dataset = MyDataset(data_path + '_validation_sps.pickle', data_path + '_validation_txt.pickle',
#                             old_args.max_len, old_args.max_num_codes, old_args.max_num_blks, pad_id, blk_pad_id, device)
#     test_dataset = MyDataset(data_path + '_testing_sps.pickle', data_path + '_testing_txt.pickle', old_args.max_len,
#                              old_args.max_num_codes, old_args.max_num_blks, pad_id, blk_pad_id, device)
#     dev_dataloader = DataLoader(dev_dataset, args.batch_size, shuffle=False, collate_fn=collate_fn)
#     test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=collate_fn)
#     # dev_acc, d_precision, d_recall, d_f1, d_roc_auc, d_pr_auc = eval_metric(dev_dataloader, model)
#     test_acc, t_precision, t_recall, t_f1, t_roc_auc, t_pr_auc, t_kappa = eval_metric(test_dataloader, model)
#     log_path = os.path.join(args.save_dir, 'result.csv')
#     with open(log_path, 'w') as fout:
#         fout.write('test_auc,test_f1,test_pre,test_recall,test_pr_auc,test_kappa\n')
#         fout.write(
#             '{},{},{},{},{},{}\n'.format(t_roc_auc, t_f1, t_precision, t_recall, t_pr_auc, t_kappa))
#     with torch.no_grad():
#         y_true = np.array([])
#         y_pred = np.array([])
#         y_score = np.array([])
#         for i, data in enumerate(test_dataloader):
#             labels, ehr, mask, txt, mask_txt, lengths, time_step, code_mask = data
#             logits = model(ehr, mask, lengths, time_step, code_mask)
#             scores = torch.softmax(logits, dim=-1)
#             scores = scores.data.cpu().numpy()
#             labels = labels.data.cpu().numpy()
#             score = scores[:, 1]
#             pred = scores.argmax(1)
#             y_true = np.concatenate((y_true, labels))
#             y_pred = np.concatenate((y_pred, pred))
#             y_score = np.concatenate((y_score, score))
#         with open(os.path.join(args.save_dir, 'prediction.csv'), 'w') as fout2:
#             fout2.write('prediciton,score,label\n')
#             for i in range(len(y_true)):
#                 fout2.write('{},{},{}\n'.format(y_pred[i], y_score[i], y_true[i]))


if __name__ == '__main__':

    modes = ['train']
    short_ICD = True
    toy = False
    seeds = [41,42,43]
    save_path = './saved_'
    baseline_names = ['F-LSTM', 'F-CNN', 'Raim', 'DCMN']
    # baseline_names = ['F-LSTM']
    # model_names = ['MedDiffGa', 'LSTM-MLP', 'LSTM-medGAN', 'synTEG', 'TWIN', 'none']
    model_names = ['LSTM-Meddiff', 'LSTM-ScoEHR']
    save_dirs = [save_path+name+'/' for name in model_names]
    datas = ['mimic']
    tasks = ['shock', 'arf']
    # tasks = ['mortality']
    max_lens = [50]
    max_nums = [20]
    for mode in modes:
        for seed in seeds:
            for baseline_name in baseline_names:
                for model_name, save_dir in zip(model_names, save_dirs):
                    for data, max_len, max_num in zip(datas, max_lens, max_nums):
                        for task in tasks:
                                main(seed, baseline_name, model_name, data, task, max_len, max_num, save_dir, mode, short_ICD, toy)