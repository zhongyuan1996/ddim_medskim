import argparse
import os
import time
import csv
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    precision_recall_curve, auc, cohen_kappa_score
from torch.optim import Adam
from tqdm import tqdm
from models.dataset import *
from models.baseline import LSTM_encoder, HitaNet
from models.toy import *
# from models.medskim_with_diff import *
from utils.utils import check_path, export_config, bool_flag
from utils.icd_rel import *
import random
def LSTM_eval_metric(eval_set, model):
    model.eval()
    with torch.no_grad():
        y_true = np.array([])
        y_pred = np.array([])
        y_score = np.array([])
        for i, data in enumerate(eval_set):
            ehr, time_step, labels, code_mask, lengths = data
            logits = model(ehr, None, lengths, time_step, code_mask)
            scores = torch.softmax(logits, dim=-1)
            scores = scores.data.cpu().numpy()
            labels = labels.data.cpu().numpy()
            labels = labels.argmax(1)
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
def eval_metric(eval_set, model):
    model.eval()
    with torch.no_grad():
        y_true = np.array([])
        y_pred = np.array([])
        y_score = np.array([])
        for i, data in enumerate(eval_set):
            # labels, ehr, mask, txt, _, lengths, time_step, code_mask = data
            # logits, _ = model(ehr, lengths, time_step, code_mask)
            ehr, time_step, code_timegaps, labels, code_mask, lengths, visit_timegaps = data
            logits = model(ehr, None, lengths, time_step, code_mask, code_timegaps, visit_timegaps)
            scores = torch.softmax(logits, dim=-1)
            scores = scores.data.cpu().numpy()
            labels = labels.data.cpu().numpy()
            labels = labels.argmax(1)
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

def eval_metric_timegap(eval_set, model):
    model.eval()
    with torch.no_grad():
        y_true = np.array([])
        y_pred = np.array([])
        y_score = np.array([])
        for i, data in enumerate(eval_set):
            # labels, ehr, mask, txt, _, lengths, time_step, code_mask = data
            # logits, _ = model(ehr, lengths, time_step, code_mask)
            ehr, time_step, code_timegaps, labels, code_mask, lengths, visit_timegaps = data
            logits, _ = model(ehr, None, lengths, time_step, code_mask, code_timegaps, visit_timegaps)
            scores = torch.softmax(logits, dim=-1)
            scores = scores.data.cpu().numpy()
            labels = labels.data.cpu().numpy()
            labels = labels.argmax(1)
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


def main(seed, name, data, max_len, max_num):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=True, type=bool_flag, nargs='?', const=True, help='use GPU')
    parser.add_argument('--seed', default=seed, type=int, help='seed')
    parser.add_argument('-bs', '--batch_size', default=64, type=int)
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
    parser.add_argument('--lamda', default=0.1, type=float)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='max grad norm (0 to disable)')
    parser.add_argument('--warmup_steps', default=200, type=int)
    parser.add_argument('--n_epochs', default=30, type=int)
    parser.add_argument('--log_interval', default=20, type=int)
    parser.add_argument('--mode', default='train', choices=['train', 'pred', 'study'], help='run training or evaluation')
    parser.add_argument('--model', default=name)
    parser.add_argument('--save_dir', default='./saved_models/', help='models output directory')
    parser.add_argument('--lambda_timegap', default=0.5, type=float)
    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
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
    if str(args.model) + '_' + str(args.target_disease) + '_' + str(args.seed) + '.csv' in files:
        print("conducted_experiments")
    else:
        config_path = os.path.join(args.save_dir, 'config.json')
        model_path = os.path.join(args.save_dir, 'models.pt')
        log_path = os.path.join(args.save_dir, 'log.csv')
        # export_config(args, config_path)
        check_path(model_path)
        # with open(log_path, 'w') as fout:
        #     fout.write('step,train_auc,dev_auc,test_auc\n')

        if args.target_disease == 'Heartfailure':
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
        else:
            raise ValueError('Invalid disease')
        device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

        if args.model in ('LSTM', 'Hita'):
            train_dataset = MyDataset(data_path + '_training_new.pickle',
                                      args.max_len, args.max_num_codes, pad_id, device)
            dev_dataset = MyDataset(data_path + '_validation_new.pickle', args.max_len,
                                    args.max_num_codes, pad_id, device)
            test_dataset = MyDataset(data_path + '_testing_new.pickle',  args.max_len,
                                     args.max_num_codes, pad_id, device)
            train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=collate_fn)
            dev_dataloader = DataLoader(dev_dataset, args.batch_size, shuffle=False, collate_fn=collate_fn)
            test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=collate_fn)
            if args.model == 'LSTM':
                model = LSTM_encoder(pad_id,args.d_model, args.dropout, args.dropout_emb, args.num_layers, None, None)
            elif args.model == 'Hita':
                model = HitaNet(pad_id, args.d_model, args.dropout, args.dropout_emb, args.num_layers, args.num_heads, args.max_len)
        elif args.model in ('toy', 'Hitatime', 'LSTMtimegap'):
            train_dataset = MyDataset_timegap(data_path + '_training_with_timegaps.pickle',
                                        args.max_len, args.max_num_codes, pad_id, device)
            dev_dataset = MyDataset_timegap(data_path + '_validation_with_timegaps.pickle', args.max_len,
                                    args.max_num_codes, pad_id, device)
            test_dataset = MyDataset_timegap(data_path + '_testing_with_timegaps.pickle', args.max_len,
                                        args.max_num_codes, pad_id, device)
            train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=collate_fn_timegap)
            dev_dataloader = DataLoader(dev_dataset, args.batch_size, shuffle=False, collate_fn=collate_fn_timegap)
            test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=collate_fn_timegap)
            if args.model == 'toy':
                model = LSTM_with_time(pad_id, args.d_model, args.dropout, args.dropout_emb, args.num_layers, None, None)
            elif args.model == 'Hitatime':
                model = HitaNet_time_diff(pad_id, args.d_model, args.dropout, args.dropout_emb, args.num_layers, args.num_heads, args.max_len)
            elif args.model == 'LSTMtimegap':
                model = LSTM_timegap(pad_id, args.d_model, args.dropout, args.dropout_emb, args.num_layers, None, None)
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
        loss_func = nn.CrossEntropyLoss(reduction='mean')
        mse = nn.MSELoss(reduction='mean')

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
                if args.model in ('toy', 'Hitatime'):
                    ehr, time_step, code_timegaps, labels, code_mask, lengths, visit_timegaps = data
                    outputs = model(ehr, None, lengths, time_step, code_mask, code_timegaps, visit_timegaps)
                elif args.model in ('LSTM', 'Hita'):
                    ehr, time_step, labels, code_mask, lengths = data
                    outputs= model(ehr, None, lengths, time_step, code_mask)
                elif args.model in ('LSTMtimegap'):
                    ehr, time_step, code_timegaps, labels, code_mask, lengths, visit_timegaps = data
                    outputs, timegap = model(ehr, None, lengths, time_step, code_mask, code_timegaps, visit_timegaps)
                else:
                    raise ValueError('Invalid model')

                if args.model in ('toy', 'Hitatime', 'LSTM', 'Hita'):
                    loss = loss_func(outputs, labels)
                elif args.model in ('LSTMtimegap'):
                    mask = (visit_timegaps != 10000)
                    filtered_timegap = timegap[mask]
                    filtered_visit_timegaps = visit_timegaps[mask]
                    loss = loss_func(outputs, labels) + args.lambda_timegap * mse(filtered_timegap, filtered_visit_timegaps.float())
                else:
                    raise ValueError('Invalid model')
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
            if args.model in ('toy', 'Hitatime'):
                train_acc, tr_precision, tr_recall, tr_f1, tr_roc_auc, tr_pr_auc, tr_kappa = eval_metric(train_dataloader,
                                                                                                         model)
                dev_acc, d_precision, d_recall, d_f1, d_roc_auc, d_pr_auc, d_kappa = eval_metric(dev_dataloader, model)
                test_acc, t_precision, t_recall, t_f1, t_roc_auc, t_pr_auc, t_kappa = eval_metric(test_dataloader, model)
            elif args.model in ('LSTM', 'Hita'):
                train_acc, tr_precision, tr_recall, tr_f1, tr_roc_auc, tr_pr_auc, tr_kappa = LSTM_eval_metric(
                    train_dataloader,
                    model)
                dev_acc, d_precision, d_recall, d_f1, d_roc_auc, d_pr_auc, d_kappa = LSTM_eval_metric(dev_dataloader, model)
                test_acc, t_precision, t_recall, t_f1, t_roc_auc, t_pr_auc, t_kappa = LSTM_eval_metric(test_dataloader, model)
            elif args.model in ('LSTMtimegap'):
                train_acc, tr_precision, tr_recall, tr_f1, tr_roc_auc, tr_pr_auc, tr_kappa = eval_metric_timegap(
                    train_dataloader,
                    model)
                dev_acc, d_precision, d_recall, d_f1, d_roc_auc, d_pr_auc, d_kappa = eval_metric_timegap(dev_dataloader, model)
                test_acc, t_precision, t_recall, t_f1, t_roc_auc, t_pr_auc, t_kappa = eval_metric_timegap(test_dataloader, model)
            else:
                raise ValueError('Invalid model')
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
        results_file = open(
            str(args.save_dir) + str(args.model) + '_' + str(args.target_disease) + '_' + str(args.seed) + '.csv',
            'w', encoding='gbk')
        csv_w = csv.writer(results_file)
        csv_w.writerow([best_epoch_pr, best_epoch_f1, best_epoch_kappa])
        print('best test pr: {:.4f}'.format(best_epoch_pr))
        print('best test f1: {:.4f}'.format(best_epoch_f1))
        print('best test kappa: {:.4f}'.format(best_epoch_kappa))
        print()

if __name__ == '__main__':

    seeds = [1,2,3,4,5,6,7,8,9,10]
    # seeds = [11]
    # names = ['toy','LSTM','Hitatime', 'Hita']
    names = ['LSTMtimegap']
    datas = ['Heartfailure', 'COPD', 'Kidney', 'Dementia', 'Amnesia', 'mimic']
    max_lens = [50, 50, 50, 50, 50, 50]
    max_nums = [20, 20, 20, 20, 20, 20]
    for seed in seeds:
        for name in names:
            for data, max_len, max_num in zip(datas, max_lens, max_nums):
                main(seed, name, data, max_len, max_num)

    csv_path = './saved_models/'

    # List all files with the specified pattern
    csv_files = [f for f in os.listdir(csv_path) if f.endswith('.csv')]

    # Container to hold results
    results = {}

    # Process each CSV file
    for csv_file in csv_files:
        # Extract model name, dataset, and seed from the file name
        parts = csv_file.split('_')
        model_name, dataset = parts[0], parts[1]
        seed = parts[2].split('.')[0]  # remove .csv from the end and get seed

        # Verify that the seed is a number
        if not seed.isdigit():
            continue

        # Read the CSV
        data = pd.read_csv(os.path.join(csv_path, csv_file), header=None).values

        # Store the results
        if (model_name, dataset) not in results:
            results[(model_name, dataset)] = []
        results[(model_name, dataset)].append(data)

    # Compute means
    mean_results = {}
    for key, values in results.items():
        mean_values = sum(values) / len(values)
        mean_results[key] = mean_values.mean(axis=0)  # mean across seeds for each metric

    # Print results
    for key, value in mean_results.items():
        print(f'Model: {key[0]}, Dataset: {key[1]}, PR Mean: {value[0]:.4f}, F1 Mean: {value[1]:.4f}, Kappa Mean: {value[2]:.4f}')