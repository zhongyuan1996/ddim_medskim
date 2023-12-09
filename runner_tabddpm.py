import argparse
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import time
import pandas as pd
import csv
import random
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    precision_recall_curve, auc, cohen_kappa_score, log_loss
from torch.optim import Adam, lr_scheduler
from models.dataset import *
from utils.utils import check_path, export_config, bool_flag
from models.tabddpm import *
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine
import yaml
from tqdm import tqdm
import nmslib
import torch.nn as nn
import warnings


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def rearrange_codes(codes, pad_id, max_num_codes):
    """Rearrange the codes to move meaningful ones to the front."""
    codes = [code for code in codes if code != pad_id]  # Filter out pad indices
    codes += [pad_id] * (max_num_codes - len(codes))  # Fill the rest with pad indices
    return codes





def eval_metrictabDDPM(eval_set, model, criterion, pad_id, args, device):
    model.eval()
    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for i, data in enumerate(tqdm(eval_set, desc="Evaluating", leave=False)):

            if args.target_disease in ['pancreas', 'mimic']:
                ehr, time_step = data

                multihot_ehr_label = torch.zeros((ehr.size(0), args.max_len, pad_id), dtype=torch.float32) + 1e-7
                multihot_ehr_label = multihot_ehr_label.to(device)
                for batch_idx in range(ehr.size(0)):
                    for seq_idx in range(args.max_len):
                        for label in range(pad_id):
                            if label != pad_id:
                                multihot_ehr_label[batch_idx, seq_idx, label] = 1.0
                _, gen_ehr = model(ehr, None)
            else:
                ehr, label, time_step = data

                multihot_ehr_label = torch.zeros((ehr.size(0), args.max_len, pad_id), dtype=torch.float32) + 1e-7
                multihot_ehr_label = multihot_ehr_label.to(device)
                for batch_idx in range(ehr.size(0)):
                    for seq_idx in range(args.max_len):
                        for label in range(pad_id):
                            if label != pad_id:
                                multihot_ehr_label[batch_idx, seq_idx, label] = 1.0
                _, gen_ehr = model(ehr, label)

            # Calculate loss
            loss = criterion(gen_ehr, multihot_ehr_label)

            running_loss += loss.item() * ehr.size(0)  # multiply by batch size
            total_samples += ehr.size(0)

        average_loss = running_loss / total_samples

    return average_loss


def eval_metric(eval_set, model):
    model.eval()
    with torch.no_grad():
        y_true = np.array([])
        y_pred = np.array([])
        y_score = np.array([])

        for i, data in enumerate(eval_set):
            ehr, _, labels, _, _ = data
            logit = model(ehr)
            scores = torch.softmax(logit, dim=-1)
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
        loss = log_loss(y_true, y_pred)
    return accuary, precision, recall, f1, roc_auc, pr_auc, kappa, loss


def main(name, seed, data, max_len, max_num, save_dir):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=True, type=bool_flag, nargs='?', const=True, help='use GPU')
    parser.add_argument('--seed', default=seed, type=int, help='seed')
    parser.add_argument('-bs', '--batch_size', default=16, type=int)
    parser.add_argument('--model', default=name)
    parser.add_argument('-me', '--max_epochs_before_stop', default=10, type=int)
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
    parser.add_argument('--target_disease', default=data,
                        choices=['Heart_failure', 'COPD', 'Kidney', 'Dementia', 'Amnesia', 'mimic', 'ARF', 'Shock',
                                 'mortality'])
    parser.add_argument('--target_att_heads', default=4, type=int, help='target disease attention heads number')
    parser.add_argument('--mem_size', default=15, type=int, help='memory size')
    parser.add_argument('--mem_update_size', default=15, type=int, help='memory update size')
    parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=0.001, type=float)
    parser.add_argument('--target_rate', default=0.3, type=float)
    parser.add_argument('--lamda', default=0.1, type=float)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='max grad norm (0 to disable)')
    parser.add_argument('--warmup_steps', default=200, type=int)
    parser.add_argument('--n_epochs', default=30, type=int)
    parser.add_argument('--log_interval', default=20, type=int)
    parser.add_argument('--mode', default='train', choices=['train', 'pred', 'study'],
                        help='run training or evaluation')
    # parser.add_argument('--model', default='Selected', choices=['Selected'])
    parser.add_argument('--save_dir', default=save_dir, help='models output directory')
    parser.add_argument("--config", type=str, default='ehr.yml', help="Path to the config file")
    parser.add_argument("--h_model", type=int, default=256, help="dimension of hidden state in LSTM")
    parser.add_argument("--lambda_DF_loss", type=float, default=0.1, help="scale of diffusion model loss")
    parser.add_argument("--lambda_CE_gen_loss", type=float, default=0.5, help="scale of generated sample loss")
    parser.add_argument("--lambda_KL_loss", type=float, default=0.01, help="scale of hidden state KL loss")
    parser.add_argument("--temperature", type=str, default='temperature',
                        help="temperature control of classifier softmax")
    parser.add_argument("--mintau", type=float, default=0.5, help="parameter mintau of temperature control")
    parser.add_argument("--maxtau", type=float, default=5.0, help="parameter maxtau of temperature control")
    parser.add_argument("--patience", type=int, default=5, help="learning rate patience")
    parser.add_argument("--factor", type=float, default=0.2, help="learning rate factor")

    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    # elif args.mode == 'gen':
    #     gen(args)
    # elif args.mode == 'study':
    #     study(args)
    else:
        raise ValueError('Invalid mode')


def train(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    files = os.listdir(args.save_dir)
    csv_filename = str(args.model) + '_' + str(args.target_disease) + '_' + str(args.seed) + '_' + '.csv'
    combinedData_filename = str(args.target_disease) + '_combinedData.pickle'
    if csv_filename in files:
        print('This experiment has been done!')
    else:
        initial_d = 0
        if args.target_disease == 'Heart_failure':
            code2id = pickle.load(open('data/hf/hf_code2idx_new.pickle', 'rb'))
            pad_id = len(code2id)
            data_path = './data/hf/hf'
            # emb_path = './data/processed/heart_failure.npy'
        elif args.target_disease == 'COPD':
            code2id = pickle.load(open('./data/copd/copd_code2idx_new.pickle', 'rb'))
            pad_id = len(code2id)
            data_path = './data/copd/copd'
            # emb_path = './data/processed/COPD.npy'
        elif args.target_disease == 'Kidney':
            code2id = pickle.load(open('./data/kidney/kidney_code2idx_new.pickle', 'rb'))
            pad_id = len(code2id)
            data_path = './data/kidney/kidney'
            # emb_path = './data/processed/kidney_disease.npy'
        elif args.target_disease == 'Dementia':
            code2id = pickle.load(open('./data/dementia/dementia_code2idx_new.pickle', 'rb'))
            pad_id = len(code2id)
            data_path = './data/dementia/dementia'
            # emb_path = './data/processed/dementia.npy'
        elif args.target_disease == 'Amnesia':
            code2id = pickle.load(open('./data/amnesia/amnesia_code2idx_new.pickle', 'rb'))
            pad_id = len(code2id)
            data_path = './data/amnesia/amnesia'
            # emb_path = './data/processed/amnesia.npy'
        elif args.target_disease == 'pancreas':
            code2id = pd.read_csv('./data/pancreas/code_to_int_mapping.csv', header=None)
            demo_len = 15
            pad_id = len(code2id)
            data_path = './data/pancreas/'
        elif args.target_disease == 'mimic':
            code2id = pd.read_csv('./data/mimic/code_to_int_mapping.csv', header=None)
            demo_len = 70
            pad_id = len(code2id)
            data_path = './data/mimic/'
        else:
            raise ValueError('Invalid disease')

        word2vec_filename = str(args.target_disease) + '_word2vec.pt'
        tabDDPM_filename = str(args.target_disease) + '_' + str(args.seed) + '_tabDDPM.pt'

        if not os.path.exists(str(save_dir)+word2vec_filename):
            if args.target_disease in ['pancreas', 'mimic']:
                with open(data_path + 'train_' + str(args.target_disease) + '.csv') as f:
                    train_set = pd.read_csv(f, index_col=False)
                    train_ehr = train_set['code_int'].apply(lambda x: ast.literal_eval(x)).tolist()
                with open(data_path + 'test_' + str(args.target_disease) + '.csv') as f:
                    test_set = pd.read_csv(f, index_col=False)
                    test_ehr = test_set['code_int'].apply(lambda x: ast.literal_eval(x)).tolist()
                with open(data_path + 'val_' + str(args.target_disease) + '.csv') as f:
                    val_set = pd.read_csv(f, index_col=False)
                    val_ehr = val_set['code_int'].apply(lambda x: ast.literal_eval(x)).tolist()
                all_set = train_ehr + test_ehr + val_ehr
                flattened_all_set = [visit for patient in all_set for visit in patient]
            else:
                with open(data_path + '_training_new.pickle', 'rb') as f:
                    train_set, _, _ = pickle.load(f)
                with open(data_path + '_testing_new.pickle', 'rb') as f:
                    test_set, _, _ = pickle.load(f)
                with open(data_path + '_validation_new.pickle', 'rb') as f:
                    val_set, _, _ = pickle.load(f)
                all_set = train_set + test_set + val_set
                flattened_all_set = [visit for patient in all_set for visit in patient]
            w2v = Word2Vec(flattened_all_set, vector_size=256, window=5, min_count=1, workers=4)
            w2v.save(str(args.save_dir) + word2vec_filename)
        else:
            print('word2vec model exists')

        if not os.path.exists(str(args.save_dir) + tabDDPM_filename):
            device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
            w2v = Word2Vec.load(str(args.save_dir) + word2vec_filename)
            all_word_vectors_matrix = w2v.wv.vectors

            index = nmslib.init(method='hnsw', space='cosinesimil')
            index.addDataPointBatch(all_word_vectors_matrix)
            index.createIndex({'post': 2}, print_progress=True)

            if args.target_disease in ['pancreas', 'mimic']:
                train_dataset = MyDataset4(data_path + 'train_' + str(args.target_disease) + '.csv',
                                              args.max_len, args.max_num_codes, pad_id, None, True, device)
                val_dataset = MyDataset4(data_path + 'val_' + str(args.target_disease) + '.csv',
                                            args.max_len, args.max_num_codes, pad_id, None, True, device)
                test_dataset = MyDataset4(data_path + 'test_' + str(args.target_disease) + '.csv',
                                             args.max_len, args.max_num_codes, pad_id, None, True, device)
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
                val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

            else:

                train_dataset = MyDataset(data_path + '_training_new.pickle',
                                          args.max_len, args.max_num_codes, pad_id, device)
                test_dataset = MyDataset(data_path + '_testing_new.pickle',
                                         args.max_len, args.max_num_codes, pad_id, device)
                val_dataset = MyDataset(data_path + '_validation_new.pickle',
                                        args.max_len, args.max_num_codes, pad_id, device)
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
                val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

            with open(os.path.join("configs/", args.config), "r") as f:
                config = yaml.safe_load(f)
            config = dict2namespace(config)

            model = tabDDPM(config, pad_id, 256, 64, args.dropout, args.max_num_codes, device)
            # model = DDPM(config, pad_id, None, None, args.dropout, args.max_len,args.max_num_codes, device)

            model.to(device)

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': args.weight_decay, 'lr': args.learning_rate},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0, 'lr': args.learning_rate}
            ]
            optimizer = Adam(grouped_parameters, lr=args.learning_rate)
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.factor,
                                                       patience=args.patience, verbose=True)

            # MSE_loss = nn.MSELoss()
            loss_fn = nn.BCEWithLogitsLoss(reduction='mean')

            print('Start training...Parameters')
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
            best_train_loss, best_test_loss, best_val_loss = 1e9, 1e9, 1e9

            time_steps = []

            for epoch_id in tqdm(range(args.n_epochs), desc="Epochs"):
                epoch_start_time = time.time()
                model.train()

                for i, data in enumerate(tqdm(train_loader, desc="Training", leave=False)):

                    if args.target_disease in ['pancreas', 'mimic']:
                        ehr, time_step = data

                        multihot_ehr_label = torch.zeros((ehr.size(0), args.max_len, pad_id), dtype=torch.float32) + 1e-7
                        multihot_ehr_label = multihot_ehr_label.to(device)
                        for batch_idx in range(ehr.size(0)):
                            for seq_idx in range(args.max_len):
                                for label in range(pad_id):
                                    if label != pad_id:
                                        multihot_ehr_label[batch_idx, seq_idx, label] = 1.0
                        _, gen_ehr = model(ehr, None)
                    else:
                        ehr, label, time_step = data

                        multihot_ehr_label = torch.zeros((ehr.size(0), args.max_len, pad_id), dtype=torch.float32) + 1e-7
                        multihot_ehr_label = multihot_ehr_label.to(device)
                        for batch_idx in range(ehr.size(0)):
                            for seq_idx in range(args.max_len):
                                for label in range(pad_id):
                                    if label != pad_id:
                                        multihot_ehr_label[batch_idx, seq_idx, label] = 1.0
                        _, gen_ehr = model(ehr, label)

                    loss =loss_fn(gen_ehr, multihot_ehr_label)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    global_step += 1

                print('Epoch {:3d} | time: {:5.2f}s | loss {:5.4f}'.format(epoch_id, (time.time() - epoch_start_time),
                                                                           loss.item()))
                model.eval()

                print('-' * 71)
                # print('Evaluating on training set...')
                # train_loss = eval_metrictabDDPM(train_loader, model)
                # print('Evaluating on testing set...')
                # test_loss = eval_metrictabDDPM(test_loader, model)
                print('Evaluating on validation set...')
                dev_loss = eval_metrictabDDPM(val_loader, model, loss_fn, pad_id, args, device)
                print('-' * 71)
                scheduler.step(dev_loss)
                # print(
                #     f'Epoch {epoch_id} | time: {(time.time() - epoch_start_time):5.2f}s | train loss {train_loss:5.4f} | test loss {test_loss:5.4f} | val loss {dev_loss:5.4f}'
                # )
                print(
                    f'Epoch {epoch_id} | time: {(time.time() - epoch_start_time):5.2f}s | val loss {dev_loss:5.4f}'
                )
                if dev_loss < best_val_loss:
                    # best_train_loss, best_test_loss, best_val_loss = train_loss, test_loss, dev_loss
                    best_val_loss = dev_loss
                    best_dev_epoch = epoch_id
                    torch.save(model.state_dict(), str(args.save_dir) + tabDDPM_filename)
                    print('Saving model (epoch {})'.format(epoch_id + 1))
                    print('-' * 71)
                if epoch_id - best_dev_epoch > args.max_epochs_before_stop:
                    print('Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch_id, best_val_loss))
                    break
        if not os.path.exists(str(args.save_dir) + combinedData_filename):
            print('Generating synthetic data...')
            with torch.no_grad():
                w2v = Word2Vec.load(str(args.save_dir) + word2vec_filename)
                all_word_vectors_matrix = w2v.wv.vectors

                index = nmslib.init(method='hnsw', space='cosinesimil')
                index.addDataPointBatch(all_word_vectors_matrix)
                index.createIndex({'post': 2}, print_progress=True)
                device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

                with open(os.path.join("configs/", args.config), "r") as f:
                    config = yaml.safe_load(f)
                config = dict2namespace(config)

                model = tabDDPM(config, pad_id, 256, 64, args.dropout, args.max_num_codes, device)
                model.load_state_dict(torch.load(str(args.save_dir) + tabDDPM_filename))
                model.to(device)
                model.eval()

                if args.target_disease in ['pancreas', 'mimic']:
                    train_dataset = MyDataset4(data_path + 'toy_' + str(args.target_disease) + '.csv',
                                               args.max_len, args.max_num_codes, pad_id, None, True, device)
                    train_loader = DataLoader(train_dataset, batch_size=args.batch_size*4, shuffle=True, drop_last=True)
                else:
                    train_dataset = MyDataset(data_path + '_training_new.pickle',
                                          args.max_len, args.max_num_codes, pad_id, device)
                    train_loader = DataLoader(train_dataset, batch_size=args.batch_size*4, shuffle=True, drop_last=True)

                synthetic_data = []
                real_data = []
                time_steps = []
                labels = []

                lpl_list = []

                for i, data in enumerate(tqdm(train_loader, desc="Generating synthetic data", leave=False)):
                    if args.target_disease in ['pancreas', 'mimic']:
                        ehr, time_step = data
                    else:
                        ehr, label, time_step = data

                    multihot_ehr_label = torch.zeros((ehr.size(0), args.max_len, pad_id), dtype=torch.float32) + 1e-7
                    multihot_ehr_label = multihot_ehr_label.to(device)
                    for batch_idx in range(ehr.size(0)):
                        for seq_idx in range(args.max_len):
                            for label in range(pad_id):
                                if label != pad_id:
                                    multihot_ehr_label[batch_idx, seq_idx, label] = 1.0

                    if args.target_disease in ['pancreas', 'mimic']:
                        _, gen_ehr = model(ehr, None)
                    else:
                        _, gen_ehr = model(ehr, label)

                    time_step = time_step.cpu().numpy().tolist()

                    # Calculate lpl per visit within this loop
                    for j in range(len(gen_ehr)):
                        logits_visit = gen_ehr[j, :, :]
                        target_visit = multihot_ehr_label[j, :, :]
                        prob = logits_visit.softmax(dim=-1)[target_visit == 1]
                        nll = -torch.log(prob + 1e-10)
                        ppl = nll.exp()
                        if torch.isnan(ppl).any():
                            warnings.warn('NaN perplexity detected during lpl calculation')
                        lpl_list.append(ppl)

                    # # Stack or extend the data as per your original logic
                    # synthetic_data.stack(gen_ehr) if args.target_disease in ['pancreas',
                    #                                                          'mimic'] else synthetic_data.extend(
                    #     gen_ehr)
                    # real_data.stack(multihot_ehr_label) if args.target_disease in ['pancreas',
                    #                                                                'mimic'] else real_data.extend(
                    #     multihot_ehr_label)
                    # time_steps.stack(time_step) if args.target_disease in ['pancreas', 'mimic'] else time_steps.extend(
                    #     time_step)

                # Calculate median perplexity outside the loop
                median_ppl = torch.median(torch.cat(lpl_list))
                print(f'Median perplexity: {median_ppl.item()}')

                with open(str(args.save_dir) + csv_filename, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([args.seed, median_ppl.item()])

                # for i, data in enumerate(tqdm(train_loader, desc="Generating synthetic data", leave=False)):
                #
                #     if args.target_disease in ['pancreas', 'mimic']:
                #         ehr, time_step = data
                #
                #         multihot_ehr_label = torch.zeros((ehr.size(0), args.max_len, pad_id), dtype=torch.float32) + 1e-7
                #         multihot_ehr_label = multihot_ehr_label.to(device)
                #         for batch_idx in range(ehr.size(0)):
                #             for seq_idx in range(args.max_len):
                #                 for label in range(pad_id):
                #                     if label != pad_id:
                #                         multihot_ehr_label[batch_idx, seq_idx, label] = 1.0
                #
                #
                #         _, gen_ehr = model(ehr, None)
                #         time_step = time_step.cpu().numpy().tolist()
                #         synthetic_data.stack(gen_ehr)
                #         real_data.stack(multihot_ehr_label)
                #         time_steps.stack(time_step)
                #     else:
                #         ehr, label, time_step = data
                #
                #         multihot_ehr_label = torch.zeros((ehr.size(0), args.max_len, pad_id), dtype=torch.float32) + 1e-7
                #         multihot_ehr_label = multihot_ehr_label.to(device)
                #         for batch_idx in range(ehr.size(0)):
                #             for seq_idx in range(args.max_len):
                #                 for label in range(pad_id):
                #                     if label != pad_id:
                #                         multihot_ehr_label[batch_idx, seq_idx, label] = 1.0
                #
                #         _, gen_ehr = model(ehr, label)
                #         time_step = time_step.cpu().numpy().tolist()
                #         synthetic_data.extend(gen_ehr)
                #         real_data.extend(multihot_ehr_label)
                #         time_steps.extend(time_step)
                #
                #     # Now calculate lpl per visit:
                # lpl_list = []
                # for i in range(len(synthetic_data)):
                #     for j in range(len(synthetic_data[i])):
                #         # Extract logits and targets for the current visit of the current batch item
                #         logits_visit = synthetic_data[i, j, :]
                #         target_visit = multihot_ehr_label[i, j, :]
                #         prob = logits_visit.softmax(dim=-1)[target_visit == 1]
                #         nll = -torch.log(prob + 1e-10)
                #         ppl = nll.exp()
                #         if torch.isnan(ppl).any():
                #             warnings.warn('NaN perplexity detected during lpl calculation')
                #         lpl_list.append(ppl)
                # median_ppl = torch.median(torch.cat(lpl_list))
                # print(f'Median perplexity: {median_ppl.item()}')
                #
                # with open(str(args.save_dir) + csv_filename, 'a') as f:
                #     writer = csv.writer(f)
                #     writer.writerow([args.seed, median_ppl.item()])


        #         og_data, og_label, og_time_step = pickle.load(open(data_path + '_training_new.pickle', 'rb'))
        #         og_data.extend(synthetic_data)
        #         og_time_step.extend(time_steps)
        #         og_label.extend(labels)
        #         pickle.dump((og_data, og_label, og_time_step), open(str(args.save_dir) + combinedData_filename, 'wb'))
        #
        # w2v = Word2Vec.load(str(args.save_dir) + word2vec_filename)
        # device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
        #
        # predictor = LSTM_predictor(pad_id, args.max_num_codes, 100, args.dropout)
        # predictor.to(device)
        #
        # train_dataset = MyDataset(str(args.save_dir) + combinedData_filename,
        #                           args.max_len, args.max_num_codes, pad_id, device)
        # test_dataset = MyDataset(data_path + '_testing_new.pickle',
        #                          args.max_len, args.max_num_codes, pad_id, device)
        # val_dataset = MyDataset(data_path + '_validation_new.pickle',
        #                         args.max_len, args.max_num_codes, pad_id, device)
        # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        #
        # optimizer = Adam(predictor.parameters(), lr=args.learning_rate)
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.factor, patience=args.patience,
        #                                            verbose=True)
        #
        # CE_loss = nn.CrossEntropyLoss(reduction='mean')
        #
        # print('Start training...Parameters')
        #
        # for name, param in predictor.named_parameters():
        #     if param.requires_grad:
        #         print('\t{:45}\ttrainable\t{}'.format(name, param.size()))
        #     else:
        #         print('\t{:45}\tfixed\t{}'.format(name, param.size()))
        # num_params = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
        # print('\ttotal:', num_params)
        # print()
        # print('-' * 71)
        #
        # global_step, best_dev_epoch = 0, 0
        # best_dev_f1 = 0.0
        # best_test_pr, best_test_f1, best_test_kappa = 0.0, 0.0, 0.0
        # total_loss = 0.0
        #
        # for epoch_id in tqdm(range(args.n_epochs), desc="Epochs"):
        #     epoch_start_time = time.time()
        #     predictor.train()
        #
        #     for i, data in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        #         ehr, _, label, _ = data
        #         prediction = predictor(ehr)
        #         loss = CE_loss(prediction, label)
        #         loss.backward()
        #         optimizer.step()
        #
        #     epoch_end_time = time.time()
        #     print(f'Training time for epoch {epoch_id}: {(epoch_end_time - epoch_start_time):.2f} seconds.')
        #
        #     predictor.eval()
        #
        #     print('-' * 71)
        #     print('evaluating train set...')
        #     train_accuary, train_precision, train_recall, train_f1, train_roc_auc, train_pr_auc, train_kappa, train_loss = eval_metric(
        #         train_loader, predictor)
        #     print('evaluating val set...')
        #     dev_accuary, dev_precision, dev_recall, dev_f1, dev_roc_auc, dev_pr_auc, dev_kappa, dev_loss = eval_metric(
        #         val_loader, predictor)
        #     print('evaluating test set...')
        #     test_accuary, test_precision, test_recall, test_f1, test_roc_auc, test_pr_auc, test_kappa, test_loss = eval_metric(
        #         test_loader, predictor)
        #     print('-' * 71)
        #     scheduler.step(dev_loss)
        #
        #     print(
        #         f'| epoch {epoch_id:03d} | train accuary {train_accuary:.3f} | val accuary {dev_accuary:.3f} | test accuary {test_accuary:.3f} |')
        #     print(
        #         f'| epoch {epoch_id:03d} | train precision {train_precision:.3f} | val precision {dev_precision:.3f} | test precision {test_precision:.3f} |')
        #     print(
        #         f'| epoch {epoch_id:03d} | train recall {train_recall:.3f} | val recall {dev_recall:.3f} | test recall {test_recall:.3f} |')
        #     print(f'| epoch {epoch_id:03d} | train f1 {train_f1:.3f} | val f1 {dev_f1:.3f} | test f1 {test_f1:.3f} |')
        #     print(
        #         f'| epoch {epoch_id:03d} | train roc_auc {train_roc_auc:.3f} | val roc_auc {dev_roc_auc:.3f} | test roc_auc {test_roc_auc:.3f} |')
        #     print(
        #         f'| epoch {epoch_id:03d} | train pr_auc {train_pr_auc:.3f} | val pr_auc {dev_pr_auc:.3f} | test pr_auc {test_pr_auc:.3f} |')
        #     print(
        #         f'| epoch {epoch_id:03d} | train kappa {train_kappa:.3f} | val kappa {dev_kappa:.3f} | test kappa {test_kappa:.3f} |')
        #     print(
        #         f'| epoch {epoch_id:03d} | train loss {train_loss:.3f} | val loss {dev_loss:.3f} | test loss {test_loss:.3f} |')
        #     print('-' * 71)
        #
        #     if dev_f1 > best_dev_f1:
        #         best_dev_f1 = dev_f1
        #         best_test_pr, best_test_f1, best_test_kappa = test_precision, test_f1, test_kappa
        #         best_dev_epoch = epoch_id
        #         torch.save(predictor.state_dict(), str(args.save_dir) + str(args.target_disease) + '_predictor.pt')
        #         print('Saving model (epoch {})'.format(epoch_id + 1))
        #         print('-' * 71)
        #     if epoch_id - best_dev_epoch > args.max_epochs_before_stop:
        #         print('Stop training at epoch {}. The highest f1 achieved is {}'.format(epoch_id, best_dev_f1))
        #         break
        #
        # results_file = open(str(args.save_dir) + csv_filename, 'w')
        # writer = csv.writer(results_file)
        # writer.writerow([best_test_pr, best_test_f1, best_test_kappa])


if __name__ == '__main__':

    name = 'tabDDPM'
    seeds = [1]
    datas = ['mimic']
    max_lens = [20]
    max_nums = [20]
    save_dir = './tabDDPM_dir/'

    for seed in seeds:
        for data, max_len, max_num in zip(datas, max_lens, max_nums):
            main(name, seed, data, max_len, max_num, save_dir)