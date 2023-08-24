import argparse
import os
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
from models.ehrGAN import *
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine
import yaml
from tqdm import tqdm
import nmslib


def eval_metricGAN(model, discriminator, dataloader, criterion, device):
    """
    Calculate the generator and discriminator losses for a given dataset.
    """
    model.eval()  # switch model to evaluation mode

    gen_loss_total = 0.0
    disc_loss_total = 0.0

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            ehr, time_step, labels, mask = data

            # Pass the data through the model
            gen_ehr, real_ehr = model(ehr, mask)

            # Compute the generator loss
            fake_score = discriminator(gen_ehr)
            gen_loss = criterion(fake_score, torch.ones_like(fake_score))

            # Compute the discriminator loss
            real_score = discriminator(real_ehr)
            real_loss = criterion(real_score, torch.ones_like(real_score))
            fake_loss = criterion(fake_score, torch.zeros_like(fake_score))
            disc_loss = (real_loss + fake_loss) / 2

            gen_loss_total += gen_loss.item()
            disc_loss_total += disc_loss.item()

    # Return the harmonic mean of the generator and discriminator losses
    if gen_loss_total == 0 or disc_loss_total == 0:
        return float('inf')
    return 2.0 / ((1.0 / (gen_loss_total / len(dataloader))) + (1.0 / (disc_loss_total / len(dataloader))))



def eval_metric(eval_set, model):
    model.eval()
    with torch.no_grad():
        y_true = np.array([])
        y_pred = np.array([])
        y_score = np.array([])

        for i, data in enumerate(eval_set):
            ehr, _, labels, _ = data
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
    parser.add_argument('-bs', '--batch_size', default=8, type=int)
    parser.add_argument('--model', default='ehrGAN')
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
    parser.add_argument('--target_disease', default=data, choices=['Heartfailure', 'COPD', 'Kidney', 'Dementia', 'Amnesia', 'mimic', 'ARF', 'Shock', 'mortality'])
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
    parser.add_argument("--temperature", type=str, default='temperature', help="temperature control of classifier softmax")
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
    csv_filename = 'ehrGAN_' + str(args.seed) + '_' + str(args.target_disease) + '_result.csv'
    combinedData_filename = str(args.target_disease) + '_combinedData.pickle'
    genData_filename = str(args.target_disease) + '_genData.pickle'
    if csv_filename in files:
        print('This experiment has been done!')
    else:
        initial_d=0
        if args.target_disease == 'Heartfailure':
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
        elif args.target_disease == 'mimic':
            pad_id = 4894
            data_path = './data/mimic/mimic'
        else:
            raise ValueError('Invalid disease')
        #check if word2vec model exists
        word2vec_filename = str(args.target_disease) + '_word2vec.pt'
        ehrGAN_filename = str(args.target_disease) + '_ehrGAN.pt'

        if not os.path.exists(str(save_dir)+word2vec_filename):
            with open(data_path + '_training_new.pickle', 'rb') as f:
                train_set, _, _ = pickle.load(f)
            with open(data_path + '_testing_new.pickle', 'rb') as f:
                test_set, _, _ = pickle.load(f)
            with open(data_path + '_validation_new.pickle', 'rb') as f:
                val_set, _, _ = pickle.load(f)
            all_set = train_set + test_set + val_set
            flattened_all_set = [visit for patient in all_set for visit in patient]
            w2v = Word2Vec(flattened_all_set, vector_size=200, window=5, min_count=1, workers=4)
            w2v.save(str(args.save_dir) + word2vec_filename)
        else:
            print('word2vec model exists')

        if not os.path.exists(str(save_dir)+ehrGAN_filename):
            w2v = Word2Vec.load(str(args.save_dir) + word2vec_filename)
            device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
            train_dataset = ehrGANDataset(data_path + '_training_new.pickle',
                                      args.max_len, args.max_num_codes, pad_id, w2v, device)
            test_dataset = ehrGANDataset(data_path + '_testing_new.pickle',
                                        args.max_len, args.max_num_codes, pad_id, w2v, device)
            val_dataset = ehrGANDataset(data_path + '_validation_new.pickle',
                                        args.max_len, args.max_num_codes, pad_id, w2v, device)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

            model = ehrGAN(x_dim=200, h_dim=100, dropout=args.dropout, device=device)
            discriminator = CNNDiscriminator(200, args.dropout)
            model.to(device)
            discriminator.to(device)
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': args.weight_decay, 'lr': args.learning_rate},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0, 'lr': args.learning_rate}
            ]
            optimizer = Adam(grouped_parameters, lr=args.learning_rate)
            dis_optimizer = Adam(discriminator.parameters(), lr=args.learning_rate)
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.factor, patience=args.patience,
                                                         verbose=True, threshold=0.0001, threshold_mode='rel',
                                                         cooldown=0, min_lr=0, eps=1e-08)
            dis_scheduler = lr_scheduler.ReduceLROnPlateau(dis_optimizer, mode='min', factor=args.factor, patience=args.patience,
                                                         verbose=True, threshold=0.0001, threshold_mode='rel',
                                                         cooldown=0, min_lr=0, eps=1e-08)

            CE_loss = nn.CrossEntropyLoss(reduction='mean')
            BCE_loss = nn.BCEWithLogitsLoss(reduction='mean')

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
            total_loss = 0.0

            for epoch_id in tqdm(range(12), desc="Epochs"):
                epoch_start_time = time.time()
                model.train()

                for i, data in enumerate(tqdm(train_loader, desc="Training", leave=False)):
                    ehr, _, _, mask = data
                    gen_ehr, real_ehr = model(ehr, mask)

                    # train discriminator
                    real_label = torch.ones(args.batch_size, 1).to(device)
                    fake_label = torch.zeros(args.batch_size, 1).to(device)

                    # real data
                    discriminator.zero_grad()
                    real_out = discriminator(real_ehr)
                    real_loss = BCE_loss(real_out, real_label)  # use an appropriate loss criterion
                    real_loss.backward(retain_graph=True)

                    # fake data
                    fake_out = discriminator(gen_ehr.detach())
                    fake_loss = BCE_loss(fake_out, fake_label)
                    fake_loss.backward(retain_graph=True)

                    d_loss = real_loss + fake_loss
                    dis_optimizer.step()  # update discriminator's weights

                    # train generator
                    model.zero_grad()
                    fake_out = discriminator(gen_ehr)
                    g_loss = BCE_loss(fake_out, real_label)
                    g_loss.backward()

                    optimizer.step()  # update generator's weights

                epoch_end_time = time.time()
                print(f'Training time for epoch {epoch_id}: {(epoch_end_time - epoch_start_time):.2f} seconds.')

                model.eval()

                print('-' * 71)
                print('evaluating train set...')
                train_loss = eval_metricGAN(model, discriminator, train_loader, BCE_loss, device)
                print('evaluating val set...')
                dev_loss = eval_metricGAN(model, discriminator, val_loader, BCE_loss, device)
                print('evaluating test set...')
                test_loss = eval_metricGAN(model, discriminator, test_loader, BCE_loss, device)
                print('-' * 71)
                scheduler.step(dev_loss)
                dis_scheduler.step(dev_loss)

                print(
                    f'| epoch {epoch_id:03d} | train loss {train_loss:.3f} | val loss {dev_loss:.3f} | test loss {test_loss:.3f} |')
                print('-' * 71)

                if dev_loss < best_val_loss:
                    best_val_loss = dev_loss
                    best_dev_epoch = epoch_id
                    torch.save(model.state_dict(), str(args.save_dir) + ehrGAN_filename)
                    print('Saving model (epoch {})'.format(epoch_id + 1))
                    print('-' * 71)
                if epoch_id - best_dev_epoch > args.max_epochs_before_stop:
                    print('Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch_id, best_val_loss))
                    break

        if not os.path.exists(str(save_dir)+combinedData_filename):
            with torch.no_grad():
                w2v = Word2Vec.load(str(args.save_dir) + word2vec_filename)
                all_word_vectors_matrix = w2v.wv.vectors

                index = nmslib.init(method='hnsw', space='cosinesimil')
                index.addDataPointBatch(all_word_vectors_matrix)
                index.createIndex({'post': 2}, print_progress=True)

                device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
                ehrganModel = ehrGAN(x_dim=200, h_dim=100, dropout=args.dropout, device=device)
                ehrganModel.load_state_dict(torch.load(str(args.save_dir) + ehrGAN_filename))
                ehrganModel.to(device)
                ehrganModel.eval()

                train_dataset = ehrGANDatasetWOAggregate(data_path + '_training_new.pickle',
                                              args.max_len, args.max_num_codes, pad_id, w2v, device)
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

                synthetic_data = []
                time_steps = []
                labels = []
                for i, data in enumerate(tqdm(train_loader, desc="Generating synthetic data", leave=False)):
                    ehr, time_step, label, mask = data
                    gen_ehr, real_ehr = ehrganModel(ehr, mask)
                    bs, seq_len, code_len, emb_dim = gen_ehr.size()
                    gen_ehr = gen_ehr.contiguous().view(bs*seq_len*code_len, emb_dim).cpu().numpy()

                    results = index.knnQueryBatch(gen_ehr,k=1)  # for each embedding, find the 1 nearest neighbor in the word vector matrix
                    ids = [result[0] for result in results]
                    distances = [result[1] for result in results]  # for each embedding, find the 1 nearest neighbor in the word vector matrix

                    gen_codes = [w2v.wv.index_to_key[i[0]] for i in ids]

                    gen_codes = np.array(gen_codes).reshape(bs, seq_len, code_len)
                    time_step = time_step.cpu().numpy().tolist()

                    label = [l.argmax(0) for l in label]
                    time_steps.extend(time_step)
                    synthetic_data.extend(gen_codes)
                    labels.extend(label)

                gen_codes_unique = []
                time_step_cleaned = []
                og_data, og_label, og_time_step = pickle.load(open(data_path + '_training_new.pickle', 'rb'))

                for i in range(len(synthetic_data)):  # iterate over patients
                    patient_visits = []
                    cleaned_patient_time_step = []
                    for j in range(len(synthetic_data[i])):  # iterate over visits for a given patient
                        if time_steps[i][j] != 100000:  # Exclude the time step equals to 100000
                            unique_codes = list(set(synthetic_data[i][j]))
                            patient_visits.append(unique_codes)
                            cleaned_patient_time_step.append(time_steps[i][j])

                    gen_codes_unique.append(patient_visits)
                    time_step_cleaned.append(cleaned_patient_time_step)

                for i in range(len(synthetic_data)):
                    gen_length = len(synthetic_data[i])
                    if gen_length < max_len:
                        if len(og_data[i]) > gen_length:
                            # Retain only the last 'gen_length' visits
                            og_data[i] = og_data[i][-gen_length:]
                            og_time_step[i] = og_time_step[i][-gen_length:]
                    elif gen_length == max_len:
                        if len(og_data[i]) > max_len:
                            # Retain only the last 'max_len' visits
                            og_data[i] = og_data[i][-max_len:]
                            og_time_step[i] = og_time_step[i][-max_len:]

                # Now, check for one patient that all the visit lengths are equal
                for i in range(len(og_data)):
                    if not (len(og_data[i]) == len(gen_codes_unique[i]) == len(og_time_step[i]) == len(
                            time_step_cleaned[i])):
                        print(f"Length mismatch at index {i}.")


                og_data.extend(gen_codes_unique)
                og_time_step.extend(time_step_cleaned)
                og_label.extend(labels)
                pickle.dump((og_data, og_label, og_time_step), open(str(args.save_dir) + combinedData_filename, 'wb'))
                pickle.dump((gen_codes_unique, labels, time_step_cleaned), open(str(args.save_dir) + genData_filename, 'wb'))


        # w2v = Word2Vec.load(str(args.save_dir) + word2vec_filename)
        # device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
        # predictor = CNNPredictor(200, 2, args.dropout)
        # predictor.to(device)
        #
        # train_dataset = ehrGANDataset(str(args.save_dir) + combinedData_filename,
        #                               args.max_len, args.max_num_codes, pad_id, w2v, device)
        # test_dataset = ehrGANDataset(data_path + '_testing_new.pickle',
        #                             args.max_len, args.max_num_codes, pad_id, w2v, device)
        # val_dataset = ehrGANDataset(data_path + '_validation_new.pickle',
        #                             args.max_len, args.max_num_codes, pad_id, w2v, device)
        # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        #
        # optmizer = Adam(predictor.parameters(), lr=args.learning_rate)
        # scheduler = lr_scheduler.ReduceLROnPlateau(optmizer, mode='min', factor=args.factor, patience=args.patience,
        #                                                 verbose=True, threshold=0.0001, threshold_mode='rel',
        #                                                 cooldown=0, min_lr=0, eps=1e-08)
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
        #         optmizer.step()
        #
        #     epoch_end_time = time.time()
        #     print(f'Training time for epoch {epoch_id}: {(epoch_end_time - epoch_start_time):.2f} seconds.')
        #
        #     predictor.eval()
        #
        #     print('-' * 71)
        #     print('evaluating train set...')
        #     train_accuary, train_precision, train_recall, train_f1, train_roc_auc, train_pr_auc, train_kappa, train_loss = eval_metric(train_loader, predictor)
        #     print('evaluating val set...')
        #     dev_accuary, dev_precision, dev_recall, dev_f1, dev_roc_auc, dev_pr_auc, dev_kappa, dev_loss = eval_metric(val_loader, predictor)
        #     print('evaluating test set...')
        #     test_accuary, test_precision, test_recall, test_f1, test_roc_auc, test_pr_auc, test_kappa, test_loss = eval_metric(test_loader, predictor)
        #     print('-' * 71)
        #     scheduler.step(dev_loss)
        #
        #     print(f'| epoch {epoch_id:03d} | train accuary {train_accuary:.3f} | val accuary {dev_accuary:.3f} | test accuary {test_accuary:.3f} |')
        #     print(f'| epoch {epoch_id:03d} | train precision {train_precision:.3f} | val precision {dev_precision:.3f} | test precision {test_precision:.3f} |')
        #     print(f'| epoch {epoch_id:03d} | train recall {train_recall:.3f} | val recall {dev_recall:.3f} | test recall {test_recall:.3f} |')
        #     print(f'| epoch {epoch_id:03d} | train f1 {train_f1:.3f} | val f1 {dev_f1:.3f} | test f1 {test_f1:.3f} |')
        #     print(f'| epoch {epoch_id:03d} | train roc_auc {train_roc_auc:.3f} | val roc_auc {dev_roc_auc:.3f} | test roc_auc {test_roc_auc:.3f} |')
        #     print(f'| epoch {epoch_id:03d} | train pr_auc {train_pr_auc:.3f} | val pr_auc {dev_pr_auc:.3f} | test pr_auc {test_pr_auc:.3f} |')
        #     print(f'| epoch {epoch_id:03d} | train kappa {train_kappa:.3f} | val kappa {dev_kappa:.3f} | test kappa {test_kappa:.3f} |')
        #     print(f'| epoch {epoch_id:03d} | train loss {train_loss:.3f} | val loss {dev_loss:.3f} | test loss {test_loss:.3f} |')
        #     print('-' * 71)
        #
        #     if dev_f1 > best_dev_f1:
        #         best_dev_f1 = dev_f1
        #         best_dev_epoch = epoch_id
        #         best_test_pr, best_test_f1, best_test_kappa = test_precision, test_f1, test_kappa
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

    name = 'ehrGAN'
    seeds = [1, 2, 3, 4, 5]
    datas = ['Heartfailure', 'COPD', 'Kidney', 'Amnesia', 'mimic']
    max_lens = [50,50,50,50,15]
    max_nums = [45,45,45,45,45]
    save_dir = './ehrGAN_dir/'

    for seed in seeds:
        for data, max_len, max_num in zip(datas, max_lens, max_nums):
            main(name, seed, data, max_len, max_num, save_dir)

    # datas = ['Heartfailure', 'COPD', 'Kidney', 'Amnesia', 'mimic']
    #
    # # Dictionaries to store max values for each dataset
    # max_visits_per_data = {}
    # max_codes_per_data = {}
    # patient_with_max_visits_per_data = {}
    #
    # for data in datas:
    #     if data == 'Heartfailure':
    #         data_path = './data/hf/hf'
    #     elif data == 'COPD':
    #         data_path = './data/copd/copd'
    #     elif data == 'Kidney':
    #         data_path = './data/kidney/kidney'
    #     elif data == 'Amnesia':
    #         data_path = './data/amnesia/amnesia'
    #     elif data == 'mimic':
    #         data_path = './data/mimic/mimic'
    #     else:
    #         raise ValueError('Invalid disease')
    #
    #     train_ehr, _, _ = pickle.load(open(data_path + '_training_new.pickle', 'rb'))
    #     test_ehr, _, _ = pickle.load(open(data_path + '_testing_new.pickle', 'rb'))
    #     val_ehr, _, _ = pickle.load(open(data_path + '_validation_new.pickle', 'rb'))
    #
    #     all_ehr = train_ehr + test_ehr + val_ehr
    #
    #     # Initialize max_visit and max_code for the current dataset
    #     max_visit = 0
    #     max_code = 0
    #     patient_with_max_visit = None  # Initialize the patient with max visit
    #
    #     # Calculate max_visit and max_code for the current dataset
    #     for patient_visits in all_ehr:
    #         if len(patient_visits) > max_visit:
    #             max_visit = len(patient_visits)
    #             patient_with_max_visit = patient_visits
    #         for visit in patient_visits:
    #             max_code = max(max_code, len(visit))
    #
    #     # Store the results in the dictionaries
    #     max_visits_per_data[data] = max_visit
    #     max_codes_per_data[data] = max_code
    #     patient_with_max_visits_per_data[data] = patient_with_max_visit
    #
    # # Print out the results
    # for data in datas:
    #     print(f"Dataset: {data}")
    #     print("Max number of visits:", max_visits_per_data[data])
    #     print("Max number of codes in a visit:", max_codes_per_data[data])
    #     print("Patient with max visits:", patient_with_max_visits_per_data[data])
    #     print("------")