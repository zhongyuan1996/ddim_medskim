import argparse
import os
import time
import csv

import torch
# import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    precision_recall_curve, auc, cohen_kappa_score, log_loss
from torch.optim import Adam, lr_scheduler
from models.dataset import MyDataset as meddiff_dataset, MyDataset_with_single_label as med_diff_dataset_single_label, collate_fn as diff_collate_fn
from utils.utils import check_path, export_config, bool_flag
from models.GAN_baselines import *
from torch.utils.data import Dataset, DataLoader
torch.autograd.set_detect_anomaly(True)
import yaml

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def eval_metric(eval_set, model, model_name = None):
    model.eval()
    with torch.no_grad():
        y_true = np.array([])
        y_pred = np.array([])
        y_score = np.array([])
        y_pred_fake = np.array([])
        for i, data in enumerate(eval_set):
            ehr, time_step, labels = data
            fp_og, fp_fake, fp_decode, og_h, fake_h, decode_h = model(ehr, time_step, labels)
            scores = torch.softmax(fp_og, dim=-1)
            scores = scores.data.cpu().numpy()
            fake_scores = torch.softmax(fp_fake, dim=-1)
            fake_scores = fake_scores.data.cpu().numpy()
            labels = labels.data.cpu().numpy()
            if model_name != 'LSTM_actGAN' and model_name != 'LSTM_medGAN':
                labels = labels.argmax(1)
            score = scores[:, 1]
            pred = scores.argmax(1)
            fake_pred = fake_scores.argmax(1)
            y_true = np.concatenate((y_true, labels))
            y_pred = np.concatenate((y_pred, pred))
            y_score = np.concatenate((y_score, score))
            y_pred_fake = np.concatenate((y_pred_fake, fake_pred))
        accuary = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_score)
        lr_precision, lr_recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(lr_recall, lr_precision)
        kappa = cohen_kappa_score(y_true, y_pred)
        loss = log_loss(y_true, y_pred)
        fake_loss = log_loss(y_true, y_pred_fake)
    return accuary, precision, recall, f1, roc_auc, pr_auc, kappa, loss, fake_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=True, type=bool_flag, nargs='?', const=True, help='use GPU')
    parser.add_argument('--seed', default=1234, type=int, help='seed')
    parser.add_argument('-bs', '--batch_size', default=128, type=int)
    parser.add_argument('-me', '--max_epochs_before_stop', default=15, type=int)
    parser.add_argument('--encoder', default='LSTM_medGAN', choices=['LSTM_ehrGAN', 'LSTM_GcGAN', 'LSTM_actGAN', 'LSTM_medGAN'])
    parser.add_argument('--d_model', default=256, type=int, help='dimension of hidden layers')
    parser.add_argument('--dense_model', default=64, type=int)
    parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate of hidden layers')
    parser.add_argument('--dropout_emb', default=0.1, type=float, help='dropout rate of embedding layers')
    parser.add_argument('--num_layers', default=1, type=int, help='number of transformer layers of EHR encoder')
    parser.add_argument('--num_heads', default=4, type=int, help='number of attention heads')
    parser.add_argument('--max_len', default=50, type=int, help='max visits of EHR')
    parser.add_argument('--max_num_codes', default=20, type=int, help='max number of ICD codes in each visit')
    parser.add_argument('--max_num_blks', default=100, type=int, help='max number of blocks in each visit')
    parser.add_argument('--blk_emb_path', default='./data/processed/block_embedding.npy',
                        help='embedding path of blocks')
    parser.add_argument('--target_disease', default='Heart_failure',
                        choices=['Heart_failure', 'COPD', 'Kidney', 'Dementia', 'Amnesia', 'mimic'])
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
    parser.add_argument('--model', default='Selected', choices=['Selected'])
    parser.add_argument('--save_dir', default='./saved_models/', help='models output directory')
    parser.add_argument("--config", type=str, default='ehr.yml', help="Path to the config file")
    parser.add_argument("--h_model", type=int, default=256, help="dimension of hidden state in LSTM")
    parser.add_argument("--lambda_DF_loss", type=float, default=0.1, help="scale of diffusion model loss")
    parser.add_argument("--lambda_CE_gen_loss", type=float, default=0.5, help="scale of generated sample loss")
    parser.add_argument("--patience", type=int, default=5, help="learning rate patience")
    parser.add_argument("--factor", type=float, default=0.2, help="learning rate factor")

    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    # elif args.mode == 'pred':
    #     pred(args)
    # elif args.mode == 'study':
    #     study(args)
    else:
        raise ValueError('Invalid mode')


def train(args):
    print(args)
    # random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if torch.cuda.is_available() and args.cuda:
    #     torch.cuda.manual_seed(args.seed)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    files = os.listdir(str(args.save_dir))
    if str(args.encoder) + '_' + str(args.target_disease) + '_' + str(args.seed) + '.csv' in files:
        print("conducted_experiments")
    else:

        config_path = os.path.join(args.save_dir, 'config.json')
        model_path = os.path.join(args.save_dir, str(args.encoder) + '_' + str(args.target_disease) + '_' + str(args.seed) + '.pt')
        # log_path = os.path.join(args.save_dir, 'log.csv')
        # log_loss_path = os.path.join(args.save_dir, 'log_loss.csv')
        # stats_path = os.path.join(args.save_dir, 'stats.csv')
        # export_config(args, config_path)
        check_path(model_path)
        # with open(log_path, 'w') as fout:
        #     fout.write('step,train_auc,dev_auc,test_auc\n')
        # with open(log_loss_path, 'w') as lossout:
        #     lossout.write('step,DF_loss,CE_loss,CE_gen_loss,KL_loss\n')
        # with open(stats_path, 'w') as statout:
        #     statout.write('train_acc,dev_acc,test_acc,train_precision,dev_precision,test_precision,train_recall,dev_recall,test_recall,train_f1,dev_f1,test_f1,train_auc,dev_auc,test_auc,train_pr,dev_pr,test_pr,train_kappa,dev_kappa,test_kappa,train_loss,dev_loss,test_loss\n')

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
        elif args.target_disease == 'mimic':
            pad_id = 4894
            data_path = './data/mimic/mimic'
        else:
            raise ValueError('Invalid disease')
        device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
        if args.encoder in ['LSTM_actGAN', 'LSTM_medGAN'] and args.target_disease != 'mimic':
            train_dataset = med_diff_dataset_single_label(data_path + '_training_new.pickle',
                                      args.max_len, args.max_num_codes, pad_id, device)
            dev_dataset = med_diff_dataset_single_label(data_path + '_validation_new.pickle', args.max_len,
                                    args.max_num_codes, pad_id, device)
            test_dataset = med_diff_dataset_single_label(data_path + '_testing_new.pickle', args.max_len,
                                     args.max_num_codes, pad_id, device)
            train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=diff_collate_fn)
            dev_dataloader = DataLoader(dev_dataset, args.batch_size, shuffle=False, collate_fn=diff_collate_fn)
            test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=diff_collate_fn)
        elif args.encoder not in ['LSTM_actGAN', 'LSTM_medGAN'] and args.target_disease != 'mimic':
            train_dataset = meddiff_dataset(data_path + '_training_new.pickle',
                                      args.max_len, args.max_num_codes, pad_id, device)
            dev_dataset = meddiff_dataset(data_path + '_validation_new.pickle', args.max_len,
                                    args.max_num_codes, pad_id, device)
            test_dataset = meddiff_dataset(data_path + '_testing_new.pickle', args.max_len,
                                     args.max_num_codes, pad_id, device)
            train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=diff_collate_fn)
            dev_dataloader = DataLoader(dev_dataset, args.batch_size, shuffle=False, collate_fn=diff_collate_fn)
            test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=diff_collate_fn)
        elif args.encoder not in ['LSTM_actGAN', 'LSTM_medGAN'] and args.target_disease == 'mimic':
            train_dataset = meddiff_dataset(data_path + '_train.pickle',
                                       args.max_len, args.max_num_blks, pad_id, device)
            dev_dataset = meddiff_dataset(data_path + '_val.pickle',
                                     args.max_len,
                                     args.max_num_codes, pad_id, device)
            test_dataset = meddiff_dataset(data_path + '_test.pickle', args.max_len,
                                      args.max_num_codes, pad_id, device)
            train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=diff_collate_fn)
            dev_dataloader = DataLoader(dev_dataset, args.batch_size, shuffle=False, collate_fn=diff_collate_fn)
            test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=diff_collate_fn)

        elif args.encoder in ['LSTM_actGAN', 'LSTM_medGAN'] and args.target_disease == 'mimic':
            train_dataset = med_diff_dataset_single_label(data_path + '_train.pickle',
                                       args.max_len, args.max_num_codes, pad_id, device)
            dev_dataset = med_diff_dataset_single_label(data_path + '_val.pickle',
                                     args.max_len,
                                     args.max_num_codes, pad_id, device)
            test_dataset = med_diff_dataset_single_label(data_path + '_test.pickle', args.max_len,
                                      args.max_num_codes, pad_id, device)
            train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=diff_collate_fn)
            dev_dataloader = DataLoader(dev_dataset, args.batch_size, shuffle=False, collate_fn=diff_collate_fn)
            test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=diff_collate_fn)

        discriminator = Discriminator(args.d_model)
        discriminator.to(device)
        if args.encoder == 'LSTM_ehrGAN':
            generator = LSTM_generator(args.d_model, args.dense_model)
            generator.to(device)
            model = LSTM_ehrGAN(pad_id, args.d_model, args.h_model, args.dropout, args.dropout_emb, device, generator=generator)
        elif args.encoder == 'LSTM_GcGAN':
            generator = FCN_generator(args.h_model)
            generator.to(device)
            model = LSTM_GcGAN(pad_id, args.d_model, args.h_model, args.dropout, args.dropout_emb, device, generator=generator)
        elif args.encoder == 'LSTM_actGAN':
            generator = LSTM_generator(args.d_model, args.d_model)
            generator.to(device)
            model = LSTM_actGAN(pad_id, args.d_model, args.h_model, args.dropout, args.dropout_emb, device, generator=generator)
        elif args.encoder == 'LSTM_medGAN':
            generator = Linear_generator(args.d_model)
            generator.to(device)
            model = LSTM_actGAN(pad_id, args.d_model, args.h_model, args.dropout, args.dropout_emb, device,
                                generator=generator)
        model.to(device)

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.learning_rate},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': args.learning_rate}
        ]
        generator_optim = Adam(generator.parameters(), lr=args.learning_rate)
        discriminator_optim = Adam(discriminator.parameters(), lr=args.learning_rate)
        optim = Adam(grouped_parameters)
        scheduler = lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=args.factor, patience=args.patience, threshold=0,
                                                   threshold_mode='rel',
                                                   cooldown=0, min_lr=1e-07, eps=1e-08, verbose=True)
        generator_scheduler = lr_scheduler.ReduceLROnPlateau(generator_optim, mode='min', factor=args.factor, patience=args.patience, threshold=0,
                                                   threshold_mode='rel',
                                                   cooldown=0, min_lr=1e-07, eps=1e-08, verbose=True)
        discriminator_scheduler = lr_scheduler.ReduceLROnPlateau(discriminator_optim, mode='min', factor=args.factor, patience=args.patience, threshold=0,
                                                   threshold_mode='rel',
                                                   cooldown=0, min_lr=1e-07, eps=1e-08, verbose=True)

        loss_func_pred = nn.CrossEntropyLoss(reduction='mean')
        loss_bce = nn.BCELoss(reduction='mean')

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
        total_GD_loss = 0.0
        model.train()

        for epoch_id in range(args.n_epochs):
            print('epoch: {:5} '.format(epoch_id))
            model.train()
            start_time = time.time()

            for i, data in enumerate(train_dataloader):

                ehr, time_step, labels = data

                optim.zero_grad()
                generator_optim.zero_grad()


                fp_og, fp_fake, fp_decode, og_h, fake_h, decode_h = model(ehr, time_step, labels)

                generator_discriminator_out = discriminator(fake_h)
                true_labels = torch.ones_like(generator_discriminator_out, device=device)
                generator_loss = loss_bce(generator_discriminator_out, true_labels)
                generator_loss.backward(retain_graph=True)
                generator_optim.step()

                discriminator_optim.zero_grad()
                true_discriminator_out = discriminator(og_h)
                true_discriminator_loss = loss_bce(true_discriminator_out, true_labels)

                generator_discriminator_out = discriminator(fake_h.detach())
                generator_discriminator_loss = loss_bce(generator_discriminator_out, 1 - true_labels)
                discriminator_loss = (true_discriminator_loss+generator_discriminator_loss)/2
                discriminator_loss.backward(retain_graph=True)
                discriminator_optim.step()

                CE_loss = loss_func_pred(fp_og, labels)

                CE_loss.backward(retain_graph=True)
                total_loss += (CE_loss.item() / labels.size(0)) * args.batch_size
                total_GD_loss += (discriminator_loss.item() / labels.size(0)) * args.batch_size

                if args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optim.step()

                if (global_step + 1) % args.log_interval == 0:
                    total_loss /= args.log_interval
                    total_GD_loss /= args.log_interval

                    ms_per_batch = 1000 * (time.time() - start_time) / args.log_interval
                    print('| step {:5} | loss {:7.4f} | ms/batch {:7.2f} |'.format(global_step,
                                                                                   total_loss,
                                                                                   ms_per_batch))
                    print('| CE_loss {:7.4f} | GD_loss {:7.4f} |'.format(total_loss, total_GD_loss))
                    # with open(log_loss_path, 'a') as lossout:
                    #     lossout.write('{},{},{}\n'.format(global_step, total_loss, total_GD_loss))

                    total_loss = 0.0
                    total_GD_loss = 0.0
                    start_time = time.time()
                global_step += 1

            model.eval()
            train_acc, tr_precision, tr_recall, tr_f1, tr_roc_auc, tr_pr_auc, tr_kappa, tr_loss, tr_fake_loss = eval_metric(
                train_dataloader,
                model, args.encoder)
            dev_acc, d_precision, d_recall, d_f1, d_roc_auc, d_pr_auc, d_kappa, d_loss, d_fake_loss = eval_metric(
                dev_dataloader, model, args.encoder)
            test_acc, t_precision, t_recall, t_f1, t_roc_auc, t_pr_auc, t_kappa, t_loss, t_fake_loss = eval_metric(
                test_dataloader, model , args.encoder)
            scheduler.step(d_loss)
            generator_scheduler.step(d_fake_loss)
            discriminator_scheduler.step(d_fake_loss)
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
            print('| step {:5} | train_loss {:7.4f} | dev_loss {:7.4f} | test_loss {:7.4f}'.format(global_step,
                                                                                              tr_loss,
                                                                                              d_loss,
                                                                                              t_loss))

            print('-' * 71)
            # with open(stats_path, 'a') as statout:
            #     statout.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(train_acc,dev_acc,test_acc,tr_precision,d_precision,t_precision,tr_recall,d_recall,t_recall,tr_f1,d_f1,t_f1,tr_roc_auc,d_roc_auc,t_roc_auc,tr_pr_auc,d_pr_auc,t_pr_auc,tr_kappa,d_kappa,t_kappa,tr_loss,d_loss,t_loss))
            if d_f1 >= best_dev_auc:
                best_dev_auc = d_f1
                final_test_auc = t_f1
                best_dev_epoch = epoch_id
                best_epoch_pr = t_pr_auc
                best_epoch_f1 = t_f1
                best_epoch_kappa = t_kappa
                torch.save([model, args], model_path)
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


# def pred(args):
#     model_path = os.path.join(args.save_dir, 'models.pt')
#     model, old_args = torch.load(model_path)
#     device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
#     model.to(device)
#     model.eval()
#     blk_emb = np.load(old_args.blk_emb_path)
#     blk_pad_id = len(blk_emb) - 1
#     if old_args.target_disease == 'Heart_failure':
#         code2id = pickle.load(open('./data/hf/hf_code2idx_new.pickle', 'rb'))
#         # id2code = {int(v): k for k, v in code2id.items()}
#         # code2topic = pickle.load(open('./data/hf/hf_code2topic.pickle', 'rb'))
#         pad_id = len(code2id)
#         data_path = './data/hf/hf'
#     elif old_args.target_disease == 'COPD':
#         code2id = pickle.load(open('./data/copd/copd_code2idx_new.pickle', 'rb'))
#         # id2code = {int(v): k for k, v in code2id.items()}
#         # code2topic = pickle.load(open('./data/copd/copd_code2topic.pickle', 'rb'))
#         pad_id = len(code2id)
#         data_path = './data/copd/copd'
#     elif old_args.target_disease == 'Kidney':
#         code2id = pickle.load(open('./data/kidney/kidney_code2idx_new.pickle', 'rb'))
#         # id2code = {int(v): k for k, v in code2id.items()}
#         # code2topic = pickle.load(open('./data/kidney/kidney_code2topic.pickle', 'rb'))
#         pad_id = len(code2id)
#         data_path = './data/kidney/kidney'
#     elif old_args.target_disease == 'Amnesia':
#         code2id = pickle.load(open('./data/amnesia/amnesia_code2idx_new.pickle', 'rb'))
#         # id2code = {int(v): k for k, v in code2id.items()}
#         # code2topic = pickle.load(open('./data/amnesia/amnesia_code2topic.pickle', 'rb'))
#         pad_id = len(code2id)
#         data_path = './data/amnesia/amnesia'
#     elif old_args.target_disease == 'Dementia':
#         code2id = pickle.load(open('./data/dementia/dementia_code2idx_new.pickle', 'rb'))
#         # id2code = {int(v): k for k, v in code2id.items()}
#         # code2topic = pickle.load(open('./data/dementia/dementia_code2topic.pickle', 'rb'))
#         pad_id = len(code2id)
#         data_path = './data/dementia/dementia'
#     else:
#         raise ValueError('Invalid disease')
#     # dev_dataset = MyDataset(data_path + '_validation_new.pickle', data_path + '_validation_txt.pickle',
#     #                         old_args.max_len, old_args.max_num_codes, old_args.max_num_blks, pad_id, blk_pad_id, device)
#     test_dataset = MyDataset(data_path + '_testing_new.pickle', old_args.max_len,
#                              old_args.max_num_codes, pad_id, device)
#     # dev_dataloader = DataLoader(dev_dataset, args.batch_size, shuffle=False, collate_fn=collate_fn)
#     test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=collate_fn)
#     # dev_acc, d_precision, d_recall, d_f1, d_roc_auc, d_pr_auc = eval_metric(dev_dataloader, model)
#     test_acc, t_precision, t_recall, t_f1, t_roc_auc, t_pr_auc, t_kappa = eval_metric(test_dataloader, model)
#     with torch.no_grad():
#         y_true = np.array([])
#         y_pred = np.array([])
#         y_score = np.array([])
#         skip_rates = np.array([])
#         for i, data in enumerate(test_dataloader):
#             ehr, time_step, labels = data
#             logits, skip_rate = model(ehr, lengths, time_step, code_mask)
#             scores = torch.softmax(logits, dim=-1)
#             scores = scores.data.cpu().numpy()
#             labels = labels.data.cpu().numpy()
#             skip_rate = skip_rate.data.cpu().numpy()
#             score = scores[:, 1]
#             pred = scores.argmax(1)
#             y_true = np.concatenate((y_true, labels))
#             y_pred = np.concatenate((y_pred, pred))
#             y_score = np.concatenate((y_score, score))
#             skip_rates = np.concatenate((skip_rates, [skip_rate]))
#         log_path = os.path.join(args.save_dir, 'result.csv')
#         with open(log_path, 'w') as fout:
#             fout.write('test_auc,test_f1,test_pre,test_recall,test_pr_auc,test_kappa,skip_rate\n')
#             fout.write(
#                 '{},{},{},{},{},{},{}\n'.format(t_roc_auc, t_f1, t_precision, t_recall, t_pr_auc, t_kappa,
#                                                 np.mean(skip_rates)))
#         with open(os.path.join(args.save_dir, 'prediction.csv'), 'w') as fout2:
#             fout2.write('prediciton,score,label\n')
#             for i in range(len(y_true)):
#                 fout2.write('{},{},{}\n'.format(y_pred[i], y_score[i], y_true[i]))


# def study(args):
#     model_path = os.path.join(args.save_dir, 'models.pt')
#     model, old_args = torch.load(model_path)
#     device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
#     model.to(device)
#     model.eval()
#     blk_emb = np.load(old_args.blk_emb_path)
#     blk_pad_id = len(blk_emb) - 1
#     if old_args.target_disease == 'Heart_failure':
#         code2id = pickle.load(open('./data/hf/hf_code2idx_new.pickle', 'rb'))
#         id2code = {int(v): k for k, v in code2id.items()}
#         code2topic = pickle.load(open('./data/hf/hf_code2topic.pickle', 'rb'))
#         pad_id = len(code2id)
#         data_path = './data/hf/hf'
#     elif old_args.target_disease == 'COPD':
#         code2id = pickle.load(open('./data/copd/copd_code2idx_new.pickle', 'rb'))
#         id2code = {int(v): k for k, v in code2id.items()}
#         code2topic = pickle.load(open('./data/copd/copd_code2topic.pickle', 'rb'))
#         pad_id = len(code2id)
#         data_path = './data/copd/copd'
#     elif old_args.target_disease == 'Kidney':
#         code2id = pickle.load(open('./data/kidney/kidney_code2idx_new.pickle', 'rb'))
#         id2code = {int(v): k for k, v in code2id.items()}
#         code2topic = pickle.load(open('./data/kidney/kidney_code2topic.pickle', 'rb'))
#         pad_id = len(code2id)
#         data_path = './data/kidney/kidney'
#     elif old_args.target_disease == 'Amnesia':
#         code2id = pickle.load(open('./data/amnesia/amnesia_code2idx_new.pickle', 'rb'))
#         id2code = {int(v): k for k, v in code2id.items()}
#         code2topic = pickle.load(open('./data/amnesia/amnesia_code2topic.pickle', 'rb'))
#         pad_id = len(code2id)
#         data_path = './data/amnesia/amnesia'
#     elif old_args.target_disease == 'Dementia':
#         code2id = pickle.load(open('./data/dementia/dementia_code2idx_new.pickle', 'rb'))
#         id2code = {int(v): k for k, v in code2id.items()}
#         code2topic = pickle.load(open('./data/dementia/dementia_code2topic.pickle', 'rb'))
#         pad_id = len(code2id)
#         data_path = './data/dementia/dementia'
#     else:
#         raise ValueError('Invalid disease')
#     dev_dataset = MyDataset(data_path + '_validation_new.pickle', data_path + '_validation_txt.pickle',
#                             old_args.max_len, old_args.max_num_codes, old_args.max_num_blks, pad_id, blk_pad_id, device)
#     test_dataset = MyDataset(data_path + '_testing_new.pickle', data_path + '_testing_txt.pickle', old_args.max_len,
#                              old_args.max_num_codes, old_args.max_num_blks, pad_id, blk_pad_id, device)
#     dev_dataloader = DataLoader(dev_dataset, args.batch_size, shuffle=False, collate_fn=collate_fn)
#     test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=collate_fn)
#     icd2des = {}
#     df = pd.read_excel('./data/CMS32_DESC_LONG_SHORT_DX.xlsx')
#     icds = df['DIAGNOSIS CODE']
#     desc = df['LONG DESCRIPTION']
#     for i in range(len(icds)):
#         icd = icds[i][0:3] + '.' + icds[i][3:]
#         icd2des[icd] = desc[i]
#     with torch.no_grad():
#         with open(os.path.join(args.save_dir, 'code_weight.txt'), 'w') as fout:
#             for i, data in enumerate(test_dataloader):
#                 labels, ehr, mask, txt, mask_txt, lengths, time_step, code_mask = data
#                 logits, v_skips, c_skips, p = model.infer(ehr, lengths, time_step)
#                 code2w = {}
#                 for id in range(0, model.vocab_size + 1):
#                     if id in id2code.keys():
#                         code = id2code[id]
#                         if code in icd2des.keys():
#                             des = icd2des[code]
#                         else:
#                             des = ''
#                         w = p.data.cpu().numpy()[id]
#                         # if w == 0.0:
#                         #     print(id)
#                         code2w[code] = w
#                         fout.write(str(code) + ',' + str(des) + ',' + str(w) + '\n')
#                 break
#
#         with open(os.path.join(args.save_dir, 'decode_test.txt'), 'w') as fout:
#
#             for i, data in enumerate(test_dataloader):
#                 labels, ehr, mask, txt, mask_txt, lengths, time_step, code_mask = data
#                 logits, v_skips, c_skips, p_mask = model.infer(ehr, lengths, time_step)
#
#                 predictions = logits.argmax(-1)
#                 for l, pred, record, v_skip, c_skip, length in zip(labels, predictions, ehr, v_skips, c_skips, lengths):
#                     output = {}
#                     output["label"] = l.item()
#                     output["prediction"] = pred.item()
#                     record = record.data.cpu().numpy()
#                     # icd
#                     rec_li = []
#                     for visit in record:
#                         v_li = []
#                         for codeid in visit:
#                             if codeid in id2code.keys():
#                                 v_li.append(id2code[codeid])
#                         if len(v_li) > 0:
#                             rec_li.append(v_li)
#                         else:
#                             break
#                     output["record_icd"] = rec_li
#                     # icd text
#                     des_li = []
#                     for vi in rec_li:
#                         v_des = []
#                         for code in vi:
#                             if code in icd2des.keys():
#                                 v_des.append(icd2des[code])
#                         des_li.append(v_des)
#                     output["record_text"] = des_li
#                     length = length.data.cpu().numpy()
#                     v_skip = v_skip.data.cpu().numpy()[0: length]
#                     c_skip = c_skip.data.cpu().numpy()[0: length]
#                     if output["label"] == 1 and len(output["record_icd"]) <= 10:
#                         fout.write("label: {}".format(str(output["label"])) + '\n')
#                         fout.write("prediction: {}".format(str(output["prediction"])) + "\n")
#                         fout.write("record_icd: {}".format(str(output["record_icd"])) + '\n')
#                         fout.write("record_text: {}".format(str(output["record_text"])) + '\n')
#                         fout.write("v_skip: {}".format(str(v_skip)) + '\n')
#                         fout.write("c_skip: {}".format(str(c_skip)) + '\n')
#                         fout.write('\n')


if __name__ == '__main__':
    main()
