import argparse
import os
import time
import pandas as pd

import torch
# import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    precision_recall_curve, auc, cohen_kappa_score, log_loss
from torch.optim import Adam, lr_scheduler
# from tqdm import tqdm
from models.dataset import *
from models.medskim import *
from utils.utils import check_path, export_config, bool_flag
# from utils.icd_rel import *
from utils.diffUtil import *
from models.softmax_LSTM_ddim import *
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


def eval_metric(eval_set, model):
    model.eval()
    with torch.no_grad():
        y_true = np.array([])
        y_pred = np.array([])
        y_score = np.array([])
        e_ts = np.array([])
        E_t_gens = np.array([])

        for i, data in enumerate(eval_set):
            ehr, time_step, labels = data
            e_t,E_t_gen,final_prediction,_,_,_ = model(ehr, time_step)

            scores = torch.softmax(final_prediction, dim=-1)
            scores = scores.data.cpu().numpy()
            e_t = e_t.data.cpu().numpy()
            E_t_gen = E_t_gen.data.cpu().numpy()

            labels = labels.data.cpu().numpy()
            labels = labels.argmax(1)
            score = scores[:, 1]
            pred = scores.argmax(1)

            y_true = np.concatenate((y_true, labels))
            y_pred = np.concatenate((y_pred, pred))
            y_score = np.concatenate((y_score, score))
            try:
                e_ts = np.concatenate((e_ts, e_t), axis=0)
            except ValueError:
                e_ts = e_t

            try:
                E_t_gens = np.concatenate((E_t_gens, E_t_gen), axis=0)
            except ValueError:
                E_t_gens = E_t_gen


        accuary = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_score)
        lr_precision, lr_recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(lr_recall, lr_precision)
        kappa = cohen_kappa_score(y_true, y_pred)
        loss = log_loss(y_true, y_pred)
    return accuary, precision, recall, f1, roc_auc, pr_auc, kappa, loss, e_ts, E_t_gens , y_true, y_pred

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=True, type=bool_flag, nargs='?', const=True, help='use GPU')
    parser.add_argument('--seed', default=1234, type=int, help='seed')
    parser.add_argument('-bs', '--batch_size', default=32, type=int)
    parser.add_argument('-me', '--max_epochs_before_stop', default=15, type=int)
    parser.add_argument('--d_model', default=256, type=int, help='dimension of hidden layers')
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
                        choices=['Heart_failure', 'COPD', 'Kidney', 'Dementia', 'Amnesia'])
    parser.add_argument('--target_att_heads', default=4, type=int, help='target disease attention heads number')
    parser.add_argument('--mem_size', default=15, type=int, help='memory size')
    parser.add_argument('--mem_update_size', default=15, type=int, help='memory update size')
    parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=0.001, type=float)
    parser.add_argument('--target_rate', default=0.3, type=float)
    parser.add_argument('--lamda', default=0.1, type=float)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='max grad norm (0 to disable)')
    parser.add_argument('--warmup_steps', default=200, type=int)
    parser.add_argument('--n_epochs', default=50, type=int)
    parser.add_argument('--log_interval', default=20, type=int)
    parser.add_argument('--mode', default='train', choices=['train', 'pred', 'study'],
                        help='run training or evaluation')
    parser.add_argument('--model', default='Selected', choices=['Selected'])
    parser.add_argument('--save_dir', default='./saved_models/', help='models output directory')
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

    config_path = os.path.join(args.save_dir, 'config.json')
    model_path = os.path.join(args.save_dir, 'models.pt')
    log_path = os.path.join(args.save_dir, 'log.csv')
    log_loss_path = os.path.join(args.save_dir, 'log_loss.csv')
    stats_path = os.path.join(args.save_dir, 'stats.csv')

    export_config(args, config_path)
    check_path(model_path)
    with open(log_path, 'w') as fout:
        fout.write('step,train_auc,dev_auc,test_auc\n')
    with open(log_loss_path, 'w') as lossout:
        lossout.write('step,DF_loss,CE_loss,CE_gen_loss,KL_loss\n')
    with open(stats_path, 'w') as statout:
        statout.write('train_acc,dev_acc,test_acc,train_precision,dev_precision,test_precision,train_recall,dev_recall,test_recall,train_f1,dev_f1,test_f1,train_auc,dev_auc,test_auc,train_pr,dev_pr,test_pr,train_kappa,dev_kappa,test_kappa,train_loss,dev_loss,test_loss\n')


    # blk_emb = np.load(args.blk_emb_path)
    # blk_pad_id = len(blk_emb) - 1
    # icd2cui = pickle.load(open('./data/semmed/icd2cui.pickle', 'rb'))
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
    else:
        raise ValueError('Invalid disease')
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    train_dataset = MyDataset(data_path + '_training_new.pickle',
                              args.max_len, args.max_num_codes, pad_id, device)
    dev_dataset = MyDataset(data_path + '_validation_new.pickle', args.max_len,
                            args.max_num_codes, pad_id, device)
    test_dataset = MyDataset(data_path + '_testing_new.pickle', args.max_len,
                             args.max_num_codes, pad_id, device)
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_dataset, args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=collate_fn)

    with open(os.path.join("configs/", args.config), "r") as f:

        config = yaml.safe_load(f)
    config = dict2namespace(config)

    # model = testSimpleRNN(config, vocab_size=pad_id, d_model=args.d_model, h_model=args.h_model,
    #                 dropout=args.dropout, dropout_emb=args.dropout_emb, device = device)
    model = RNNdiff(config, vocab_size=pad_id, d_model=args.d_model, h_model=args.h_model,
                    dropout=args.dropout, dropout_emb=args.dropout_emb, device = device)

    # if args.model == 'Selected':
    #     model = Selected(pad_id, args.d_model, args.dropout, args.dropout_emb)
    # else:
    #     raise ValueError('Invalid model')
    model.to(device)

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': args.learning_rate}
    ]
    optim = Adam(grouped_parameters)
    scheduler = lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=args.factor, patience=args.patience, threshold=0,
                                               threshold_mode='rel',
                                               cooldown=0, min_lr=1e-07, eps=1e-08, verbose=True)
    Loss_func_diff = nn.MSELoss(reduction='mean')
    # Loss_func_h = nn.KLDivLoss(reduction='batchmean')
    loss_func_pred = nn.CrossEntropyLoss(reduction='mean')
    # scheduler = get_cosine_schedule_with_warmup(optim, args.warmup_steps, 2500)
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
    total_DF_loss, total_CE_loss, total_CE_gen_loss, total_KL_loss = 0.0, 0.0, 0.0, 0.0
    model.train()
    if args.temperature == 'temperature':
        tau_schedule = np.linspace(args.maxtau, args.mintau, num=int(args.n_epochs/2))
        constant = np.full(len(tau_schedule), args.mintau)
        tau_schedule = np.append(tau_schedule, constant)

        assert len(tau_schedule) == args.n_epochs

    for epoch_id in range(args.n_epochs):
        print('epoch: {:5} '.format(epoch_id))
        model.train()
        start_time = time.time()

        for i, data in enumerate(train_dataloader):

            ehr, time_step, labels = data

            optim.zero_grad()
            h_res, h_gen_v2, pred, pred_v2, noise, diff_noise = model(ehr, time_step)

            # if args.temperature == 'temperature':
            #     pred = pred/tau_schedule[epoch_id]
            #     pred_v2 = pred_v2/tau_schedule[epoch_id]

            DF_loss = Loss_func_diff(diff_noise, noise) * args.lambda_DF_loss
            # KL_loss = Loss_func_h(h_res.log(), h_gen_v2) * args.lambda_KL_loss
            CE_loss = loss_func_pred(pred, labels)
            CE_gen_loss = loss_func_pred(pred_v2, labels) * args.lambda_CE_gen_loss
            loss = DF_loss + CE_loss + CE_gen_loss
            # loss = CE_loss
            loss.backward()
            total_loss += (loss.item() / labels.size(0)) * args.batch_size
            total_DF_loss += (DF_loss.item() / labels.size(0)) * args.batch_size
            total_CE_loss += (CE_loss.item() / labels.size(0)) * args.batch_size
            total_CE_gen_loss += (CE_gen_loss.item() / labels.size(0)) * args.batch_size
            # total_KL_loss += (KL_loss.item() / labels.size(0)) * args.batch_size
            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optim.step()
            # scheduler.step()

            if (global_step + 1) % args.log_interval == 0:
                total_loss /= args.log_interval
                total_DF_loss /= args.log_interval
                total_CE_loss /= args.log_interval
                total_CE_gen_loss /= args.log_interval
                total_KL_loss /= args.log_interval

                ms_per_batch = 1000 * (time.time() - start_time) / args.log_interval
                print('| step {:5} | loss {:7.4f} | ms/batch {:7.2f} |'.format(global_step,
                                                                               total_loss,
                                                                               ms_per_batch))
                print('| DF_loss {:7.4f} | CE_loss {:7.4f} | CE_gen_loss {:7.4f} | KL_loss {:7.4f} |'.format(total_DF_loss,
                                                                               total_CE_loss,total_CE_gen_loss,total_KL_loss))
                with open(log_loss_path, 'a') as lossout:
                    lossout.write('{},{},{},{},{}\n'.format(global_step, total_DF_loss, total_CE_loss, total_CE_gen_loss, total_KL_loss))

                total_loss = 0.0
                total_DF_loss, total_CE_loss, total_CE_gen_loss, total_KL_loss = 0.0, 0.0, 0.0, 0.0
                start_time = time.time()
            global_step += 1

        model.eval()
        train_acc, tr_precision, tr_recall, tr_f1, tr_roc_auc, tr_pr_auc, tr_kappa, tr_loss,_,_,_,_ = eval_metric(train_dataloader,
                                                                                                 model)
        dev_acc, d_precision, d_recall, d_f1, d_roc_auc, d_pr_auc, d_kappa, d_loss,_,_,_,_ = eval_metric(dev_dataloader, model)
        test_acc, t_precision, t_recall, t_f1, t_roc_auc, t_pr_auc, t_kappa, t_loss, e_t, gen_e_t, t_label,t_pred = eval_metric(test_dataloader, model)
        scheduler.step(d_loss)
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
        with open(stats_path, 'a') as statout:
            statout.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(train_acc,dev_acc,test_acc,tr_precision,d_precision,t_precision,tr_recall,d_recall,t_recall,tr_f1,d_f1,t_f1,tr_roc_auc,d_roc_auc,t_roc_auc,tr_pr_auc,d_pr_auc,t_pr_auc,tr_kappa,d_kappa,t_kappa,tr_loss,d_loss,t_loss))
        if d_f1 >= best_dev_auc:
            best_dev_auc = d_f1
            final_test_auc = t_f1
            best_dev_epoch = epoch_id
            torch.save([model, args], model_path)
            with open(log_path, 'a') as fout:
                fout.write('{},{},{},{}\n'.format(global_step, tr_pr_auc, d_pr_auc, t_pr_auc))
            print(f'model saved to {model_path}')


            softmaxres_fileName = 'e_t_epoch_' + str(epoch_id) + '.csv'
            gen_softmaxres_gen_fileName = 'gen_e_t_epoch_' + str(epoch_id) + '.csv'

            label_fileName = 'label_epoch_' + str(epoch_id) + '.csv'
            pred_fileName = 'pred_epoch_' + str(epoch_id) + '.csv'

            softmax_path = os.path.join(args.save_dir, softmaxres_fileName)
            gen_softmax_path = os.path.join(args.save_dir, gen_softmaxres_gen_fileName)

            label_path = os.path.join(args.save_dir, label_fileName)
            pred_path = os.path.join(args.save_dir, pred_fileName)


            np.savetxt(softmax_path, e_t, delimiter=',')
            np.savetxt(gen_softmax_path, gen_e_t, delimiter=',')
            np.savetxt(label_path, t_label, delimiter=',')
            np.savetxt(pred_path, t_pred, delimiter=',')


        # if epoch_id - best_dev_epoch >= args.max_epochs_before_stop:
        #     break

    print()
    print('training ends in {} steps'.format(global_step))
    print('best dev auc: {:.4f} (at epoch {})'.format(best_dev_auc, best_dev_epoch))
    print('final test auc: {:.4f}'.format(final_test_auc))
    print()



if __name__ == '__main__':
    main()
