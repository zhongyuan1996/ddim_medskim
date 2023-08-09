import argparse
import os
import time
import pandas as pd
import csv
import random

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
from models.embGen_LSTM_ddim import *
import yaml

target_disease = 'Heart_failure'
max_len = 50
max_num_codes = 20
save_dir = './ehrMap/Heart_failure'
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default=True, type=bool_flag, nargs='?', const=True, help='use GPU')
parser.add_argument('--seed', default=1234, type=int, help='seed')
parser.add_argument('-bs', '--batch_size', default=32, type=int)
parser.add_argument('--model', default='medDiff')
parser.add_argument('-me', '--max_epochs_before_stop', default=10, type=int)
parser.add_argument('--d_model', default=256, type=int, help='dimension of hidden layers')
parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate of hidden layers')
parser.add_argument('--dropout_emb', default=0.1, type=float, help='dropout rate of embedding layers')
parser.add_argument('--num_layers', default=1, type=int, help='number of transformer layers of EHR encoder')
parser.add_argument('--num_heads', default=4, type=int, help='number of attention heads')
parser.add_argument('--max_len', default=max_len, type=int, help='max visits of EHR')
parser.add_argument('--max_num_codes', default=max_num_codes, type=int, help='max number of ICD codes in each visit')
parser.add_argument('--max_num_blks', default=100, type=int, help='max number of blocks in each visit')
parser.add_argument('--blk_emb_path', default='./data/processed/block_embedding.npy',
                    help='embedding path of blocks')
parser.add_argument('--target_disease', default=target_disease,
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
parser.add_argument("--temperature", type=str, default='temperature', help="temperature control of classifier softmax")
parser.add_argument("--mintau", type=float, default=0.5, help="parameter mintau of temperature control")
parser.add_argument("--maxtau", type=float, default=5.0, help="parameter maxtau of temperature control")
parser.add_argument("--patience", type=int, default=5, help="learning rate patience")
parser.add_argument("--factor", type=float, default=0.2, help="learning rate factor")
args = parser.parse_args()
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

device = torch.device("cuda:0" if torch.cuda.is_available() and True else "cpu")


test_dataset = MyDataset_mapping(data_path + '_testing_new.pickle', args.max_len ,
                         args.max_num_codes, pad_id, device)

test_dataloader = DataLoader(test_dataset, 8, shuffle=False, collate_fn=collate_fn)

with open(os.path.join("configs/", args.config), "r") as f:
    config = yaml.safe_load(f)
def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace
config = dict2namespace(config)
model = RNNdiff


for j, data in enumerate(test_dataloader):
    ehr_map, time_step_map, labels_map, codemask_map = data

exit(0)
