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
from utils.utils import check_path, export_config, bool_flag
from utils.icd_rel import *
from models.evaluators import Evaluator
from generatingModelTestingGround import Presence_Disclosure, Attribute_Disclosure
import warnings
import random
torch.autograd.set_detect_anomaly(True)
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



def indices_to_codes(indices, mapping):
    return [[mapping.get(idx) for idx in seq] for seq in indices]

def kl_loss(concat_miu, concat_logvar):
    loss = -0.5 * torch.sum(1 + concat_logvar - concat_miu.pow(2) - concat_logvar.exp())
    return loss.mean()


def compute_gradient_penalty(critic, real_samples, fake_samples):
    # Assuming real_samples and fake_samples are of shape [batch, seq_len, d_model]
    alpha = torch.rand(real_samples.size(0), 1, 1, device=real_samples.device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    critic_interpolates = critic(interpolates)

    # Compute the gradient of the critic scores with respect to the inputs
    gradients = torch.autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(critic_interpolates.size(), device=real_samples.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def main(seed, name, model, data, max_len, max_num, sav_dir, mode, short_ICD, toy):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=True, type=bool_flag, nargs='?', const=True, help='use GPU')
    parser.add_argument('--seed', default=seed, type=int, help='seed')
    parser.add_argument('-bs', '--batch_size', default=128, type=int)
    parser.add_argument('-me', '--max_epochs_before_stop', default=4, type=int)
    parser.add_argument('--d_model', default=256, type=int, help='dimension of hidden layers')
    parser.add_argument('--h_model', default=256, type=int, help='dimension of hidden layers')
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
    parser.add_argument('--n_epochs', default=15, type=int)
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
    parser.add_argument('--subtask', default='')
    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'gen':
        gen(args)
    else:
        raise ValueError('Invalid mode')

def gen(args):
    saving_path = './data/synthetic/'
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
    checkpoint_filepath = os.path.join(args.save_dir, str(args.model) + '_' + str(args.target_disease) + '_' + str(args.seed) + '.pt')

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
    elif args.target_disease == 'breast':
        diag2id = pd.read_csv('./data/breast/ae_to_int_mapping.csv', header=None)
        drug2id = pd.read_csv('./data/breast/drug_to_int_mapping.csv', header=None)
        lab2id = pd.read_csv('./data/breast/lab_to_int_mapping.csv', header=None)
        proc2id = pd.read_csv('./data/breast/proc_to_int_mapping.csv', header=None)
        id2diag = dict(zip(diag2id[1], diag2id[0]))
        id2drug = dict(zip(drug2id[1], drug2id[0]))
        id2lab = dict(zip(lab2id[1], lab2id[0]))
        id2proc = dict(zip(proc2id[1], proc2id[0]))
        demo_len = 10
        diag_pad_id = len(diag2id)
        drug_pad_id = len(drug2id)
        lab_pad_id = len(lab2id)
        proc_pad_id = len(proc2id)
        diag_nan_id =  51
        drug_nan_id = 101
        lab_nan_id = 10
        proc_nan_id = 4
        pad_id_list = [diag_pad_id, drug_pad_id, lab_pad_id, proc_pad_id]
        nan_id_list = [diag_nan_id, drug_nan_id, lab_nan_id, proc_nan_id]
        data_path = './data/breast/'
    else:
        raise ValueError('Invalid disease')
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")


    if args.subtask == 'arf':
        train_dataset = pancreas_Gendataset(data_path + 'train_3dig' + str(args.target_disease) + '_arf' + '.csv',
                                            args.max_len,
                                            args.max_num_codes, pad_id_list, nan_id_list, device, task='arf')
    elif args.subtask == 'shock':
        train_dataset = pancreas_Gendataset(data_path + 'train_3dig' + str(args.target_disease) + '_shock' + '.csv',
                                            args.max_len,
                                            args.max_num_codes, pad_id_list, nan_id_list, device, task='shock')
    elif args.subtask == 'mortality':
        train_dataset = pancreas_Gendataset(data_path + 'train_3dig' + str(args.target_disease) + '_mortality' + '.csv',
                                            args.max_len,
                                            args.max_num_codes, pad_id_list, nan_id_list, device, task='mortality')
    elif args.short_ICD and not args.toy:
        train_dataset = pancreas_Gendataset(data_path + 'train_3dig' + str(args.target_disease) + '.csv',
                                            args.max_len,
                                            args.max_num_codes, pad_id_list, nan_id_list, device)
    elif args.toy:
        train_dataset = pancreas_Gendataset(data_path + 'train_3dig' + str(args.target_disease) + '3_subset' + '.csv',
                                            args.max_len,
                                            args.max_num_codes, pad_id_list, nan_id_list, device)
    else:
        train_dataset = pancreas_Gendataset(data_path + 'train_5dig' + str(args.target_disease) + '.csv',
                                           args.max_len,
                                           args.max_num_codes, pad_id_list, nan_id_list, device)
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=False, collate_fn=gen_collate_fn)

    model, _ = torch.load(checkpoint_filepath)
    model.eval()

    diag_pd_list, drug_pd_list, lab_pd_list, proc_pd_list = [], [], [], []
    diag_ad_list, drug_ad_list, lab_ad_list, proc_ad_list = [], [], [], []
    diag_data, drug_data, lab_data, proc_data, timegap = [], [], [], [], []
    labels = []
    demographic = []

    for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Generating"):
        diag_seq, drug_seq, lab_seq, proc_seq, time_step, visit_timegaps, diag_timegaps, drug_timegaps, lab_timegaps, proc_timegaps, \
            diag_mask, drug_mask, lab_mask, proc_mask, diag_length, drug_length, lab_length, proc_length, demo, label = data
        if args.model == 'LSTM-medGAN':
            _, _, _, _, gen_diag_logits, gen_drug_logits, gen_lab_logits, gen_proc_logits, _, _ = model(diag_seq, drug_seq, lab_seq, proc_seq, diag_length)
        elif args.model == 'LSTM-MLP':
            gen_diag_logits, gen_drug_logits, gen_lab_logits, gen_proc_logits, _, _, _, _, _, _ = model(diag_seq, drug_seq, lab_seq, proc_seq, diag_length)
        elif args.model == 'synTEG':
            _, _, _, _, gen_diag_logits, gen_drug_logits, gen_lab_logits, gen_proc_logits, _, _ = model(diag_seq, drug_seq, lab_seq, proc_seq, diag_length)
        elif args.model == 'TWIN':
            gen_diag_logits, gen_drug_logits, gen_lab_logits, gen_proc_logits, _, _ = model(diag_seq, drug_seq, lab_seq, proc_seq, diag_length)
        elif args.model == 'LSTM-TabDDPM':
            gen_diag_logits, gen_drug_logits, gen_lab_logits, gen_proc_logits, _, _, _, _, _, _, _ ,_ = model(diag_seq, drug_seq, lab_seq, proc_seq, diag_length)
        elif args.model == 'EVA':
            gen_diag_logits, gen_drug_logits, gen_lab_logits, gen_proc_logits, _, _ = model(diag_seq, drug_seq, lab_seq, proc_seq, diag_length)
        else:
            raise ValueError('Invalid model')
        k = 20
        _, topk_diag_indices = torch.topk(gen_diag_logits, min(k,gen_diag_logits.shape[-1]), dim=-1)
        _, topk_drug_indices = torch.topk(gen_drug_logits,  min(k,gen_drug_logits.shape[-1]), dim=-1)
        _, topk_lab_indices = torch.topk(gen_lab_logits,  min(k,gen_lab_logits.shape[-1]), dim=-1)
        _, topk_proc_indices = torch.topk(gen_proc_logits,  min(k,gen_proc_logits.shape[-1]), dim=-1)

        diag_data.extend(topk_diag_indices.tolist())
        drug_data.extend(topk_drug_indices.tolist())
        lab_data.extend(topk_lab_indices.tolist())
        proc_data.extend(topk_proc_indices.tolist())
        labels.extend(label.tolist())
        timegap_placeholder = torch.arange(0, args.max_len).unsqueeze(0).repeat(topk_diag_indices.shape[0], 1)
        timegap.extend(timegap_placeholder.tolist())
        demographic.extend(demo.tolist())

        diag_mask = torch.arange(diag_seq.size(1), device=device)[None, :] < diag_length[:, None]
        drug_mask = torch.arange(drug_seq.size(1), device=device)[None, :] < drug_length[:, None]
        lab_mask = torch.arange(lab_seq.size(1), device=device)[None, :] < lab_length[:, None]
        proc_mask = torch.arange(proc_seq.size(1), device=device)[None, :] < proc_length[:, None]

        # Apply masks to the topk indices and real sequences
        topk_diag_indices = topk_diag_indices * diag_mask.unsqueeze(-1)
        topk_drug_indices = topk_drug_indices * drug_mask.unsqueeze(-1)
        topk_lab_indices = topk_lab_indices * lab_mask.unsqueeze(-1)
        topk_proc_indices = topk_proc_indices * proc_mask.unsqueeze(-1)

        perscent = 0.35

        diag_pd = Presence_Disclosure(model, diag_seq, topk_diag_indices, 'diag', diag_length, perscent)
        drug_pd = Presence_Disclosure(model, drug_seq, topk_drug_indices, 'drug', drug_length, perscent)
        lab_pd = Presence_Disclosure(model, lab_seq, topk_lab_indices, 'lab', lab_length, perscent)
        proc_pd = Presence_Disclosure(model, proc_seq, topk_proc_indices, 'proc', proc_length, perscent)

        diag_ad = Attribute_Disclosure(model, diag_seq, topk_diag_indices, 'diag', diag_length, perscent, diag_pad_id)
        drug_ad = Attribute_Disclosure(model, drug_seq, topk_drug_indices, 'drug', drug_length, perscent, drug_pad_id)
        lab_ad = Attribute_Disclosure(model, lab_seq, topk_lab_indices, 'lab', lab_length, perscent, lab_pad_id)
        proc_ad = Attribute_Disclosure(model, proc_seq, topk_proc_indices, 'proc', proc_length, perscent, proc_pad_id)

        diag_pd_list.append(diag_pd)
        drug_pd_list.append(drug_pd)
        lab_pd_list.append(lab_pd)
        proc_pd_list.append(proc_pd)

        diag_ad_list.append(diag_ad)
        drug_ad_list.append(drug_ad)
        lab_ad_list.append(lab_ad)
        proc_ad_list.append(proc_ad)

    assert len(labels) == len(diag_data)

    patients_df = pd.DataFrame({
        'DIAGNOSES_int': diag_data,
        'DRG_CODE_int': drug_data,
        'LAB_ITEM_int': lab_data,
        'PROC_ITEM_int': proc_data,
        'time_gaps': timegap,
        'MORTALITY': labels,
        'demo': demographic
    })


    if args.subtask == 'arf':
        patients_df.to_csv(data_path + str(args.model) + '_synthetic_3dig' + str(args.target_disease) + '_arf.csv', index=False)
    elif args.subtask == 'shock':
        patients_df.to_csv(data_path + str(args.model) + '_synthetic_3dig' + str(args.target_disease) + '_shock.csv', index=False)
    elif args.subtask == 'mortality':
        patients_df.to_csv(data_path + str(args.model) + '_synthetic_3dig' + str(args.target_disease) + '_mortality.csv', index=False)
    elif args.short_ICD:
        patients_df.to_csv(data_path + str(args.model) + '_synthetic_3dig' + str(args.target_disease) + '.csv', index=False)
    else:
        patients_df.to_csv(data_path + str(args.model) + '_synthetic_5dig' + str(args.target_disease) + '.csv', index=False)

    print('diag_pd', np.mean(diag_pd_list))
    print('drug_pd', np.mean(drug_pd_list))
    print('lab_pd', np.mean(lab_pd_list))
    print('proc_pd', np.mean(proc_pd_list))
    print('total_pd', np.mean(diag_pd_list) + np.mean(drug_pd_list) + np.mean(lab_pd_list) + np.mean(proc_pd_list))
    print('*'*50)
    print('diag_ad', np.mean(diag_ad_list))
    print('drug_ad', np.mean(drug_ad_list))
    print('lab_ad', np.mean(lab_ad_list))
    print('proc_ad', np.mean(proc_ad_list))
    print('total_ad', np.mean(diag_ad_list) + np.mean(drug_ad_list) + np.mean(lab_ad_list) + np.mean(proc_ad_list))

    return

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
        elif args.target_disease == 'mimic4':
            diag2id = pd.read_csv('./data/mimic4/diagnosis_to_int_mapping_mimic4.csv', header=None)
            drug2id = pd.read_csv('./data/mimic4/drug_to_int_mapping_mimic4.csv', header=None)
            lab2id = pd.read_csv('./data/mimic4/lab_to_int_mapping_mimic4.csv', header=None)
            proc2id = pd.read_csv('./data/mimic4/proc_to_int_mapping_mimic4.csv', header=None)
            demo_len = 2
            diag_pad_id = len(diag2id)
            drug_pad_id = len(drug2id)
            lab_pad_id = len(lab2id)
            proc_pad_id = len(proc2id)
            diag_nan_id = 855
            drug_nan_id = 38
            lab_nan_id = 147
            proc_nan_id = 2
            pad_id_list = [diag_pad_id, drug_pad_id, lab_pad_id, proc_pad_id]
            nan_id_list = [diag_nan_id, drug_nan_id, lab_nan_id, proc_nan_id]
            data_path = './data/mimic4/'
        elif args.target_disease == 'eicu':
            diag2id = pd.read_csv('./data/eicu/diagnosis_to_int_mapping_3dig.csv', header=None)
            drug2id = pd.read_csv('./data/eicu/drug_to_int_mapping_3dig.csv', header=None)
            lab2id = pd.read_csv('./data/eicu/lab_to_int_mapping_3dig.csv', header=None)
            proc2id = pd.read_csv('./data/eicu/proc_to_int_mapping_3dig.csv', header=None)
            demo_len = 2
            diag_pad_id = len(diag2id)
            drug_pad_id = len(drug2id)
            lab_pad_id = len(lab2id)
            proc_pad_id = len(proc2id)
            diag_nan_id = 0
            drug_nan_id = 148
            lab_nan_id = 158
            proc_nan_id = 1207
            pad_id_list = [diag_pad_id, drug_pad_id, lab_pad_id, proc_pad_id]
            nan_id_list = [diag_nan_id, drug_nan_id, lab_nan_id, proc_nan_id]
            data_path = './data/eicu/'
        elif args.target_disease == 'breast':
            diag2id = pd.read_csv('./data/breast/ae_to_int_mapping.csv', header=None)
            drug2id = pd.read_csv('./data/breast/drug_to_int_mapping.csv', header=None)
            lab2id = pd.read_csv('./data/breast/lab_to_int_mapping.csv', header=None)
            proc2id = pd.read_csv('./data/breast/proc_to_int_mapping.csv', header=None)
            id2diag = dict(zip(diag2id[1], diag2id[0]))
            id2drug = dict(zip(drug2id[1], drug2id[0]))
            id2lab = dict(zip(lab2id[1], lab2id[0]))
            id2proc = dict(zip(proc2id[1], proc2id[0]))
            demo_len = 10
            diag_pad_id = len(diag2id)
            drug_pad_id = len(drug2id)
            lab_pad_id = len(lab2id)
            proc_pad_id = len(proc2id)
            diag_nan_id = 51
            drug_nan_id = 101
            lab_nan_id = 10
            proc_nan_id = 4
            pad_id_list = [diag_pad_id, drug_pad_id, lab_pad_id, proc_pad_id]
            nan_id_list = [diag_nan_id, drug_nan_id, lab_nan_id, proc_nan_id]
            data_path = './data/breast/'
        else:
            raise ValueError('Invalid disease')
        device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

        if args.short_ICD and not args.toy:
            train_dataset = pancreas_Gendataset(data_path + 'train_3dig' + str(args.target_disease) + '.csv',
                                                args.max_len, args.max_num_codes, pad_id_list, nan_id_list, device)
            dev_dataset = pancreas_Gendataset(data_path + 'val_3dig' + str(args.target_disease) + '.csv',
                                              args.max_len, args.max_num_codes, pad_id_list, nan_id_list, device)
            test_dataset = pancreas_Gendataset(data_path + 'test_3dig' + str(args.target_disease) + '.csv',
                                               args.max_len,
                                               args.max_num_codes, pad_id_list, nan_id_list, device)
        elif args.toy:
            train_dataset = pancreas_Gendataset(data_path + 'toy_3dig' + str(args.target_disease) + '.csv',
                                                args.max_len, args.max_num_codes, pad_id_list, nan_id_list, device)
            dev_dataset = pancreas_Gendataset(data_path + 'toy_3dig' + str(args.target_disease) + '.csv',
                                              args.max_len, args.max_num_codes, pad_id_list, nan_id_list, device)
            test_dataset = pancreas_Gendataset(data_path + 'toy_3dig' + str(args.target_disease) + '.csv',
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

        if args.model == 'LSTM-medGAN':
            generator = Linear_generator(args.d_model, args.h_model)
            generator.to(device)
            discriminator = Discriminator(args.d_model)
            discriminator.to(device)
            model = LSTM_medGAN(args.name, pad_id_list, args.d_model, args.dropout, generator)
        elif args.model == 'LSTM-MLP':
            model = LSTM_MLP(args.name, pad_id_list, args.d_model, args.dropout, None)
        elif args.model == 'synTEG':
            generator = Linear_generator(args.d_model, args.h_model)
            generator.to(device)
            discriminator = Discriminator(args.d_model)
            discriminator.to(device)
            model = synTEG(args.name, pad_id_list, args.d_model, args.dropout, generator)
        elif args.model == 'TWIN':
            generator = VAE_generator(args.d_model, int(args.h_model / 2), int(args.h_model / 4))
            generator.to(device)
            model = TWIN(args.name, pad_id_list, args.d_model, args.dropout, 5, generator)
        elif args.model == 'LSTM-TabDDPM':
            generator = Diff_generator(args.d_model, args.dropout, device)
            generator.to(device)
            model = LSTM_TabDDPM(args.name, pad_id_list, args.d_model, args.dropout, generator)
        elif args.model == 'EVA':
            generator = VAE_generator(args.d_model, int(args.h_model / 2), int(args.h_model / 4))
            generator.to(device)
            model = EVA(args.name, pad_id_list, args.d_model, args.dropout, generator)
        elif args.model == 'LSTM-Meddiff':
            generator = Diff_generator(args.d_model * 4, args.dropout, device)
            generator.to(device)
            model = LSTM_Meddiff(args.name, pad_id_list, args.d_model, args.dropout, generator)
        elif args.model == 'LSTM-ScoEHR':
            generator = Diff_generator(args.d_model, args.dropout, device)
            generator.to(device)
            model = LSTM_ScoEHR(args.name, pad_id_list, args.d_model, args.dropout, generator)
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
        if args.name == 'GAN' or args.name == 'WGAN-GP':
            generator_optim = Adam(generator.parameters(), lr=args.learning_rate)
            discriminator_optim = Adam(discriminator.parameters(), lr=args.learning_rate)
        optim = Adam(grouped_parameters)

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
        time_pred_loss = 0.0
        best_choosing_statistic = 1e10

        if args.name == 'WGAN-GP':
            current_step = 0
        for epoch_id in range(args.n_epochs):
            print('epoch: {:5} '.format(epoch_id))
            model.train()
            start_time = time.time()

            for i, data in enumerate(tqdm(train_dataloader, desc="Training", leave=False)):

                diag_seq, drug_seq, lab_seq, proc_seq, time_step, visit_timegaps, diag_timegaps, drug_timegaps, lab_timegaps, proc_timegaps,\
                    diag_mask, drug_mask, lab_mask, proc_mask, diag_length, drug_length, lab_length, proc_length, demo, _ = data
                if args.model == 'LSTM-medGAN':
                    real_diag_logits, real_drug_logits, real_lab_logits, real_proc_logits, gen_diag_logits, gen_drug_logits, gen_lab_logits, gen_proc_logits, h, v_gen = model(diag_seq, drug_seq, lab_seq, proc_seq, diag_length, None)
                elif args.model == 'LSTM-MLP':
                    real_diag_logits, real_drug_logits, real_lab_logits, real_proc_logits, _, _, _, _, h, _ = model(diag_seq, drug_seq, lab_seq, proc_seq, diag_length)
                elif args.model == 'synTEG':
                    real_diag_logits, real_drug_logits, real_lab_logits, real_proc_logits, gen_diag_logits, gen_drug_logits, gen_lab_logits, gen_proc_logits, h, v_gen = model(diag_seq, drug_seq, lab_seq, proc_seq, diag_length)
                elif args.model == 'TWIN':
                    real_diag_logits, real_drug_logits, real_lab_logits, real_proc_logits, miu, logvar, time_pred = model(diag_seq, drug_seq, lab_seq, proc_seq, diag_length, time_step)
                elif args.model == 'LSTM-TabDDPM' or args.model == 'LSTM-Meddiff' or args.model == 'LSTM-ScoEHR':
                    real_diag_logits, real_drug_logits, real_lab_logits, real_proc_logits, gen_diag_logits, gen_drug_logits, gen_lab_logits, gen_proc_logits, h, v_gen, added_noise, learned_noise = model(diag_seq, drug_seq, lab_seq, proc_seq, diag_length)
                elif args.model == 'EVA':
                    real_diag_logits, real_drug_logits, real_lab_logits, real_proc_logits, miu, logvar  = model(diag_seq, drug_seq, lab_seq, proc_seq, diag_length)
                else:
                    raise ValueError('Invalid model')

                if args.name == 'GAN' or args.name == 'WGAN-GP':

                    if args.name == 'GAN':
                        optim.zero_grad()
                        generator_optim.zero_grad()

                        generator_discriminator_out = discriminator(v_gen[:, 0:-1, :])
                        #generator_discriminator_out = discriminator(v_gen)
                        true_labels = torch.ones_like(generator_discriminator_out, device=device)
                        generator_loss = CE(generator_discriminator_out, true_labels) * args.lambda_gen
                        generator_loss.backward(retain_graph=True)
                        generator_optim.step()

                        discriminator_optim.zero_grad()
                        true_discriminator_out = discriminator(h[:, 1:, :])
                        # true_discriminator_out = discriminator(h)
                        true_discriminator_loss = CE(true_discriminator_out, true_labels)

                        generator_discriminator_out = discriminator(v_gen.detach()[:, 0:-1, :])
                        # generator_discriminator_out = discriminator(v_gen.detach())
                        generator_discriminator_loss = CE(generator_discriminator_out, 1 - true_labels)
                        discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2 * args.lambda_gen
                        discriminator_loss.backward(retain_graph=True)
                        discriminator_optim.step()

                    elif args.name == 'WGAN-GP':
                        current_step += 1
                        # Update critic
                        discriminator_optim.zero_grad()

                        # Wasserstein critic loss for real and fake data
                        true_critic_out = discriminator(h)
                        fake_critic_out = discriminator(v_gen.detach())
                        critic_loss = fake_critic_out.mean() - true_critic_out.mean()

                        # Compute and apply gradient penalty
                        gradient_penalty = compute_gradient_penalty(discriminator, h, v_gen)
                        critic_loss_total = critic_loss + args.lambda_gp * gradient_penalty
                        critic_loss_total.backward(retain_graph=True)  # No need for retain_graph=True here
                        discriminator_optim.step()

                        # Update generator less frequently
                        if current_step % args.critic_iterations == 0:
                            generator_optim.zero_grad()
                            # Generator tries to maximize the critic's score for its fake data
                            fake_critic_out = discriminator(v_gen)
                            generator_loss = -fake_critic_out.mean()
                            generator_loss.backward(retain_graph=True)  # No need for retain_graph=True here
                            generator_optim.step()

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

                    loss = args.lambda_ce * (CE(real_diag_logits, multihot_diag) + CE(real_drug_logits, multihot_drug) + CE(real_lab_logits, multihot_lab) + CE(real_proc_logits, multihot_proc))/4 + 1e-10
                    loss.backward()
                    # fake_loss = (CE(gen_diag_logits, multihot_diag) + CE(gen_drug_logits, multihot_drug) + CE(gen_lab_logits.detach(), multihot_lab) + CE(gen_proc_logits, multihot_proc))/4 + 1e-10
                    # fake_loss.backward(retain_graph=True)
                    optim.step()

                    if not args.name == 'WGAN-GP':
                        prediction_module_loss += (loss.item() / visit_timegaps.size(0)) * args.batch_size
                        generation_module_loss += (generator_loss.item() + discriminator_loss.item() / visit_timegaps.size(0)) * args.batch_size
                    else:
                        prediction_module_loss += (loss.item() / visit_timegaps.size(0)) * args.batch_size
                        generation_module_loss += (critic_loss_total.item() / visit_timegaps.size(0)) * args.batch_size

                    if args.max_grad_norm > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)


                    if (global_step + 1) % args.log_interval == 0:
                        avg_prediction_module_loss = prediction_module_loss / args.log_interval
                        avg_generation_module_loss = generation_module_loss / args.log_interval

                        print(
                            '| step {:5} | prediction module loss {:7.4f} | generation module loss {:7.4f} |'.format(
                                global_step, avg_prediction_module_loss, avg_generation_module_loss))

                        prediction_module_loss, generation_module_loss = 0.0, 0.0
                        start_time = time.time()
                    global_step += 1
                elif args.name == 'MLP':
                    optim.zero_grad()
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

                    loss = args.lambda_ce * (CE(real_diag_logits, multihot_diag) + CE(real_drug_logits, multihot_drug) + CE(real_lab_logits, multihot_lab) + CE(real_proc_logits, multihot_proc))/4 + 1e-10
                    loss.backward(retain_graph=True)
                    optim.step()
                    prediction_module_loss += (loss.item() / visit_timegaps.size(0)) * args.batch_size

                    if args.max_grad_norm > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    if (global_step + 1) % args.log_interval == 0:
                        avg_prediction_module_loss = prediction_module_loss / args.log_interval

                        print(
                            '| step {:5} | prediction module loss {:7.4f} |'.format(
                                global_step, avg_prediction_module_loss))

                        prediction_module_loss = 0.0
                        start_time = time.time()
                    global_step += 1
                elif args.name == 'VAE':
                    optim.zero_grad()
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

                    if args.model == 'TWIN' and time_pred is not None:
                        sequence_range = torch.arange(args.max_len, device=diag_seq.device).expand(diag_seq.size(0),
                                                                                                   args.max_len)
                        expanded_lengths = diag_length.unsqueeze(-1).expand_as(sequence_range)
                        length_mask = (sequence_range < expanded_lengths).float()
                        c = MSE(time_step * length_mask, time_pred * length_mask)
                    else :
                        c = 0

                    a = args.lambda_ce * (CE(real_diag_logits, multihot_diag) + CE(real_drug_logits, multihot_drug) + CE(real_lab_logits, multihot_lab) + CE(real_proc_logits, multihot_proc))/4 + 1e-10
                    b = args.lambda_gen * kl_loss(miu, logvar)
                    loss = a + b + c
                    loss.backward()
                    optim.step()

                    prediction_module_loss += (a.item() / visit_timegaps.size(0)) * args.batch_size
                    generation_module_loss += (b.item() / visit_timegaps.size(0)) * args.batch_size
                    time_pred_loss += c

                    if args.max_grad_norm > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    if (global_step + 1) % args.log_interval == 0:
                        avg_prediction_module_loss = prediction_module_loss / args.log_interval
                        avg_generation_module_loss = generation_module_loss / args.log_interval
                        avg_time_pred_loss = time_pred_loss / args.log_interval

                        print(
                            '| step {:5} | prediction module loss {:7.4f} | generation module loss {:7.4f} |'.format(
                                global_step, avg_prediction_module_loss, avg_generation_module_loss))
                        print('| step {:5} | time prediction loss {:7.4f} |'.format(global_step, avg_time_pred_loss))

                        prediction_module_loss, generation_module_loss = 0.0, 0.0
                        time_pred_loss = 0.0
                        start_time = time.time()
                    global_step += 1
                elif args.name == 'Diff':
                    optim.zero_grad()
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

                    a = args.lambda_ce * (CE(real_diag_logits, multihot_diag) + CE(real_drug_logits, multihot_drug) + CE(real_lab_logits, multihot_lab) + CE(real_proc_logits, multihot_proc))/4 + 1e-10
                    # a = args.lambda_ce * (CE(gen_diag_logits, multihot_diag) + CE(gen_drug_logits, multihot_drug) + CE(gen_lab_logits, multihot_lab) + CE(gen_proc_logits, multihot_proc))/4 + 1e-10
                    b = args.lambda_gen * (MSE(added_noise,learned_noise))
                    loss = a + b
                    loss.backward()
                    optim.step()

                    prediction_module_loss += (a.item() / visit_timegaps.size(0)) * args.batch_size
                    generation_module_loss += (b.item() / visit_timegaps.size(0)) * args.batch_size

                    if args.max_grad_norm > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

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

            with torch.no_grad():
                # train_res = evaluator.eval(train_dataloader, [diag_pad_id, drug_pad_id, lab_pad_id, proc_pad_id],
                #                      [diag_pad_id, drug_nan_id, lab_nan_id, proc_nan_id],
                #                      ['diag', 'drug', 'lab', 'proc'], ['lpl', 'mpl'])
                val_res = evaluator.eval(dev_dataloader, [diag_pad_id, drug_pad_id, lab_pad_id, proc_pad_id],
                                   [diag_pad_id, drug_nan_id, lab_nan_id, proc_nan_id],
                                   ['diag', 'drug', 'lab', 'proc'], ['lpl', 'mpl'])
                test_res = evaluator.eval(test_dataloader, [diag_pad_id, drug_pad_id, lab_pad_id, proc_pad_id],
                                    [diag_pad_id, drug_nan_id, lab_nan_id, proc_nan_id],
                                    ['diag', 'drug', 'lab', 'proc'], ['lpl', 'mpl'])
                # train_diag_lpl, tran_drug_lpl, train_lab_lpl, train_proc_lpl = train_res['lpl_diag'], train_res[
                #     'lpl_drug'], train_res['lpl_lab'], train_res['lpl_proc']
                val_diag_lpl, val_drug_lpl, val_lab_lpl, val_proc_lpl = val_res['lpl_diag'], val_res['lpl_drug'], \
                val_res['lpl_lab'], val_res['lpl_proc']
                test_diag_lpl, test_drug_lpl, test_lab_lpl, test_proc_lpl = test_res['lpl_diag'], test_res[
                    'lpl_drug'], test_res['lpl_lab'], test_res['lpl_proc']

                # train_diag_mpl, tran_drug_mpl, train_lab_mpl, train_proc_mpl = train_res['mpl_diag'], train_res[
                #     'mpl_drug'], train_res['mpl_lab'], train_res['mpl_proc']
                val_diag_mpl, val_drug_mpl, val_lab_mpl, val_proc_mpl = val_res['mpl_diag'], val_res['mpl_drug'], \
                val_res['mpl_lab'], val_res['mpl_proc']
                test_diag_mpl, test_drug_mpl, test_lab_mpl, test_proc_mpl = test_res['mpl_diag'], test_res[
                    'mpl_drug'], test_res['mpl_lab'], test_res['mpl_proc']

                choosing_statistic = np.median(
                    [val_diag_lpl, val_drug_lpl, val_lab_lpl, val_proc_lpl, val_diag_mpl, val_drug_mpl, val_lab_mpl,
                     val_proc_mpl])

                print('-' * 71)
                print('Epoch: {:5}'.format(epoch_id))
                print('Time: {:5.2f}s'.format(time.time() - start_time))
                train_diag_lpl, tran_drug_lpl, train_lab_lpl, train_proc_lpl = 0, 0, 0, 0
                train_diag_mpl, tran_drug_mpl, train_lab_mpl, train_proc_mpl = 0, 0, 0, 0
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
    subset = False
    seeds = [111]
    save_path = './saved_rebuttal_Q1_'
    # model_names = ['LSTM-Meddiff', 'LSTM-ScoEHR']
    model_names = ['TWIN']
    # model_names = ['LSTM-MLP', 'LSTM-medGAN', 'synTEG', 'TWIN', 'LSTM-TabDDPM', 'EVA']
    save_dirs = [save_path+name+'/' for name in model_names]
    datas = ['mimic']
    max_lens = [20]
    max_nums = [10]
    for mode in modes:
        for seed in seeds:
            for model_name, save_dir in zip(model_names, save_dirs):
                for data, max_len, max_num in zip(datas, max_lens, max_nums):
                    if model_name in ['LSTM-medGAN']:
                        model_type = 'GAN'
                        main(seed, model_type, model_name, data, max_len, max_num, save_dir, mode, short_ICD, subset)
                    elif model_name in ['LSTM-MLP']:
                        model_type = 'MLP'
                        main(seed, model_type, model_name, data, max_len, max_num, save_dir, mode, short_ICD, subset)
                    elif model_name in ['synTEG']:
                        model_type = 'WGAN-GP'
                        main(seed, model_type, model_name, data, max_len, max_num, save_dir, mode, short_ICD, subset)
                    elif model_name in ['TWIN']:
                        model_type = 'VAE'
                        main(seed, model_type, model_name, data, max_len, max_num, save_dir, mode, short_ICD, subset)
                    elif model_name in ['LSTM-TabDDPM']:
                        model_type = 'Diff'
                        main(seed, model_type, model_name, data, max_len, max_num, save_dir, mode, short_ICD, subset)
                    elif model_name in ['EVA']:
                        model_type = 'VAE'
                        main(seed, model_type, model_name, data, max_len, max_num, save_dir, mode, short_ICD, subset)
                    elif model_name in ['LSTM-Meddiff']:
                        model_type = 'Diff'
                        main(seed, model_type, model_name, data, max_len, max_num, save_dir, mode, short_ICD, subset)
                    elif model_name in ['LSTM-ScoEHR']:
                        model_type = 'Diff'
                        main(seed, model_type, model_name, data, max_len, max_num, save_dir, mode, short_ICD, subset)