# -*- coding: utf-8 -*-
import os
import pandas as pd
import torch
from numpy import *
import numpy as np
import random
from model_S1131_v56 import *
import timeit
from sklearn.model_selection import KFold
import pickle


class feature(object):
    def __init__(self, seq):
        self.seq = seq
        # self.length = max

    def seq2onehot(self):
        aas = {'X': 0, 'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5,
               'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
               'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15,
               'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20}
        seq_onehot = np.zeros((len(self.seq), len(aas)))
        for i, aa in enumerate(self.seq[:]):
            if aa not in aas:
                aa = 'X'
            seq_onehot[i, (aas[aa])] = 1
        # seq_onehot = ''.join(seq_onehot)
        seq_onehot = seq_onehot[:, 1:]  # except X
        return seq_onehot

    def seq2Blosum62(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            firline = f.readline().strip().split(' ')
            aaindex = [aa for aa in firline if aa != '' and aa != '*']
            lenth = len(aaindex)
            blos62 = np.zeros((lenth, lenth))
            index = 0
            for l in f:
                if index == 23:
                    break
                newl = l.strip().split(' ')
                line = [num for num in newl if num != '' and num != "*" and not num.isalpha()]
                blos62[index] = line[:len(line) - 1]
                index += 1
            seqlen = len(self.seq)
            finblos62 = np.zeros((seqlen, 23))
            for i, se in enumerate(self.seq):
                ind = aaindex.index(se)
                finblos62[i] = blos62[:, ind]
            return finblos62

    def rsa_asa(self, sec_csvfile_path, feature_columns=[" rsa", " asa", " p[q3_H]", " p[q3_E]", " p[q3_C]"]):

        # 读取 CSV 文件
        if os.path.exists(sec_csvfile_path):
            df = pd.read_csv(sec_csvfile_path, sep=None, engine='python')  # 以制表符分隔
            f5 = df[feature_columns].to_numpy(dtype=np.float32)  # 转换为 numpy 数组
            return f5
        else:
            print(f"文件 {sec_csvfile_path} 不存在！")

    def meiler(self, csv_file):
        df = pd.read_csv(csv_file)
        df.set_index('residue', inplace=True)
        feature_names = ['steric_parameter', 'polarizability', 'volume',
                         'hydrophobicity', 'isoelectric_pt', 'helix_prob', 'sheet_prob']
        residue_map = {
            'A': 'ALA', 'G': 'GLY', 'V': 'VAL', 'L': 'LEU', 'I': 'ILE',
            'F': 'PHE', 'Y': 'TYR', 'W': 'TRP', 'T': 'THR', 'S': 'SER',
            'R': 'ARG', 'K': 'LYS', 'H': 'HIS', 'D': 'ASP', 'E': 'GLU',
            'N': 'ASN', 'Q': 'GLN', 'M': 'MET', 'P': 'PRO', 'C': 'CYS'
        }
        n = len(self.seq)
        m = len(feature_names)
        feature_matrix = np.zeros((n, m))
        for i, se in enumerate(self.seq):
            try:
                # 检查是否为未知残基
                if se not in residue_map:
                    # feature_matrix[i, :] = np.zeros(len(feature_names))
                    raise ValueError(f"未知残基 {se} 在序列中")
                three_letter = residue_map[se]
                # 检查CSV文件中是否有该残基的数据
                if three_letter not in df.index:
                    raise ValueError(f"残基 {three_letter} 在CSV文件中未找到")
                # 提取特征
                features = df.loc[three_letter, feature_names]
                feature_matrix[i, :] = features.values
            except ValueError:
                feature_matrix[i, :] = np.zeros(len(feature_names))
        return feature_matrix


def all_feature(ls, names, mutations, variable_name, server_dir):
    all = []
    index = 0

    base_dir = f'{server_dir}/data/S1131_PSSM'
    pssm_dir = os.path.join(base_dir, variable_name)

    base_ESM2_dir = f'{server_dir}/data/S1131_esm2_emb'
    pkl_dir = os.path.join(base_ESM2_dir, variable_name)

    base_sec_dir = f'{server_dir}/data/S1131_secondary_structure'
    sec_dir = os.path.join(base_sec_dir, variable_name)

    base_prott5_dir = f'{server_dir}/data/S1131_prott5'
    prott5_pkl_dir = os.path.join(base_prott5_dir, variable_name)


    for s in ls:
        if s == s:  # 检查序列是否为 NaN

            # f1 ONE-HOT [L,20]
            f1 = feature(s).seq2onehot()
            # 构建 PSSM 文件名

            # f2 PSSM [L,20]
            f2 = None
            pssm_filename = f"{names[index]}_{mutations[index]}_{variable_name}.pssm"
            pssm_filepath = os.path.join(pssm_dir, pssm_filename)
            # 检查 PSSM 文件是否存在
            if os.path.exists(pssm_filepath):
                with open(pssm_filepath, 'r') as inputpssm:
                    count = 0
                    pssm_matrix = []
                    for eachline in inputpssm:
                        count += 1
                        if count <= 3:  # 跳过前 3 行
                            continue
                        if not len(eachline.strip()):  # 结束时为空行
                            break
                        col = eachline.strip()
                        col = col.split(' ')
                        col = [x for x in col if x != '']  # 移除多余空格
                        col = col[2:22]  # 提取 PSSM 列
                        col = [int(x) for x in col]  # 转换为整数
                        pssm_matrix.append(col)
                    f2 = np.array(pssm_matrix)
            else:
                print(f"PSSM file not found: {pssm_filepath}")

            # f3 ESM2 [L,1280]
            f3 = None
            # 构造 ESM2 `pkl` 文件名
            pkl_file_name = f"{names[index]}_{mutations[index]}_{variable_name}_emb.pkl"
            pkl_file_path = os.path.join(pkl_dir, pkl_file_name)
            if os.path.exists(pkl_file_path):
                # 加载 `pkl` 文件
                with open(pkl_file_path, 'rb') as pkl_file:
                    esm2_data = pickle.load(pkl_file)
                    # 使用键名读取特征
                    key_name = f"{names[index]}_{mutations[index]}_{variable_name}"
                    if key_name in esm2_data:
                        f3 = esm2_data[key_name]
                    else:
                        print(f"Key '{key_name}' not found in {pkl_file_name}")
            else:
                print(f"PKL file '{pkl_file_name}' not found in directory {pkl_dir}")
            if not isinstance(f3, np.ndarray):
                f3 = np.array(f3)


            # f4 Prott5 [L,1024]
            f4 = None  # [L,1024]
            if names[index] == "1.00E+96":
                names_key = "1_00E+96"
                prott5_pkl_file_name = f"{names_key}_{mutations[index]}_{variable_name}_prott5.pkl"
            else:
                prott5_pkl_file_name = f"{names[index]}_{mutations[index]}_{variable_name}_prott5.pkl"
            prott5_pkl_file_path = os.path.join(prott5_pkl_dir, prott5_pkl_file_name)
            if os.path.exists(prott5_pkl_file_path):
                # 加载 `pkl` 文件
                with open(prott5_pkl_file_path, 'rb') as prott5_pkl_file:
                    prott5_data = pickle.load(prott5_pkl_file)
                    # 使用键名读取特征
                    if names[index] == "1.00E+96":
                        names_key = "1_00E+96"
                        prott5_key_name = f"{names_key}_{mutations[index]}_{variable_name}"
                    else:
                        prott5_key_name = f"{names[index]}_{mutations[index]}_{variable_name}"
                    if prott5_key_name in prott5_data:
                        f4 = prott5_data[prott5_key_name]
                    else:
                        print(f"Key '{prott5_key_name}' not found in {prott5_pkl_file_name}")
            else:
                print(f"prott5_PKL file '{prott5_pkl_file_name}' not found in directory {prott5_pkl_dir}")
            if not isinstance(f4, np.ndarray):
                f4 = np.array(f4)

            f5 = feature(s).seq2Blosum62(
                f'{server_dir}/data/blosum62.txt')  # [L,23]
            sec_csvfile_name = f'{names[index]}_{mutations[index]}_{variable_name}.csv'
            sec_dir_name = f'{names[index]}_{mutations[index]}_{variable_name}'
            sec_csvfile_path = os.path.join(sec_dir, sec_dir_name, sec_csvfile_name)

            f6 = feature(s).rsa_asa(sec_csvfile_path, feature_columns=[
                " rsa"])

            f7 = feature(s).meiler(f'{server_dir}/data/Meiler.csv')

            # 合并 f1 和 f2
            f = np.concatenate((f1, f2, f3, f4, f5, f6, f7), axis=1)
            all.append(f)
            index += 1
        else:
            all.append(s)  # 如果 s 为 NaN，直接将其附加到 all
            index += 1

    return all



if __name__ == "__main__":
    SEED = 1234
    random.seed(SEED)
    torch.manual_seed(SEED)

    """CPU or GPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    base_dir = './SAMPPI'

    """Load preprocessed data."""
    csv = pd.read_csv(f'{base_dir}/data/S1131.csv',usecols=['PDB','mutation_clean','a','b','a_mut','b_mut', 'ddG'])
    names = csv['PDB'].tolist()
    mutations = csv['mutation_clean'].tolist()
    ab = csv['a'].tolist()
    ag = csv['b'].tolist()
    ab_m = csv['a_mut'].tolist()
    ag_m = csv['b_mut'].tolist()
    labels = csv['ddG'].tolist()

    antibodies = all_feature(ab, names, mutations,"a",base_dir)
    antigens = all_feature(ag, names, mutations,"b",base_dir)
    antibodies_mut = all_feature(ab_m, names, mutations,"a_m",base_dir)
    antigens_mut = all_feature(ag_m, names, mutations,"b_m",base_dir)
    
    interactions = np.array(labels)

    """Start training."""
    print('Training...')

    n_splits = 10
    kf = KFold(n_splits=10, shuffle=True,random_state=SEED)

    i=0
    start_fold = 1  # 设置从第6折开始

    for train_index, val_index in kf.split(interactions):

        i += 1
        if i < start_fold:
            print(f"Skipping Fold {i}...")
            continue

        """ create model ,trainer and tester """
        protein_dim = 2375  # 20+20+1280+7+3+1024+1+23
        hid_dim = 256
        n_layers = 3  #3
        n_heads = 8
        pf_dim = 64
        dropout = 0.1
        batch = 4  # 64
        lr = 1e-4
        weight_decay = 1e-4
        decay_interval = 5
        lr_decay = 0.98
        iteration = 200
        kernel_size = 7  # 7
        best_pearson = -1000

        
        encoder_ab = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout, device)
        encoder_ag = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout, device)

        decoder_ab = Decoder(protein_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)
        decoder_ag = Decoder(protein_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)

        model = Predictor(encoder_ab, encoder_ag, decoder_ab, decoder_ag, device)

        model.to(device)

        trainer = Trainer(model, lr, weight_decay, batch,protein_dim)
        tester = Tester(model,device,protein_dim)

        print('*************************** start training on Fold %s ***************************'%i)

        # 定义目标路径
        output_dir = f'{base_dir}/output/S1131'

        # 如果路径不存在，则创建路径
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 写入文件
        with open(os.path.join(output_dir, 'fold{}.txt'.format(i)), 'w') as file_fold:
            file_fold.write(str(train_index))
            file_fold.write('\n')
            file_fold.write(str(val_index))
            file_fold.write('\n')

        antigens_train, antigens_val = np.array(antigens)[train_index], np.array(antigens)[val_index]  # ag
        antibodies_train, antibodies_val = np.array(antibodies)[train_index], np.array(antibodies)[val_index]  # ab
        antigens_mut_train, antigens_mut_val = np.array(antigens_mut)[train_index], np.array(antigens_mut)[val_index]  # ag_mut
        antibodies_mut_train, antibodies_mut_val = np.array(antibodies_mut)[train_index], np.array(antibodies_mut)[val_index]  # ab_mut
        interactions_train, interactions_val = np.array(interactions)[train_index], np.array(interactions)[val_index]  # Y

        dataset_train = list(zip(antigens_train, antibodies_train, antigens_mut_train, antibodies_mut_train, interactions_train))
        dataset_val = list(zip(antigens_val, antibodies_val, antigens_mut_val, antibodies_mut_val, interactions_val))

        """Output files."""
        file_best_pcc_PCCS =f'{base_dir}/output/S1131/RECORD_{i}.txt'
        file_best_pcc_model =f'{base_dir}/output/S1131/model_{i}'

        os.makedirs(os.path.dirname(file_best_pcc_PCCS), exist_ok=True)
        PCCS = ('Epoch\tTime(sec)\tLoss_train\tLoss_val\tpearson\tpearson_reversed\tMAE\tMSE\tRMSE\tr2')
        print("Score:",PCCS)

        with open(file_best_pcc_PCCS, 'w') as f:
            f.write(PCCS + '\n')

        start = timeit.default_timer()

        for epoch in range(1, iteration + 1):

            print('Epoch:',epoch)
            loss_train_fold, y_train_true, y_train_predict = trainer.train(dataset_train, device)
            pccs_val,pccs_reversed_val, mae_val, mse_val, rmse_val, r2_val, loss_val_fold, y_val_true, y_val_predict = tester.test(dataset_val)

            end = timeit.default_timer()
            time = end - start
            
            if epoch % decay_interval == 0:
                trainer.optimizer.param_groups[0]['lr'] *= lr_decay

            PCCS = [epoch, time, loss_train_fold.tolist(), loss_val_fold.tolist(), pccs_val.tolist(),pccs_reversed_val.tolist(), mae_val, mse_val, rmse_val, r2_val]
            
            if pccs_val.tolist() > best_pearson:
                print(f'Validation loss decreased ({best_pearson:.4f} --> {pccs_val.tolist():.4f}).  Saving model ...')
                tester.save_pccs(PCCS, file_best_pcc_PCCS)
                tester.save_model(model, file_best_pcc_model)
                best_pearson = pccs_val.tolist()

            print('\t'.join(f"{pcc:.4f}" for pcc in PCCS))