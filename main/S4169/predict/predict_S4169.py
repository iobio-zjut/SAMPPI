# -*- coding: utf-8 -*-
import os
import pandas as pd
import torch
from numpy import *
import numpy as np
import random
from model_S4169 import *
import pickle


class feature(object):
    def __init__(self, seq):
        self.seq = seq

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


def all_feature(ls, names, mutations, variable_name, sever_dir):
    all = []
    index = 0

    base_ESM2_dir = f'{sever_dir}/data/S4169_esm2_emb'
    pkl_dir = os.path.join(base_ESM2_dir, variable_name)

    base_prott5_dir = f'{sever_dir}/data/S4169_prott5'
    prott5_pkl_dir = os.path.join(base_prott5_dir, variable_name)

    for s in ls:
        if s == s:  # 检查序列是否为 NaN

            # f1 ONE-HOT [L,20]
            f1 = feature(s).seq2onehot()
            # 构建 PSSM 文件名
            #f3 ESM2 [L,1280]
            f2 = None
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
                        f2 = esm2_data[key_name]
                    else:
                        print(f"Key '{key_name}' not found in {pkl_file_name}")
            else:
                print(f"PKL file '{pkl_file_name}' not found in directory {pkl_dir}")
            if not isinstance(f2, np.ndarray):
                f2 = np.array(f2)

            #f3 Prott5 [L,1024]
            f3 = None  # [L,1024]
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
                        f3 = prott5_data[prott5_key_name]
                    else:
                        print(f"Key '{prott5_key_name}' not found in {prott5_pkl_file_name}")
            else:
                print(f"prott5_PKL file '{prott5_pkl_file_name}' not found in directory {prott5_pkl_dir}")
            if not isinstance(f3, np.ndarray):
                f3 = np.array(f3)

            f4 = feature(s).seq2Blosum62(f'{sever_dir}/data/blosum62.txt')  # [L,23]

            f5 = feature(s).meiler(f'{sever_dir}/data/Meiler.csv')

            # 合并 f1 和 f2
            f = np.concatenate((f1, f2, f3, f4, f5), axis=1)
            all.append(f)
            index += 1
        else:
            all.append(s)  # 如果 s 为 NaN，直接将其附加到 all
            index += 1

    return all


def is_valid_chain(chain):
    return isinstance(chain, np.ndarray) and chain.ndim == 2 and not np.isnan(chain).all()

def concatenate_multiple_chains(base_list, *chains):
    """
    拼接任意数量的链，中间用全零向量分隔，忽略无效链（如 NaN 或非法类型）。
    """
    valid_chains = [chain for chain in chains if is_valid_chain(chain)]

    if not valid_chains:
        return  # 没有有效链，直接跳过

    try:
        sep_token = np.zeros((1, valid_chains[0].shape[1]), dtype=np.float32)  # 用于分隔的 token
    except Exception as e:
        print(f"[分隔符构造失败] 可能是链为空或格式错误: {e}")
        return

    combined = valid_chains[0]
    for chain in valid_chains[1:]:
        combined = np.concatenate((combined, sep_token, chain), axis=0)

    base_list.append(combined)



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

    base_dir = "./SAMPPI"

    for n in range(1,11):
        """Load preprocessed data."""
        csv = pd.read_csv(f'{base_dir}/data/test_fold/S4169/S4169_test_fold{n}.csv',
                          usecols=['PDB','mutation_clean','R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'L1', 'L2', 'L3', 'Rm1', 'Rm2', 'Rm3', 'Rm4', 'Rm5', 'Rm6', 'Lm1', 'Lm2', 'Lm3', 'DDG'])

        names = csv['PDB'].tolist()
        mutations = csv['mutation_clean'].tolist()

        R1 = csv['R1'].tolist()
        R2 = csv['R2'].tolist()
        R3 = csv['R3'].tolist()
        R4 = csv['R4'].tolist()
        R5 = csv['R5'].tolist()
        R6 = csv['R6'].tolist()
        L1 = csv['L1'].tolist()
        L2 = csv['L2'].tolist()
        L3 = csv['L3'].tolist()
        Rm1 = csv['Rm1'].tolist()
        Rm2 = csv['Rm2'].tolist()
        Rm3 = csv['Rm3'].tolist()
        Rm4 = csv['Rm4'].tolist()
        Rm5 = csv['Rm5'].tolist()
        Rm6 = csv['Rm6'].tolist()
        Lm1 = csv['Lm1'].tolist()
        Lm2 = csv['Lm2'].tolist()
        Lm3 = csv['Lm3'].tolist()
        labels = csv['DDG'].tolist()

        r1 = all_feature(R1, names, mutations, "R1", base_dir)
        r2 = all_feature(R2, names, mutations, "R2", base_dir)
        r3 = all_feature(R3, names, mutations, "R3", base_dir)
        r4 = all_feature(R4, names, mutations, "R4", base_dir)
        r5 = all_feature(R5, names, mutations, "R5", base_dir)
        r6 = all_feature(R6, names, mutations, "R6", base_dir)

        l1 = all_feature(L1, names, mutations, "L1", base_dir)
        l2 = all_feature(L2, names, mutations, "L2", base_dir)
        l3 = all_feature(L3, names, mutations, "L3", base_dir)

        rm1 = all_feature(Rm1, names, mutations, "Rm1", base_dir)
        rm2 = all_feature(Rm2, names, mutations, "Rm2", base_dir)
        rm3 = all_feature(Rm3, names, mutations, "Rm3", base_dir)
        rm4 = all_feature(Rm4, names, mutations, "Rm4", base_dir)
        rm5 = all_feature(Rm5, names, mutations, "Rm5", base_dir)
        rm6 = all_feature(Rm6, names, mutations, "Rm6", base_dir)

        lm1 = all_feature(Lm1, names, mutations, "Lm1", base_dir)
        lm2 = all_feature(Lm2, names, mutations, "Lm2", base_dir)
        lm3 = all_feature(Lm3, names, mutations, "Lm3", base_dir)

        interactions = np.array(labels)

        recpetor = []
        ligand = []
        recpetor_mut = []
        ligand_mut = []

        for i in range(len(R1)):
            concatenate_multiple_chains(recpetor, r1[i], r2[i], r3[i], r4[i], r5[i], r6[i])
            concatenate_multiple_chains(ligand, l1[i], l2[i], l3[i])
            concatenate_multiple_chains(recpetor_mut, rm1[i], rm2[i], rm3[i], rm4[i], rm5[i], rm6[i])
            concatenate_multiple_chains(ligand_mut, lm1[i], lm2[i], lm3[i])

        print("interactions len ", len(interactions))

        antigens_test = np.array(ligand)
        antibodies_test = np.array(recpetor)
        antigens_mut_test = np.array(ligand_mut)
        antibodies_mut_test = np.array(recpetor_mut)

        interactions_test = np.array(interactions)

        """ Combine test data """
        dataset_test = list(zip(antigens_test, antibodies_test, antigens_mut_test, antibodies_mut_test, interactions_test))

        """Start prediction"""
        print(f'Predicting fold{n}')

        """ create model ,trainer and tester """
        protein_dim = 2354
        hid_dim = 256
        n_layers = 3  # 3
        n_heads = 8
        pf_dim = 64
        dropout = 0.1
        kernel_size = 7  # 7

        encoder_ab = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout, device)
        encoder_ag = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout, device)

        decoder_ab = Decoder(protein_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention,
                             PositionwiseFeedforward, dropout, device)
        decoder_ag = Decoder(protein_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention,
                             PositionwiseFeedforward, dropout, device)

        # 初始化记录每个模型预测结果的列表

        pccs_list, mae_list, mse_list, rmse_list, r2_list, loss_list = [], [], [], [], [], []
        y_test_predict_all = []  # 存储所有模型的预测分布

        model_path = f'{base_dir}/save_models/S4169/model_{n}'  # 动态路径
        print(f"Loading model from: {model_path}")

        # 初始化模型
        model = Predictor(encoder_ab, encoder_ag, decoder_ab, decoder_ag, device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
    
        # 创建测试器
        tester = Tester(model, device,protein_dim)
        # 获取预测结果
        pccs_test,pccs_test_r, mae_test, mse_test, rmse_test, r2_test, loss_test_fold, y_test_true, y_test_predict = tester.test(dataset_test)
        np.set_printoptions(suppress=True, precision=4)
        print("True: ", y_test_true)
        print("Predict: ", y_test_predict)
        # 存储该模型的预测分布
        y_test_predict_all.append(y_test_predict)

        # 打印该模型的预测结果
        print(f"Model results:")
        print("pccs: ", pccs_test.item())
        print("pccs_r: ", pccs_test_r.item())
        print("mae: ", mae_test)
        print("mse: ", mse_test)
        print("rmse: ", rmse_test)
        print("r2: ", r2_test)

        # 定义保存路径
        save_dir = f'{base_dir}/data/test_fold/S4169'
        os.makedirs(save_dir, exist_ok=True)
        # 保存文件
        with open(os.path.join(save_dir, f'fold{n}_result.txt'), "w") as f:
            f.write("true: " + np.array2string(y_test_true, precision=4, suppress_small=True) + "\n")
            f.write("predict: " + np.array2string(y_test_predict, precision=4, suppress_small=True) + "\n")
            f.write(f"pccs: {pccs_test.item()}\n")
            f.write(f"pccs: {pccs_test_r.item()}\n")
            f.write(f"mae: {mae_test}\n")
            f.write(f"mse: {mse_test}\n")
            f.write(f"rmse: {rmse_test}\n")
            f.write(f"r2: {r2_test}\n")



