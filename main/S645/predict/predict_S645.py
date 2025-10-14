# -*- coding: utf-8 -*-
import os
import pandas as pd
import torch
from numpy import *
import numpy as np
import random
from model_S645 import *
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


def all_feature(ls, names, mutations, ddG, variable_name, sever_dir):
    all = []
    index = 0

    base_dir = f"{sever_dir}/data/M1101_PSSM"
    pssm_dir = os.path.join(base_dir, variable_name)

    base_ESM2_dir = f"{sever_dir}/data/M1101_esm2_emb"
    pkl_dir = os.path.join(base_ESM2_dir, variable_name)

    base_antiberty_dir = f"{sever_dir}/data/M1101_antiberty"
    antiberty_dir = os.path.join(base_antiberty_dir, variable_name)

    base_prott5_dir = f"{sever_dir}/data/M1101_prott5"
    prott5_pkl_dir = os.path.join(base_prott5_dir, variable_name)

    base_sec_dir = f"{sever_dir}/data/M1101_secondary_structure"
    sec_dir = os.path.join(base_sec_dir, variable_name)

    for s in ls:
        if s == s:  # 检查序列是否为 NaN

            f1 = feature(s).seq2onehot()
            # 构建 PSSM 文件名

            # 初始化 f2
            f2 = None
            pssm_filename = f"{names[index]}_{mutations[index]}_{ddG[index]}_{variable_name}.pssm"
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

            # 初始化 f3
            f3 = None
            # 构造 ESM2 `pkl` 文件名
            pkl_file_name = f"{names[index]}_{mutations[index]}_{ddG[index]}_{variable_name}_emb.pkl"
            pkl_file_path = os.path.join(pkl_dir, pkl_file_name)
            if os.path.exists(pkl_file_path):
                # 加载 `pkl` 文件
                with open(pkl_file_path, 'rb') as pkl_file:
                    esm2_data = pickle.load(pkl_file)
                    # 使用键名读取特征
                    key_name = f"{names[index]}_{mutations[index]}_{ddG[index]}_{variable_name}"
                    if key_name in esm2_data:
                        f3 = esm2_data[key_name]
                    else:
                        print(f"Key '{key_name}' not found in {pkl_file_name}")
            else:
                print(f"PKL file '{pkl_file_name}' not found in directory {pkl_dir}")
            if not isinstance(f3, np.ndarray):
                f3 = np.array(f3)

            f6 = feature(s).seq2Blosum62(f"{sever_dir}/data/blosum62.txt")  # [L,23]

            ##secondary_struct
            sec_csvfile_name = f'{names[index]}_{mutations[index]}_{ddG[index]}_{variable_name}.csv'
            sec_dir_name = f'{names[index]}_{mutations[index]}_{ddG[index]}_{variable_name}'
            sec_csvfile_path = os.path.join(sec_dir, sec_dir_name, sec_csvfile_name)
            f7 = feature(s).rsa_asa(sec_csvfile_path, [" rsa"])

            f8 = feature(s).meiler(f"{sever_dir}/data/Meiler.csv")

            if variable_name in ['ab_h', 'ab_l', 'ab_h_m', 'ab_l_m']:
                f4 = None  # [L,512]
                antiberty_file_name = f"{names[index]}_{mutations[index]}_{ddG[index]}_{variable_name}_antiberty.pkl"
                antiberty_file_path = os.path.join(antiberty_dir, antiberty_file_name)
                # 判断是否存在文件
                if os.path.exists(antiberty_file_path):
                    with open(antiberty_file_path, 'rb') as antiberty_file:
                        embedding = pickle.load(antiberty_file)
                        if isinstance(embedding, torch.Tensor):
                            f4 = embedding.detach().cpu().numpy()
                            f4 = f4[1:-1]
                        else:
                            print(f"[Warning] Unexpected data type in {antiberty_file_name}: {type(embedding)}")
                else:
                    print(f"[Missing] PKL file '{antiberty_file_name}' not found in {antiberty_dir}")
                if not isinstance(f4, np.ndarray):
                    f4 = np.array(f4)

                f = np.concatenate((f1, f2, f3, f4, f6, f7, f8), axis=1)
                all.append(f)
                index += 1

            elif variable_name in ['ag_a', 'ag_b', 'ag_a_m', 'ag_b_m']:
                f5 = None  # [L,1024]
                prott5_pkl_file_name = f"{names[index]}_{mutations[index]}_{variable_name}_prott5.pkl"
                prott5_pkl_file_path = os.path.join(prott5_pkl_dir, prott5_pkl_file_name)
                if os.path.exists(prott5_pkl_file_path):
                    # 加载 `pkl` 文件
                    with open(prott5_pkl_file_path, 'rb') as prott5_pkl_file:
                        prott5_data = pickle.load(prott5_pkl_file)
                        # 使用键名读取特征
                        prott5_key_name = f"{names[index]}_{mutations[index]}_{variable_name}"
                        if prott5_key_name in prott5_data:
                            f5 = prott5_data[prott5_key_name]
                        else:
                            print(f"Key '{prott5_key_name}' not found in {prott5_pkl_file_name}")
                else:
                    print(f"prott5_PKL file '{prott5_pkl_file_name}' not found in directory {prott5_pkl_dir}")
                if not isinstance(f5, np.ndarray):
                    f5 = np.array(f5)

                f = np.concatenate((f1, f2, f3, f5, f6, f7, f8), axis=1)
                all.append(f)
                index += 1
        else:
            all.append(s)  # 如果 s 为 NaN，直接将其附加到 all
            index += 1

    return all


def concatenate_or_append(base_list, low_seq, high_seq):
    if isinstance(high_seq, float):  # 如果是浮动类型，直接添加低序列
        base_list.append(low_seq)
    else:  # 否则拼接低序列和高序列
        base_list.append(np.concatenate((low_seq, high_seq), axis=0))



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
        csv = pd.read_csv(f'{base_dir}/data/test_fold/S645/S645_test_fold{n}.csv',
                          usecols=['PDB','mutation_clean','antibody_light_seq','antibody_heavy_seq','antigen_a_seq','antigen_b_seq','antibody_light_seq_mut','antibody_heavy_seq_mut','antigen_a_seq_mut','antigen_b_seq_mut', 'ddG'])
        names = csv['PDB'].tolist()
        mutations = csv['mutation_clean'].tolist()
        abls = csv['antibody_light_seq'].tolist()
        abhs = csv['antibody_heavy_seq'].tolist()
        agas = csv['antigen_a_seq'].tolist()
        agbs = csv['antigen_b_seq'].tolist()
        abls_m = csv['antibody_light_seq_mut'].tolist()
        abhs_m = csv['antibody_heavy_seq_mut'].tolist()
        agas_m = csv['antigen_a_seq_mut'].tolist()
        agbs_m = csv['antigen_b_seq_mut'].tolist()
        labels = csv['ddG'].tolist()

        antibodies_l = all_feature(abls, names, mutations, labels,"ab_l")
        antibodies_h = all_feature(abhs, names, mutations,labels, "ab_h")
        antigens_a = all_feature(agas, names, mutations,labels, "ag_a")
        antigens_b = all_feature(agbs, names, mutations,labels, "ag_b")
        antibodies_l_mut = all_feature(abls_m, names, mutations,labels, "ab_l_m")
        antibodies_h_mut = all_feature(abhs_m, names, mutations,labels, "ab_h_m")
        antigens_a_mut = all_feature(agas_m, names, mutations,labels, "ag_a_m")
        antigens_b_mut = all_feature(agbs_m, names, mutations,labels, "ag_b_m")

        antibodies = []  # 1101
        antigens = []
        antibodies_mut = []
        antigens_mut = []

        for i in range(len(antigens_a)):
            concatenate_or_append(antibodies, antibodies_l[i], antibodies_h[i])
            concatenate_or_append(antigens, antigens_a[i], antigens_b[i])
            concatenate_or_append(antibodies_mut, antibodies_l_mut[i], antibodies_h_mut[i])
            concatenate_or_append(antigens_mut, antigens_a_mut[i], antigens_b_mut[i])
        interactions = np.array(labels)
        print("interactions len ", len(interactions))

        antigens_test = np.array(antigens)
        antibodies_test = np.array(antibodies)
        antigens_mut_test = np.array(antigens_mut)
        antibodies_mut_test = np.array(antibodies_mut)
        interactions_test = np.array(interactions)

        """ Combine test data """
        dataset_test = list(zip(antigens_test, antibodies_test, antigens_mut_test, antibodies_mut_test, interactions_test))

        """Start prediction"""
        print(f'Predicting fold{n}')

        """ create model ,trainer and tester """
        ab_dim = 1863
        ag_dim = 2375
        hid_dim = 256
        n_layers = 3  # 3
        n_heads = 8
        pf_dim = 64
        dropout = 0.1
        kernel_size = 7  # 7

        encoder_ab = Encoder(ab_dim, hid_dim, n_layers, kernel_size, dropout, device)
        encoder_ag = Encoder(ag_dim, hid_dim, n_layers, kernel_size, dropout, device)

        decoder_ab = Decoder(ab_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention,
                             PositionwiseFeedforward, dropout, device)
        decoder_ag = Decoder(ag_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention,
                             PositionwiseFeedforward, dropout, device)

        # 初始化记录每个模型预测结果的列表

        pccs_list, mae_list, mse_list, rmse_list, r2_list, loss_list = [], [], [], [], [], []
        y_test_predict_all = []  # 存储所有模型的预测分布

        model_path = f'{base_dir}/save_models/S645/model_{n}'  # 动态路径
        print(f"Loading model from: {model_path}")

        # 初始化模型
        model = Predictor(encoder_ab, encoder_ag, decoder_ab, decoder_ag, device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
    
        # 创建测试器
        tester = Tester(model, device, ab_dim, ag_dim)

        print(f"Length of dataset_test: {len(dataset_test)}")
        # 获取预测结果
        pccs_test, mae_test, mse_test, rmse_test, r2_test, loss_test_fold, y_test_true, y_test_predict = tester.test(
            dataset_test)

        # 存储该模型的预测分布
        y_test_predict_all.append(y_test_predict)

        # 打印该模型的预测结果
        print(f"Model results:")
        print("pccs: ", pccs_test.item())
        print("mae: ", mae_test)
        print("mse: ", mse_test)
        print("rmse: ", rmse_test)
        print("r2: ", r2_test)

        # 定义保存路径
        save_dir = f'{base_dir}/output/S645'
        # 保存文件
        with open(os.path.join(save_dir, f'fold{n}_result.txt'), "w") as f:
            f.write(f"true: {y_test_true}\n")
            f.write(f"predict: {y_test_predict}\n")
            f.write(f"pccs: {pccs_test.item()}\n")
            f.write(f"mae: {mae_test}\n")
            f.write(f"mse: {mse_test}\n")
            f.write(f"rmse: {rmse_test}\n")
            f.write(f"r2: {r2_test}\n")



