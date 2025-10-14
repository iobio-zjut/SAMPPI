# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import torch
import time
import pickle
import os
import glob

from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer

def embedding_generate(esm_model_path, fasta, embedding_result, nogpu):
    esm_model, alphabet = pretrained.load_model_and_alphabet(esm_model_path)
    esm_model.eval()
    if isinstance(esm_model, MSATransformer):
        raise ValueError(
            "This script currently does not handle models with MSA input (MSA Transformer)."
        )

    if torch.cuda.is_available() and not nogpu:
        esm_model = esm_model.cuda()
        print("Transferred model to GPU")

    dataset = FastaBatchedDataset.from_file(fasta)
    batches = dataset.get_batch_indices(16384, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(5000), batch_sampler=batches
    )
    print(f"Read {fasta} with {len(dataset)} sequences")

    embedding_result_dic = {}
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )
            if torch.cuda.is_available() and not nogpu:
               toks = toks.to(device="cuda", non_blocking=True)

            out = esm_model(toks, repr_layers=[33], return_contacts=False)["representations"][33]

            for i, label in enumerate(labels):
                #get mean embedding
                esm_embedding = out[i, 1 : len(strs[i]) + 1].clone().cpu()
                embedding_result_dic[label]=esm_embedding
                print('esm_embedding',esm_embedding.shape,type(esm_embedding))
        with open(embedding_result, 'wb') as handle:
            pickle.dump(embedding_result_dic, handle, protocol=4)


def is_nan_fasta(fasta_file):
    """检查 fasta 文件内容是否为 'nan'"""
    with open(fasta_file, 'r') as file:
        lines = file.readlines()
        # 检查第二行是否为 'nan'
        return len(lines) > 1 and lines[1].strip().lower() == 'nan'

def create_empty_pkl(output_file):
    """创建一个空的 .pkl 文件"""
    with open(output_file, 'wb') as f:
        pickle.dump({}, f)  # 空字典作为内容
    print(f"Created empty .pkl file: {output_file}")


# 处理多个目录的函数
def process_multiple_directories(base_fasta_dir, base_output_dir, suffix_list):
    # 开始计时
    time_start = time.time()

    for suffix in suffix_list:
        # 动态构建输入路径和输出路径
        fasta_dir = os.path.join(base_fasta_dir, suffix)
        output_dir = os.path.join(base_output_dir, suffix)

        # 检查并创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        # 处理指定目录中的所有 FASTA 文件
        fasta_files = glob.glob(os.path.join(fasta_dir, '*.fasta'))
        print(f"Found {len(fasta_files)} FASTA files to process in {fasta_dir}.")

        for fasta_file in fasta_files:
            # 根据输入文件名构建输出文件名
            base_name = os.path.basename(fasta_file)  # 获取文件名
            file_prefix = os.path.splitext(base_name)[0]  # 去掉文件后缀
            output_file = os.path.join(output_dir, f"{file_prefix}_emb.pkl")  # 构建输出文件名

            # 检测 fasta 是否为 'nan'
            if is_nan_fasta(fasta_file):
                print(f"FASTA file {fasta_file} contains 'nan'. Skipping processing.")
                create_empty_pkl(output_file)
                continue

            # 调用生成嵌入的函数
            embedding_generate(args.esm_model_path, fasta_file, output_file, args.nogpu)

    # 结束计时
    time_end = time.time()
    print(f"Processed all files.")
    print('Embedding generation time cost:', time_end - time_start, 's')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 输入
    parser.add_argument('-emp', '--esm_model_path', type=str,
                        default='/home/data/user/zhuangjianan/zjn/EMS_MSA/esm/save_models/ESM-2/esm2_t33_650M_UR50D.pt',
                        help="ESM model location")

    # 输入基础路径
    parser.add_argument('-f', '--fasta_dir', type=str,
                        default='./data/S4169_fasta',
                        help="Base directory containing subdirectories with single-sequence FASTA files")

    # 输出基础路径
    parser.add_argument('-o', '--output_dir', type=str,
                        default='./data/S4169_emb',
                        help="Base directory to save generated ESM embeddings")

    # 字典列表
    parser.add_argument('--suffix_list', type=str,
                        # default='["a","a_m","b","b_m"]',
                        # default='["ab_l", "ab_h", "ag_a", "ag_b", "ab_l_m", "ab_h_m", "ag_a_m", "ag_b_m"]',
                        default='["R1", "R2", "R3", "R4", "R5", "R6", "L1", "L2", "L3","Rm1", "Rm2", "Rm3", "Rm4", "Rm5", "Rm6", "Lm1", "Lm2", "Lm3"]',
                        help="List of suffixes to create subdirectories and process")

    # 参数
    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    args = parser.parse_args()

    # 解析suffix_list为字典
    suffix_list = eval(args.suffix_list)  # 将字符串解析为实际的列表

    # 调用函数处理多个目录
    process_multiple_directories(args.fasta_dir, args.output_dir, suffix_list)
