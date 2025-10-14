import os
import torch
import pickle
from antiberty import AntiBERTyRunner

# 初始化 AntiBERTyRunner
antiberty = AntiBERTyRunner()

# 指定需要处理的子文件夹
folders_to_process = ['ab_h', 'ab_h_m', 'ab_l', 'ab_l_m']

# 输入输出路径
input_base = './data/M1101_fasta'
output_base = './data/'

# 遍历指定文件夹
for subfolder in folders_to_process:
    subfolder_path = os.path.join(input_base, subfolder)
    if not os.path.isdir(subfolder_path):
        print(f"[Warning] Folder not found: {subfolder_path}")
        continue

    print(f"\n[Processing Folder] {subfolder}")

    for fasta_filename in os.listdir(subfolder_path):
        if fasta_filename.endswith('.fasta'):
            fasta_path = os.path.join(subfolder_path, fasta_filename)

            # 准备输出文件路径
            output_subfolder = os.path.join(output_base, subfolder)
            os.makedirs(output_subfolder, exist_ok=True)
            pkl_filename = fasta_filename.replace('.fasta', '_antiberty.pkl')
            pkl_path = os.path.join(output_subfolder, pkl_filename)

            try:
                with open(fasta_path, 'r') as f:
                    lines = f.readlines()
                    if len(lines) < 2:
                        raise ValueError("FASTA file has no valid sequence line.")
                    sequence = lines[1].strip()

                if sequence.lower() == 'nan' or len(sequence) == 0:
                    raise ValueError("Sequence is NaN or empty.")

                if len(sequence) > 512:
                    print(f"[Skip] {fasta_path} — sequence length {len(sequence)} exceeds 512.")
                    # 保存空pkl
                    with open(pkl_path, 'wb') as pkl_file:
                        pickle.dump(None, pkl_file)
                    continue

                # 正常生成embedding
                embedding = antiberty.embed([sequence])[0]
                with open(pkl_path, 'wb') as pkl_file:
                    pickle.dump(embedding, pkl_file)

                print(f"[Success] {fasta_filename} -> {pkl_path}")

            except Exception as e:
                # 如果出错/无效内容，保存空pkl
                with open(pkl_path, 'wb') as pkl_file:
                    pickle.dump(None, pkl_file)
                print(f"[Empty] {fasta_filename} — reason: {e}")


