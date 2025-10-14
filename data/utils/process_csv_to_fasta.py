
import pandas as pd
import os

def create_fasta_from_csv(csv_file, output_dir, fasta_suffix):
    """
    根据CSV文件生成fasta文件。

    Args:
        csv_file (str): CSV文件的路径。
        output_dir (str): 保存fasta文件的目录。
        sequence_column (str): CSV文件中用于提取序列的列名。
        fasta_suffix (str): 输出fasta文件的后缀，默认是'_a_mut'。
    """
    # 读取CSV文件
    data = pd.read_csv(csv_file)

    # 遍历每一行，生成fasta文件
    for index, row in data.iterrows():
        # 构建fasta文件名
        fasta_name = f"{row['PDB']}_{row['mutation_clean']}_{fasta_suffix}.fasta"
        fasta_path = os.path.join(output_dir, fasta_name)

        # 获取序列内容
        sequence = row[fasta_suffix]

        # 写入fasta文件
        with open(fasta_path, 'w') as fasta_file:
            fasta_file.write(f">{row['PDB']}_{row['mutation_clean']}_{fasta_suffix}\n")
            fasta_file.write(str(sequence))
        print(f"Saved {fasta_path}")


# 示例调用
fasta_suffix_list = ["R1", "R2", "R3", "R4", "R5", "R6", "L1", "L2", "L3","Rm1", "Rm2", "Rm3", "Rm4", "Rm5", "Rm6", "Lm1", "Lm2", "Lm3"]  # fasta_suffix 字符串列表
# fasta_suffix_list = ["a", "b", "a_m", "b_m"]  # fasta_suffix 字符串列表
# fasta_suffix_list = ["ab_l", "ab_h", "ag_a", "ag_b", "ab_l_m", "ab_h_m", "ag_a_m", "ag_b_m"]
csv_file = './data/S4169/S4169.csv'  # 替换为你的CSV文件路径
output_dir_base = "./data/S4169/S4169_fasta"  # 替换为保存fasta文件的根目录

# 遍历fasta_suffix列表，执行对应操作
for fasta_suffix in fasta_suffix_list:
    output_dir = os.path.join(output_dir_base, fasta_suffix)  # 创建每个后缀的输出文件夹路径
    os.makedirs(output_dir, exist_ok=True)  # 创建保存目录（如果不存在）

    # 生成fasta文件
    create_fasta_from_csv(csv_file, output_dir, fasta_suffix)
    print(f"Finished generating FASTA files for suffix: {fasta_suffix}")


