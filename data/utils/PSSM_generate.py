import os
import numpy as np
from multiprocessing import Pool
import logging

class PSSMGenerator:
    def __init__(self, input_fasta_dir, output_pssm_dir):
        self.input_fasta_dir = input_fasta_dir
        self.output_pssm_dir = output_pssm_dir

        # 创建输出目录（如果不存在）
        os.makedirs(self.output_pssm_dir, exist_ok=True)
        # 初始化Logger

    def is_nan_fasta(self, fasta_file):
        """检查 fasta 文件内容是否包含 'nan' 行"""
        with open(fasta_file, 'r') as file:
            lines = file.readlines()
        # 检查是否有 'nan' 行
        for line in lines:
            if line.strip().lower() == "nan":
                return True
        return False

    def create_empty_output(self, fasta_file):
        """创建空的 .pssm 和 .txt 文件"""
        fasta_basename = os.path.basename(fasta_file)
        pssm_output_file = os.path.join(self.output_pssm_dir, fasta_basename.replace('.fasta', '.pssm'))
        txt_output_file = os.path.join(self.output_pssm_dir, fasta_basename.replace('.fasta', '.txt'))

        # 创建空文件
        open(pssm_output_file, 'w').close()
        open(txt_output_file, 'w').close()
        print(f"Empty PSSM and TXT files created for {fasta_file}")

    def seq2pssm(self, fasta_file):
        # 检查文件是否为 'nan'
        if self.is_nan_fasta(fasta_file):
            self.create_empty_output(fasta_file)
            return None

        # 定义 PSSM 输出文件名
        fasta_basename = os.path.basename(fasta_file)
        pssm_output_file = os.path.join(self.output_pssm_dir, fasta_basename.replace('.fasta', '.pssm'))
        txt_output_file = os.path.join(self.output_pssm_dir, fasta_basename.replace('.fasta', '.txt'))

        # ✅ 如果 PSSM 已存在，跳过处理
        if os.path.exists(pssm_output_file):
            print(f"✓ Skip (PSSM already exists): {fasta_file}")
            return pssm_output_file

        # 运行 psiblast 命令
        os.system(
            f"./data/ncbi-blast-2.12.0+/bin/psiblast "
            f"-query {fasta_file} "
            f"-db ./data/ncbi-blast-2.12.0+/bin/swissprot "
            f"-num_iterations 3 "
            f"-out {txt_output_file} "
            f"-out_ascii_pssm {pssm_output_file}"
        )

        # 检查 PSSM 文件是否生成成功
        if not os.path.exists(pssm_output_file):
            message = f"✗ Failed to generate PSSM for {fasta_file}"
            print(message)
            # 将信息写入日志文件
            with open('/mydata/zhuangjianan/AttABseq-main/cross_validation/data/utils/PSSM.log', 'a') as log_file:
                log_file.write(message + '\n')
            return None

        return pssm_output_file  # ✅ 成功时返回

    # def seq2pssm(self, fasta_file):
    #     # 检查文件是否为 'nan'
    #     if self.is_nan_fasta(fasta_file):
    #         self.create_empty_output(fasta_file)
    #         return None
    #
    #     # 定义 PSSM 输出文件名
    #     fasta_basename = os.path.basename(fasta_file)
    #     pssm_output_file = os.path.join(self.output_pssm_dir, fasta_basename.replace('.fasta', '.pssm'))
    #     txt_output_file = os.path.join(self.output_pssm_dir, fasta_basename.replace('.fasta', '.txt'))
    #
    #     # 运行 psiblast 命令
    #     os.system(
    #         f"/mydata/zhuangjianan/AttABseq-main/cross_validation/ncbi-blast-2.12.0+/bin/psiblast "
    #         f"-query {fasta_file} "
    #         f"-db /mydata/zhuangjianan/AttABseq-main/cross_validation/ncbi-blast-2.12.0+/bin/swissprot "
    #         f"-num_iterations 3 "
    #         f"-out {txt_output_file} "
    #         f"-out_ascii_pssm {pssm_output_file}"
    #     )
    #
    #     # 检查 PSSM 文件是否生成成功
    #     if not os.path.exists(pssm_output_file):
    #         print(f"Failed to generate PSSM for {fasta_file}")
    #         return None
    #
    #     return pssm_output_file  # ✅ 成功时返回
    def generate_all_pssms(self):
        fasta_files = [os.path.join(self.input_fasta_dir, f) for f in os.listdir(self.input_fasta_dir) if
                       f.endswith('.fasta')]

        for fasta_file in fasta_files:
            print(f"Processing {fasta_file}...")
            pssm = self.seq2pssm(fasta_file)

            if pssm is not None:
                print(f"PSSM generated and saved for {fasta_file}")
            else:
                print(f"Skip to generate PSSM for {fasta_file}")



def process_pssm(seq):
    input_fasta_dir = f'./data/S4169_fasta/{seq}'
    output_pssm_dir = f'./data/S4169_PSSM/{seq}'

    # 创建 PSSMGenerator 实例并生成 PSSM
    pssm_generator = PSSMGenerator(input_fasta_dir, output_pssm_dir)
    pssm_generator.generate_all_pssms()


if __name__ == '__main__':
    # seq = ['ab_h', 'ab_h_m', 'ab_l', 'ab_l_m', 'ag_a', 'ag_a_m', 'ag_b', 'ag_b_m']
    # seq = ['a', 'a_m', 'b', 'b_m']
    seq = ["R1", "R2", "R3", "R4", "R5", "R6", "L1", "L2", "L3","Rm1", "Rm2", "Rm3", "Rm4", "Rm5", "Rm6", "Lm1", "Lm2", "Lm3"]
    # 使用 Pool 创建并行进程池，指定进程数为 CPU 核心数
    with Pool(processes=os.cpu_count()) as pool:
        pool.map(process_pssm, seq)
