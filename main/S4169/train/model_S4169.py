# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from math import sqrt
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from Radam import *
from lookahead import Lookahead
from scipy.stats import pearsonr
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device) # hid_dim//n_heads = 32

    def forward(self, query, key, value, mask=None):

        bsz = query.shape[0]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))

        x = torch.matmul(attention, V)

        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(bsz, -1, self.hid_dim)

        x = self.fc(x)

        return x

class Encoder(nn.Module):
    """protein feature extraction."""
    def __init__(self, protein_dim, hid_dim, n_layers, kernel_size, dropout, device):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"

        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.convs = nn.ModuleList(
            [nn.Conv1d(hid_dim, 2 * hid_dim, kernel_size, padding=(kernel_size - 1) // 2) for _ in
             range(self.n_layers)])  # convolutional layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim, self.hid_dim)
        self.gn = nn.GroupNorm(8, hid_dim * 2)
        self.ln = nn.LayerNorm(hid_dim )

    def forward(self, protein):

        conv_input = self.fc(protein)
        conv_input = conv_input.permute(0, 2, 1)

        for i, conv in enumerate(self.convs):

            conved = conv(self.dropout(conv_input))

            conved = F.glu(conved, dim=1)

            conved = (conved + conv_input) * self.scale

            conv_input = conved

        conved = conved.permute(0, 2, 1)

        conved = self.ln(conved)

        return conved


class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.pf_dim = pf_dim

        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)  # convolution neural units

        self.do = nn.Dropout(dropout)

    def forward(self, x):

        x = x.permute(0, 2, 1)

        x = self.fc_2(self.do(F.relu(self.fc_1(x))))

        x = x.permute(0, 2, 1)

        return x



class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()

        # 为每个子层分别添加独立的 LayerNorm
        self.ln_sa = nn.LayerNorm(hid_dim)  # self-attention后的LayerNorm
        self.ln_csa = nn.LayerNorm(hid_dim)  # self-attention后的LayerNorm
        self.ln_ca = nn.LayerNorm(hid_dim)
        self.ln = nn.LayerNorm(hid_dim)
        # 各个子层的组件
        self.sa = self_attention(hid_dim, n_heads, dropout, device)  # self-attention
        self.ca = self_attention(hid_dim, n_heads, dropout, device)  # cross-attention
        self.csa = self_attention(hid_dim, n_heads, dropout, device)  # cross-attention
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)  # position-wise feedforward
        self.do = nn.Dropout(dropout)  # Dropout层

    def forward(self, trg, src, trg_m, trg_mask=None, src_mask=None):


        # Self-attention + LayerNorm
        trg = self.ln_sa(trg + self.do(self.sa(trg, trg, trg, trg_mask)))  # self-attention

        trg = self.ln_csa(trg + self.do(self.csa(trg, trg_m, trg_m, trg_mask)))
        # Cross-attention + LayerNorm
        trg= self.ln_ca(trg + self.do(self.ca(trg, src, src, src_mask)))  # cross-attention
        # Feedforward + LayerNorm
        trg = self.ln(trg + self.do(self.pf(trg)))  # position-wise feedforward

        return trg


POLY_DEGREE = 3
def make_features(x):
    """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, POLY_DEGREE+1)], 1)

class Decoder(nn.Module):
    """ compound feature extraction."""
    def __init__(self, antigen_dim, hid_dim, n_layers, n_heads, pf_dim, decoder_layer, self_attention,
                 positionwise_feedforward, dropout, device):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.output_dim = antigen_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.decoder_layer = decoder_layer
        self.self_attention = self_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.dropout = dropout
        self.device = device
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.layers = nn.ModuleList([decoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device) for _ in range(n_layers)])
        self.ft = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.fc_1 = nn.Linear(hid_dim, 64)
        self.fc_2 = nn.Linear(64, 16)
        self.fc_3 = nn.Linear(16, 1)
        self.gn = nn.GroupNorm(8, 64)

    def forward(self, trg, src,trg_m, trg_mask=None, src_mask=None):
        trg = self.ft(trg)

        for layer in self.layers:
            trg = layer(trg, src, trg_m, trg_mask, src_mask)  # ag:ab

        """Use norm to determine which atom is significant. """
        norm = F.softmax(torch.norm(trg, dim=2), dim=1)

        summ = torch.sum(trg * norm.unsqueeze(-1), dim=1)

        return summ

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, delta_G_forward, delta_G_reverse, y_true):
        """
        计算损失：
        loss = logcosh((ΔΔG_Forward - ΔΔG_Reverse) / 2 - y) + |ΔΔG_Forward + ΔΔG_Reverse|

        Args:
            delta_G_forward (Tensor): ΔΔG_Forward
            delta_G_reverse (Tensor): ΔΔG_Reverse
            y_true (Tensor): 真实值 (y)

        Returns:
            Tensor: 计算出的损失值
        """
        # 计算 logcosh 项
        logcosh_term = torch.log(torch.cosh((delta_G_forward - delta_G_reverse) / 2 - y_true))

        # 计算绝对值项
        abs_term = torch.abs(delta_G_forward + delta_G_reverse)

        # 总损失
        loss = logcosh_term + abs_term
        return loss.mean()  # 取均值，使损失可用于优化

class Predictor(nn.Module):
    def __init__(self, encoder_ab, encoder_ag, decoder_ab, decoder_ag, device):
        super().__init__()

        self.encoder_ab = encoder_ab
        self.encoder_ag = encoder_ag
        self.decoder_ab = decoder_ab
        self.decoder_ag = decoder_ag
        self.device = device
        # self.init_weight()
        self.do = nn.Dropout(0.1)
        self.fc_11 = nn.Linear(256*2, 128)
        self.fc_12 = nn.Linear(256*2, 128)
        self.fc_21 = nn.Linear(128, 64)
        self.fc_22 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(128*3, 64)
        self.fc_3r = nn.Linear(128 * 3, 64)
        self.fc_4 = nn.Linear(64, 1)
        self.fc_4r = nn.Linear(64, 1)

        self.f1_linear = nn.Linear(20, 20)
        self.f2_linear = nn.Linear(1280, 1280)
        self.f3_linear = nn.Linear(1024, 1024)
        self.f4_linear = nn.Linear(23, 23)
        self.f5_linear = nn.Linear(7, 7)

    def make_masks(self, p11n, p21n, p11_max_len, p21_max_len):
        N = len(p11n)  # batch size
        p11_mask = torch.zeros((N, p11_max_len))
        p21_mask = torch.zeros((N, p21_max_len))
        for i in range(N):
            p11_mask[i, :p11n[i]] = 1
            p21_mask[i, :p21n[i]] = 1
        p11_mask = p11_mask.unsqueeze(1).unsqueeze(3).to(self.device)
        p21_mask = p21_mask.unsqueeze(1).unsqueeze(2).to(self.device)
        return p11_mask, p21_mask

    def process_tensor(self, x):
        """
        处理输入张量：
        1. 拆分 f1, f2, f3,f4,f5,f6,f7
        2. 通过线性层投影
        3. 重新拼接
        """
        x = x.clone().detach().to(dtype=torch.float32)  # 推荐方式
        f1,f2,f3,f4,f5 = x[:,:, :20], x[:, :, 20:1300], x[:, :, 1300:2324], x[:, :, 2324:2347], x[:, :, 2347:2354]
        f1_proj = self.f1_linear(f1)  # 线性变换
        f2_proj = self.f2_linear(f2)
        f3_proj = self.f3_linear(f3)
        f4_proj = self.f4_linear(f4)
        f5_proj = self.f5_linear(f5)


        out = torch.cat([f1_proj,f2_proj, f3_proj,f4_proj,f5_proj], dim=-1)

        return out

    def forward(self, ag_s, ab_s, ag_m_s, ab_m_s, ag_s_num, ab_s_num, ag_m_s_num, ab_m_s_num):
        # compound = [batch,atom_num,atom_dim]
        # protein = [batch,protein len,protein_dim]
        ag_s_max_len = ag_s.shape[1]
        ab_s_max_len = ab_s.shape[1]
        ag_s_mask, ab_s_mask = self.make_masks(ag_s_num, ab_s_num, ag_s_max_len, ab_s_max_len)

        ag_m_s_max_len = ag_m_s.shape[1]
        ab_m_s_max_len = ab_m_s.shape[1]
        ag_m_s_mask, ab_m_s_mask = self.make_masks(ag_m_s_num, ab_m_s_num, ag_m_s_max_len, ab_m_s_max_len)

        ag_s = self.process_tensor(ag_s)
        ab_s = self.process_tensor(ab_s)
        ag_m_s = self.process_tensor(ag_m_s)
        ab_m_s = self.process_tensor(ab_m_s)

        enc_ab_s = self.encoder_ab(ab_s)
        enc_ab_m_s = self.encoder_ab(ab_m_s)
        # enc_protein = [batch size, protein len, hid dim]
        enc_ag_s = self.encoder_ag(ag_s)
        enc_ag_m_s = self.encoder_ag(ag_m_s)
        # enc_compound = [batch size, compound len, hid dim]

        ag_ab= self.decoder_ag(enc_ag_s, enc_ab_s, enc_ag_m_s, ag_s_mask, ab_s_mask)
        ab_s_mask_change = ab_s_mask.permute(0, 1, 3, 2)
        ag_s_mask_change = ag_s_mask.permute(0, 1, 3, 2)
        ab_ag= self.decoder_ab(enc_ab_s, enc_ag_s, enc_ab_m_s, ab_s_mask_change, ag_s_mask_change)

        ag_ab_m= self.decoder_ag(enc_ag_m_s, enc_ab_m_s,enc_ag_s, ag_m_s_mask, ab_m_s_mask)
        ab_m_s_mask_change = ab_m_s_mask.permute(0, 1, 3, 2)
        ag_m_s_mask_change = ag_m_s_mask.permute(0, 1, 3, 2)
        ab_ag_m= self.decoder_ab(enc_ab_m_s, enc_ag_m_s, enc_ab_s, ab_m_s_mask_change, ag_m_s_mask_change)
        # graph???


        complex_wt = self.do(self.fc_11(torch.cat([ag_ab,ab_ag],-1)))
        complex_mut = self.do(self.fc_12(torch.cat([ag_ab_m,ab_ag_m],-1)))
        complex_sub = complex_wt - complex_mut
        complex_sub_r = -complex_sub
        # complex_mul = self.do(self.fc_22(complex_wt * complex_mut))
        final1 = self.do(self.fc_3(torch.cat([complex_wt, complex_mut,complex_sub],-1)))
        final1 = self.fc_4(final1)
        final1 = final1.view(-1)
        final2 = self.do(self.fc_3r(torch.cat([complex_wt, complex_mut,complex_sub_r],-1)))
        final2 = self.fc_4r(final2)
        final2 = final2.view(-1)


        return final1,final2

    def __call__(self, data, train=True):
        ag_s, ab_s, ag_m_s, ab_m_s, correct_interaction, ag_s_num, ab_s_num, ag_m_s_num, ab_m_s_num = data

        # 使用自定义损失函数
        criterion = CustomLoss()

        correct_interaction = correct_interaction.to(torch.float32)

        if train:
            predicted_interaction, predicted_interaction_reversed = self.forward(
                ag_s, ab_s, ag_m_s, ab_m_s, ag_s_num, ab_s_num, ag_m_s_num, ab_m_s_num
            )

            # 计算损失，使用自定义损失函数
            loss = criterion(predicted_interaction, predicted_interaction_reversed, correct_interaction)
            loss = loss.float()

            return loss, correct_interaction, predicted_interaction,predicted_interaction_reversed

        else:
            predicted_interaction, predicted_interaction_reversed = self.forward(
                ag_s, ab_s, ag_m_s, ab_m_s, ag_s_num, ab_s_num, ag_m_s_num, ab_m_s_num
            )

            return correct_interaction, predicted_interaction,predicted_interaction_reversed


def custom_collate_fn(batch, max_len=512):
    """
    替代 pack() 的功能：pad、tensor 转换、记录长度
    输入 batch 是列表，每个元素是 (ag_s, ab_s, ag_m_s, ab_m_s, label)
    """

    ag_s_list, ab_s_list, ag_m_s_list, ab_m_s_list, labels = zip(*batch)

    def pad_and_record_length(seq_list):
        seq_lengths = [len(seq) for seq in seq_list]
        feature_dim = seq_list[0].shape[1]
        max_seq_len = min(max(seq_lengths), max_len)
        padded_tensor = torch.zeros((len(seq_list), max_seq_len, feature_dim), dtype=torch.float32)
        for i, seq in enumerate(seq_list):
            seq = torch.tensor(seq, dtype=torch.float32)
            seq = seq[:max_seq_len]
            padded_tensor[i, :seq.shape[0], :] = seq
        return padded_tensor, seq_lengths

    # pad and get lengths
    ag_s_new, ag_s_num = pad_and_record_length(ag_s_list)
    ab_s_new, ab_s_num = pad_and_record_length(ab_s_list)
    ag_m_s_new, ag_m_s_num = pad_and_record_length(ag_m_s_list)
    ab_m_s_new, ab_m_s_num = pad_and_record_length(ab_m_s_list)

    # labels
    labels_new = torch.tensor(labels, dtype=torch.float32)

    return (
        ag_s_new, ab_s_new, ag_m_s_new, ab_m_s_new,
        labels_new, ag_s_num, ab_s_num, ag_m_s_num, ab_m_s_num
    )

class Trainer(object):
    def __init__(self, model, lr, weight_decay, batch, protein_dim):
        self.model = model
        weight_p, bias_p = [], []
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p)
        for name, p in self.model.named_parameters():
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)

        self.optimizer_inner = RAdam([
            {'params': weight_p, 'weight_decay': weight_decay},
            {'params': bias_p, 'weight_decay': 0}
        ], lr=lr)

        self.optimizer = Lookahead(self.optimizer_inner, k=5, alpha=0.5)
        self.batch = batch
        self.protein_dim = protein_dim

    def train(self, dataset, device):
        self.model.train()
        # dataloader = DataLoader(dataset, batch_size=self.batch, shuffle=True, collate_fn=custom_collate_fn)
        dataloader = DataLoader(dataset,num_workers=24, batch_size=self.batch, shuffle=True,
                                collate_fn=lambda x: custom_collate_fn(x, max_len=512))

        if hasattr(self.optimizer, 'optimizer'):
            self.optimizer.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()

        train_correct_fold = torch.zeros((1, 0), device=device)
        train_predict_fold = torch.zeros((1, 0), device=device)
        lo = []
        iteration = 0

        for (ag_s, ab_s, ag_m_s, ab_m_s, labels,
             ag_s_num, ab_s_num, ag_m_s_num, ab_m_s_num) in dataloader:

            # 迁移到 GPU（如果需要）
            ag_s = ag_s.to(device)
            ab_s = ab_s.to(device)
            ag_m_s = ag_m_s.to(device)
            ab_m_s = ab_m_s.to(device)
            labels = labels.to(device)

            data_pack = (ag_s, ab_s, ag_m_s, ab_m_s, labels,
                         ag_s_num, ab_s_num, ag_m_s_num, ab_m_s_num)

            loss, correct, predicted, predicted_reversed = self.model(data_pack)

            correct = correct.view(1, -1)
            predicted = predicted.view(1, -1)
            train_correct_fold = torch.cat([train_correct_fold, correct], dim=-1)
            train_predict_fold = torch.cat([train_predict_fold, predicted], dim=-1)

            lo.append(loss)
            loss.backward()

            self.optimizer.step()
            if hasattr(self.optimizer, 'optimizer'):
                self.optimizer.optimizer.zero_grad()
            else:
                self.optimizer.zero_grad()
            iteration += 1
        loss_train = sum(lo) / iteration
        return loss_train, train_correct_fold, train_predict_fold


def get_corr(fake_Y, Y):
    fake_Y, Y = fake_Y.reshape(-1), Y.reshape(-1)
    fake_Y_mean, Y_mean = torch.mean(fake_Y.float()), torch.mean(Y.float())
    corr = (torch.sum((fake_Y - fake_Y_mean) * (Y - Y_mean))) / (
        torch.sqrt(torch.sum((fake_Y - fake_Y_mean) ** 2)) * torch.sqrt(torch.sum((Y - Y_mean) ** 2)))
    return corr

class Tester(object):
    def __init__(self, model, device, protein_dim):
        self.model = model
        self.device = device
        self.protein_dim = protein_dim

    def test(self, dataset):
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

        T, Y, Y_reversed = [], [], []

        with torch.no_grad():
            for (ag_s, ab_s, ag_m_s, ab_m_s, labels,
                 ag_s_num, ab_s_num, ag_m_s_num, ab_m_s_num) in dataloader:
                # 迁移到 GPU（如果需要）

                ag_s = ag_s.to(self.device)
                ab_s = ab_s.to(self.device)
                ag_m_s = ag_m_s.to(self.device)
                ab_m_s = ab_m_s.to(self.device)
                labels = labels.to(self.device)

                data_pack = (ag_s, ab_s, ag_m_s, ab_m_s, labels,
                             ag_s_num, ab_s_num, ag_m_s_num, ab_m_s_num)

                correct, predicted, predicted_rev = self.model(data_pack, train=False)
                T.append(correct.cpu())
                Y.append(predicted.cpu())
                Y_reversed.append(predicted_rev.cpu())

        T = torch.cat(T).squeeze()
        Y = torch.cat(Y).squeeze()
        Y_reversed = torch.cat(Y_reversed).squeeze()
        T_reversed = -T

        loss = nn.MSELoss()(T, Y)
        pccs = get_corr(T, Y)
        pccs_reversed = get_corr(T_reversed, Y_reversed)

        T_np = T.numpy()
        Y_np = Y.numpy()

        mae = mean_absolute_error(T_np, Y_np)
        mse = mean_squared_error(T_np, Y_np)
        rmse = sqrt(mse)
        r2 = r2_score(T_np, Y_np)

        return pccs, pccs_reversed, mae, mse, rmse, r2, loss, T_np, Y_np

    def save_pccs(self, pccs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, pccs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)


