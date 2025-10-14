# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from Radam import *
from lookahead import Lookahead


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

        self.ln_sa = nn.LayerNorm(hid_dim)
        self.ln_csa = nn.LayerNorm(hid_dim)
        self.ln_ca = nn.LayerNorm(hid_dim)
        self.ln = nn.LayerNorm(hid_dim)
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.ca = self_attention(hid_dim, n_heads, dropout, device)
        self.csa = self_attention(hid_dim, n_heads, dropout, device)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_m, trg_mask=None, src_mask=None):


        # Self-attention + LayerNorm
        trg = self.ln_sa(trg + self.do(self.sa(trg, trg, trg, trg_mask)))  # self-attention
        # Mutation Cross-attention + LayerNorm
        trg = self.ln_csa(trg + self.do(self.csa(trg, trg_m, trg_m, trg_mask)))
        # Interface Cross-attention + LayerNorm
        trg= self.ln_ca(trg + self.do(self.ca(trg, src, src, src_mask)))  # cross-attention
        # Feedforward + LayerNorm
        trg = self.ln(trg + self.do(self.pf(trg)))  # position-wise feedforward

        return trg

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
        self.fc_4 = nn.Linear(64, 1)

        self.f1_linear = nn.Linear(20, 20)
        self.f2_linear = nn.Linear(20, 20)
        self.f3_linear = nn.Linear(1280, 1280)
        self.f6_linear = nn.Linear(1024, 1024)
        self.f8_linear=nn.Linear(23,23)

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

    def forward(self, ag_s, ab_s, ag_m_s, ab_m_s, ag_s_num, ab_s_num, ag_m_s_num, ab_m_s_num):
        ag_s_max_len = ag_s.shape[1]
        ab_s_max_len = ab_s.shape[1]
        ag_s_mask, ab_s_mask = self.make_masks(ag_s_num, ab_s_num, ag_s_max_len, ab_s_max_len)

        ag_m_s_max_len = ag_m_s.shape[1]
        ab_m_s_max_len = ab_m_s.shape[1]
        ag_m_s_mask, ab_m_s_mask = self.make_masks(ag_m_s_num, ab_m_s_num, ag_m_s_max_len, ab_m_s_max_len)

        enc_ab_s = self.encoder_ab(ab_s)
        enc_ab_m_s = self.encoder_ab(ab_m_s)
        enc_ag_s = self.encoder_ag(ag_s)
        enc_ag_m_s = self.encoder_ag(ag_m_s)

        ag_ab= self.decoder_ag(enc_ag_s, enc_ab_s, enc_ag_m_s, ag_s_mask, ab_s_mask)
        ab_s_mask_change = ab_s_mask.permute(0, 1, 3, 2)
        ag_s_mask_change = ag_s_mask.permute(0, 1, 3, 2)
        ab_ag= self.decoder_ab(enc_ab_s, enc_ag_s, enc_ab_m_s, ab_s_mask_change, ag_s_mask_change)

        ag_ab_m= self.decoder_ag(enc_ag_m_s, enc_ab_m_s,enc_ag_s, ag_m_s_mask, ab_m_s_mask)
        ab_m_s_mask_change = ab_m_s_mask.permute(0, 1, 3, 2)
        ag_m_s_mask_change = ag_m_s_mask.permute(0, 1, 3, 2)
        ab_ag_m= self.decoder_ab(enc_ab_m_s, enc_ag_m_s, enc_ab_s, ab_m_s_mask_change, ag_m_s_mask_change)

        complex_wt = self.do(self.fc_11(torch.cat([ag_ab,ab_ag],-1)))
        complex_mut = self.do(self.fc_12(torch.cat([ag_ab_m,ab_ag_m],-1)))
        complex_sub = complex_wt - complex_mut
        final1 = self.do(self.fc_3(torch.cat([complex_wt, complex_mut,complex_sub],-1)))
        final1 = self.fc_4(final1)
        final1 = final1.view(-1)

        return final1

    def __call__(self, data, train=True):

        ag_s, ab_s, ag_m_s, ab_m_s, correct_interaction, ag_s_num, ab_s_num, ag_m_s_num, ab_m_s_num = data

        Loss = nn.MSELoss()
        correct_interaction = correct_interaction.to(torch.float32)

        if train:
            predicted_interaction = self.forward(ag_s, ab_s, ag_m_s, ab_m_s, ag_s_num, ab_s_num, ag_m_s_num, ab_m_s_num)
            loss = Loss(predicted_interaction, correct_interaction)
            loss = loss.float()
            return loss, correct_interaction, predicted_interaction

        else:
            predicted_interaction = self.forward(ag_s, ab_s, ag_m_s, ab_m_s, ag_s_num, ab_s_num, ag_m_s_num, ab_m_s_num)
            return correct_interaction, predicted_interaction


# zero pack
def pack(ag_s, ab_s, ag_m_s, ab_m_s, labels, device,ab_dim,ag_dim):

    ag_s_len = 0
    ab_s_len = 0
    ag_m_s_len = 0
    ab_m_s_len = 0
    N = len(labels)

    ag_s_num = []
    for ag in ag_s:
        ag_s_num.append(ag.shape[0])
        if ag.shape[0] >= ag_s_len:
            ag_s_len = ag.shape[0]
    ab_s_num = []
    for ab in ab_s:
        ab_s_num.append(ab.shape[0])
        if ab.shape[0] >= ab_s_len:
            ab_s_len = ab.shape[0]
    ag_m_s_num = []
    for agm in ag_m_s:
        ag_m_s_num.append(agm.shape[0])
        if agm.shape[0] >= ag_m_s_len:
            ag_m_s_len = agm.shape[0]
    ab_m_s_num = []
    for abm in ab_m_s:
        ab_m_s_num.append(abm.shape[0])
        if abm.shape[0] >= ab_m_s_len:
            ab_m_s_len = abm.shape[0]

    ag_s_new = torch.zeros((N, ag_s_len, ag_dim), device=device)
    i = 0
    for ag in ag_s:
        #ag = ag.astype(float)
        ag = torch.tensor(ag)
        a_len = ag.shape[0]
        ag_s_new[i, :a_len, :] = ag
        i += 1
    ab_s_new = torch.zeros((N, ab_s_len, ab_dim), device=device)
    i = 0
    for ab in ab_s:
        #ab = ab.astype(float)
        ab = torch.tensor(ab)
        a_len = ab.shape[0]
        ab_s_new[i, :a_len, :] = ab
        i += 1

    ag_m_s_new = torch.zeros((N, ag_m_s_len, ag_dim), device=device)
    i = 0
    for agm in ag_m_s:
        #ag = ag.astype(float)
        agm = torch.tensor(agm)
        a_len = agm.shape[0]
        ag_m_s_new[i, :a_len, :] = agm
        i += 1
    ab_m_s_new = torch.zeros((N, ab_m_s_len, ab_dim), device=device)
    i = 0
    for abm in ab_m_s:
        #ab = ab.astype(float)
        abm = torch.tensor(abm)
        a_len = abm.shape[0]
        ab_m_s_new[i, :a_len, :] = abm
        i += 1

    labels_new = torch.zeros(N, dtype=torch.float, device=device)
    i = 0
    for label in labels:
        labels_new[i] = label
        i += 1

    return (ag_s_new, ab_s_new, ag_m_s_new, ab_m_s_new, labels_new,
            ag_s_num, ab_s_num, ag_m_s_num, ab_m_s_num)


class Trainer(object):
    def __init__(self, model, lr, weight_decay, batch, ab_dim,ag_dim):
        self.model = model
        weight_p, bias_p = [], []
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p)
        for name, p in self.model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        self.optimizer_inner = RAdam([{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=lr)
        self.optimizer = Lookahead(self.optimizer_inner, k=5, alpha=0.5)
        self.batch = batch
        self.ab_dim = ab_dim
        self.ag_dim = ag_dim

    def train(self, dataset, device):
        self.model.train()
        N = len(dataset)
        i = 0
        iteration = 0
        if hasattr(self.optimizer, 'optimizer'):  # Lookahead 内部有个 optimizer 属性
            self.optimizer.optimizer.zero_grad()  # 调用基础优化器的 zero_grad()
        else:
            self.optimizer.zero_grad()  # 如果是普通优化器，直接调用
        ag_s, ab_s, ag_m_s, ab_m_s, labels = [], [], [], [], []
        train_correct_fold = torch.zeros((1,0), device=device)
        train_predict_fold = torch.zeros((1,0), device=device)
        lo = []
        for data in dataset:
            i = i+1
            ag, ab, agm, abm, label = data
            ag_s.append(ag)
            ab_s.append(ab)
            ag_m_s.append(agm)
            ab_m_s.append(abm)
            labels.append(label)
            if i % self.batch == 0 or i == N:
                iteration += 1
                data_pack = pack(ag_s, ab_s, ag_m_s, ab_m_s, labels, device,self.ab_dim,self.ag_dim)
                loss, correct, predicted= self.model(data_pack)  # predictor.train()
                correct = correct.view(1,-1)
                predicted = predicted.view(1,-1)
                train_correct_fold = torch.cat([train_correct_fold,correct], dim=-1)
                train_predict_fold = torch.cat([train_predict_fold,predicted], dim=-1)
                lo.append(loss)
                loss.backward()

                ag_s, ab_s, ag_m_s, ab_m_s, labels = [], [], [], [], []
                self.optimizer.step()
                if hasattr(self.optimizer, 'optimizer'):  # Lookahead 内部有个 optimizer 属性
                    self.optimizer.optimizer.zero_grad()  # 调用基础优化器的 zero_grad()
                else:
                    self.optimizer.zero_grad()  # 如果是普通优化器，直接调用
            else:
                continue

        loss_train = sum(lo)/iteration
        return loss_train, train_correct_fold, train_predict_fold

def get_corr(fake_Y, Y):
    fake_Y, Y = fake_Y.reshape(-1), Y.reshape(-1)
    fake_Y_mean, Y_mean = torch.mean(fake_Y.float()), torch.mean(Y.float())
    corr = (torch.sum((fake_Y - fake_Y_mean) * (Y - Y_mean))) / (
        torch.sqrt(torch.sum((fake_Y - fake_Y_mean) ** 2)) * torch.sqrt(torch.sum((Y - Y_mean) ** 2)))
    return corr

class Tester(object):
    def __init__(self, model, device, ab_dim,ag_dim):
        self.model = model
        self.device = device  # 设备作为超参数传入
        self.ab_dim = ab_dim
        self.ag_dim = ag_dim


    def test(self, dataset):
        self.model.eval()
        T = torch.zeros((1,0), device=self.device)  # 使用传入的 device
        Y = torch.zeros((1,0), device=self.device)  # 使用传入的 device
        with torch.no_grad():
            for data in dataset:
                ag_s, ab_s, ag_m_s, ab_m_s, labels = [], [], [], [], []
                ag, ab, agm, abm, label = data
                ag_s.append(ag)
                ab_s.append(ab)
                ag_m_s.append(agm)
                ab_m_s.append(abm)
                labels.append(label)
                data = pack(ag_s, ab_s, ag_m_s, ab_m_s, labels, self.device,self.ab_dim,self.ag_dim)  # 使用传入的 device
                correct, predicted = self.model(data, train=False)
                correct = correct.view(1,-1)
                predicted = predicted.view(1,-1)
                T = torch.cat([T,correct], dim=-1)
                Y = torch.cat([Y,predicted], dim=-1)

        T = T.squeeze()
        Y = Y.squeeze()
        Loss = nn.MSELoss()
        loss = Loss(T, Y)  # an epoch's val's loss
        print('true:', T)
        print('predict:', Y)
        pccs = get_corr(T,Y)  # an epoch's val's pccs

        T = T.detach().cpu().numpy()
        Y = Y.detach().cpu().numpy()
        mae = mean_absolute_error(T, Y)
        mse = mean_squared_error(T, Y)
        rmse = sqrt(mean_squared_error(T, Y))
        r2 = r2_score(T, Y)
        return pccs, mae, mse, rmse, r2, loss, T, Y

    def save_pccs(self, pccs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, pccs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)

