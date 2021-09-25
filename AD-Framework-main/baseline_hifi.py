from pathlib import Path
import matplotlib as mpl
from sklearn import preprocessing
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams["font.size"] = 16
import numpy as np
from sklearn.metrics import recall_score, precision_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from utils.config import HIFIConfig
from utils.device import get_free_device
from utils.outputs import HIFIOutput
from utils.utils import str2bool
import os
import torch
import torch.nn as nn
import argparse
from utils.logger import create_logger
from utils.metrics import MetricsResult
from utils.data_provider import get_loader, rolling_window_2D, cutting_window_2D, unroll_window_3D
from statistics import mean
from torch.optim import Adam, lr_scheduler

from utils.data_provider import dataset2path, read_dataset
from utils.metrics import SD_autothreshold, MAD_autothreshold, IQR_autothreshold, get_labels_by_threshold
from utils.utils import make_result_dataframe
from sklearn.metrics import f1_score
import torch.nn.functional as F


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('nvl,vw->nwl',(x,A))
        return x.contiguous()

class mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = nn.Linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.alpha = alpha

    def forward(self,x,adj):        # batch_size, seq_len, num_nodes
        x = x.transpose(1,2)        # batch_size, num_nodes, seq_len
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            out.append(h)
        ho = torch.cat(out,dim=1)   # batch_size, num_nodes*(1+gdep), seq_len
        ho = ho.transpose(1,2)      # batch_size, seq_len, num_nodes*(1+gdep)
        ho = self.mlp(ho)           # batch_size, seq_len, c_out
        return ho

class graph_constructor(nn.Module):
    def __init__(self, nnodes, dim, alpha=3, static_feat=None, k=2):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(idx.device)
        mask.fill_(float('0'))
        s1,t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj

    def fullA(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        return adj


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.dropout_rate = dropout

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        return q, attn

    def __repr__(self):
        return "MultiHeadAttention ({}, {}, {}, {}, {})" \
               "".format(self.n_head, self.d_model, self.d_k, self.d_v, self.dropout_rate)


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()

        self.d_in = d_in
        self.d_hid = d_hid
        self.dropout_rate = dropout

        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x
        x = self.layer_norm(x)

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        return x

    def __repr__(self):
        return "PositionwiseFeedForward ({}, {}, {})".format(
            self.d_in, self.d_hid, self.dropout_rate
        )


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

    def __repr__(self):
        return "EncoderLayer (\n" \
               "\t {}, \n" \
               "\t {})".format(self.slf_attn.__repr__(), self.pos_ffn.__repr__())


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(
            self, dec_input, enc_output,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn

    def __repr__(self):
        return "DecoderLayer (\n" \
               "\t {}, \n" \
               "\t {}, \n" \
               "\t {})".format(self.slf_attn.__repr__(), self.enc_attn.__repr__(), self.pos_ffn.__repr__())


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.d_hid = d_hid
        self.n_position = n_position

        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

    def __repr__(self):
        return "PositionalEncoding ({}, {})".format(self.d_hid, self.n_position)


class VEncoder(nn.Module):
    """Variational Encoder"""
    def __init__(
            self, ad_size, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1, n_position=200, sequence_length=100,
            gcn_layers=2, gcn_alpha=0.2, k=2):

        super().__init__()

        self.ad_size = ad_size
        self.d_word_vec = d_word_vec
        self.dropout_rate = dropout

        # Multivariate Feature Interaction Module
        self.src_emb = nn.Linear(ad_size, d_word_vec)
        self.gc = graph_constructor(d_word_vec, d_k, k=k)
        self.mixgcn_left = mixprop(d_word_vec, d_word_vec, gcn_layers, gcn_alpha)
        self.mixgcn_right = mixprop(d_word_vec, d_word_vec, gcn_layers, gcn_alpha)
        self.gcn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        self.mu_layer = nn.Linear(d_model, d_model)
        self.mu_bn = nn.BatchNorm1d(sequence_length)
        self.mu_bn.weight.requires_grad = False
        # Batch Normalized is appended to Mean layer,
        # which is inspired from "A Batch Normalized Inference Network Keeps the KL Vanishing Away"
        self.logvar_layer = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward

        src_features = self.src_emb(src_seq)
        src_features = F.dropout(src_features, self.dropout_rate, training=self.training)
        adj = self.gc(torch.arange(self.d_word_vec).to(src_features.device))
        ho = self.mixgcn_left(src_features, adj) + self.mixgcn_right(src_features, adj.transpose(1, 0))
        src_features = ho + src_features
        src_features = self.gcn_layer_norm(src_features)

        enc_output = self.dropout(self.position_enc(src_features))

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        enc_output = self.layer_norm(enc_output)

        mean = self.mu_bn(self.mu_layer(enc_output))
        logvar = self.logvar_layer(enc_output)

        if return_attns:
            return mean, logvar, enc_slf_attn_list
        return mean, logvar,

    def encode(self, src_seq, src_mask, return_attns=False):

        # batch_size, sequence_length, d_model
        mu, logvar = self.forward(src_seq, src_mask, return_attns)

        # batch_size, sequence_length, d_model
        # For now, number of samples are set to one. As shown in original variational autoencoder, one sample is enough.
        z = self.reparameterize(mu, logvar)

        # batch_size, sequence_length
        KL = 0.5 * (mu.pow(2) + logvar.exp() - logvar -1).sum(-1)

        return z, KL

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp()
        eps = torch.zeros_like(std).normal_()
        return mu + torch.mul(eps, std)


class VDecoder(nn.Module):

    def __init__(
            self, ad_size, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, n_position=200, dropout=0.1,
            gcn_layers=2, gcn_alpha=0.2, k=2):

        super().__init__()

        self.ad_size = ad_size
        self.dropout_rate = dropout
        self.d_word_vec = d_word_vec
        self.src_emb = nn.Linear(ad_size, d_word_vec)

        # Multivariate Feature Interaction Module
        self.gc = graph_constructor(d_word_vec, d_k, k=k)
        self.mixgcn_left = mixprop(d_word_vec, d_word_vec, gcn_layers, gcn_alpha)
        self.mixgcn_right = mixprop(d_word_vec, d_word_vec, gcn_layers, gcn_alpha)
        self.gcn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.trg_emb = nn.Linear(ad_size, d_word_vec)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        trg_features = self.trg_emb(trg_seq)
        trg_features = F.dropout(trg_features, self.dropout_rate, training=self.training)
        adj = self.gc(torch.arange(self.d_word_vec).to(trg_features.device))
        ho = self.mixgcn_left(trg_features, adj) + self.mixgcn_right(trg_features, adj.transpose(1, 0))
        trg_features = ho + trg_features
        trg_features = self.gcn_layer_norm(trg_features)

        # -- Forward
        dec_output = self.dropout(self.position_enc(trg_features))

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        dec_output = self.layer_norm(dec_output)

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class HIFIEncoderDecoder(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, ad_size,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200, sequence_length=100,
            gcn_layers=2, gcn_alpha=0.2, k=2):

        super().__init__()

        self.encoder = VEncoder(
            ad_size=ad_size,
            n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout,
            sequence_length=sequence_length,
            gcn_layers=gcn_layers, gcn_alpha=gcn_alpha, k=k)

        self.decoder = VDecoder(
            ad_size=ad_size,
            n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        # Map the encoded hidden representation to input space
        self.decoder_to_input = nn.Linear(d_model, ad_size)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

    def forward(self, src_seq, trg_seq):

        src_mask = torch.ones(src_seq.shape[0], src_seq.shape[1]).to(src_seq.device).unsqueeze(-2)
        trg_mask = torch.ones(src_seq.shape[0], src_seq.shape[1]).to(src_seq.device)
        trg_mask = get_subsequent_mask(trg_mask)

        enc_output, kl_loss = self.encoder.encode(src_seq, src_mask)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        reconstruct_input = self.decoder_to_input(dec_output)

        return reconstruct_input, kl_loss


class HIFI(nn.Module):
    def __init__(self, file_name, config):
        super(HIFI, self).__init__()
        # file info
        self.dataset = config.dataset
        self.file_name = file_name

        # dim info
        self.x_dim = config.x_dim
        self.h_dim = config.h_dim
        self.d_inner = config.d_inner
        self.n_layers = config.n_layers
        self.n_head = config.n_head
        self.d_k = config.d_k
        self.d_v = config.d_v
        self.gcn_layers = config.gcn_layers
        self.gcn_alpha = config.gcn_alpha
        self.k = config.k
        self.window_size = config.rolling_size

        # sequence info
        self.preprocessing = config.preprocessing
        self.use_overlapping = config.use_overlapping
        self.use_last_point = config.use_last_point
        self.rolling_size = config.rolling_size

        # optimization info
        self.epochs = config.epochs
        self.milestone_epochs = config.milestone_epochs
        self.lr = config.lr
        self.gamma = config.gamma
        self.batch_size = config.batch_size
        self.weight_decay = config.weight_decay
        self.early_stopping = config.early_stopping
        self.loss_function = config.loss_function
        self.display_epoch = config.display_epoch
        self.use_clip_norm = config.use_clip_norm
        self.gradient_clip_norm = config.gradient_clip_norm
        self.kl_start = config.kl_start
        self.kl_warmup = config.kl_warmup

        # dropout
        self.dropout = config.dropout
        self.continue_training = config.continue_training

        self.robustness = config.robustness

        # pid
        self.pid = config.pid

        self.save_model = config.save_model
        if self.save_model:
            if not os.path.exists('./save_model/{}/'.format(self.dataset)):
                os.makedirs('./save_model/{}/'.format(self.dataset))
            self.save_model_path = \
                './save_model/{}/HIFI_hdim_{}_rollingsize_{}' \
                '_{}_pid={}.pt'.format(self.dataset, config.h_dim, config.rolling_size, Path(self.file_name).stem, self.pid)
        else:
            self.save_model_path = None

        self.load_model = config.load_model
        if self.load_model:
            self.load_model_path = \
                './save_model/{}/HIFI_hdim_{}_rollingsize_{}' \
                '_{}_pid={}.pt'.format(self.dataset, config.h_dim, config.rolling_size, Path(self.file_name).stem, self.pid)
        else:
            self.load_model_path = None

        # units
        self.model = HIFIEncoderDecoder(ad_size=self.x_dim, d_word_vec=self.h_dim, d_model=self.h_dim, d_inner=self.d_inner,
                                        n_layers=self.n_layers, n_head=self.n_head, d_k=self.d_k, d_v=self.d_v,
                                        dropout=self.dropout, n_position=self.window_size, sequence_length=self.window_size,
                                        gcn_layers=self.gcn_layers, gcn_alpha=self.gcn_alpha, k=self.k)

    def reset_parameters(self):
        #print("reset_parameters in HIFI will do nothing because the initial methods is called by HIFIEncoderDecoder.")
        pass

    def forward(self, src_seq):
        trg_seq = src_seq.detach().clone()
        decoded_output, kl_loss = self.model(src_seq, trg_seq)
        return decoded_output, kl_loss

    def fit(self, train_input, train_label, valid_input, valid_label, test_input, test_label, abnormal_data, abnormal_label, original_x_dim):
        loss_fn = nn.MSELoss()
        opt = Adam(list(self.parameters()), lr=self.lr, weight_decay=self.weight_decay)
        sched = lr_scheduler.StepLR(optimizer=opt, step_size=self.milestone_epochs, gamma=self.gamma)
        # get batch data
        train_data = get_loader(input=train_input, label=train_label, batch_size=self.batch_size, from_numpy=True,
                                drop_last=False, shuffle=False)
        valid_data = get_loader(input=valid_input, label=valid_label, batch_size=self.batch_size, from_numpy=True,
                                drop_last=False, shuffle=False)
        test_data = get_loader(input=test_input, label=test_label, batch_size=self.batch_size, from_numpy=True,
                               drop_last=False, shuffle=False)
        min_valid_loss, all_patience, cur_patience, best_epoch = 1e20, 10, 1, 0
        if self.load_model == True and self.continue_training == False:
            epoch_valid_losses = [-1]
            self.load_state_dict(torch.load(self.load_model_path))
        elif self.load_model == True and self.continue_training == True:
            self.load_state_dict(torch.load(self.load_model_path))
            # train model
            kl_weight = self.kl_start
            if self.kl_warmup > 0:
                anneal_rate = (1.0 - self.kl_start) / (
                            self.kl_warmup * len(train_input) / self.batch_size)
            else:
                anneal_rate = 0
            epoch_losses = []
            epoch_valid_losses = []
            for epoch in range(self.epochs):
                self.train()
                kl_weight = min(1.0, kl_weight+anneal_rate)
                train_losses = []
                # opt.zero_grad()
                for i, (batch_x, batch_y) in enumerate(train_data):
                    opt.zero_grad()
                    batch_x = batch_x.to(device)
                    batch_x_reconstruct, kl_loss = self.forward(batch_x)
                    batch_loss = loss_fn(batch_x_reconstruct, batch_x)
                    batch_loss = batch_loss + kl_weight * kl_loss.mean()
                    batch_loss.backward()
                    if self.use_clip_norm:
                        torch.nn.utils.clip_grad_norm_(list(self.parameters()), self.gradient_clip_norm)
                    opt.step()
                    sched.step()
                    train_losses.append(batch_loss.item())
                epoch_losses.append(mean(train_losses))
                if epoch % self.display_epoch == 0:
                    train_logger.info('epoch = {} , train loss = {}'.format(epoch, epoch_losses[-1]))

                valid_losses = []
                # opt.zero_grad()
                self.eval()
                with torch.no_grad():
                    for i, (val_batch_x, val_batch_y) in enumerate(valid_data):
                        val_batch_x = val_batch_x.to(device)
                        val_batch_x_reconstruct, _ = self.forward(val_batch_x)
                        val_batch_loss = loss_fn(val_batch_x_reconstruct, val_batch_x)
                        valid_losses.append(val_batch_loss.item())
                epoch_valid_losses.append(mean(valid_losses))
                if epoch % self.display_epoch == 0:
                    train_logger.info('epoch = {} , valid loss = {}'.format(epoch, epoch_valid_losses[-1]))

                if self.early_stopping:
                    if len(epoch_valid_losses) > 1:
                        if epoch_valid_losses[best_epoch] - epoch_valid_losses[-1] < 3e-4:
                            train_logger.info('EarlyStopping counter: {} out of {}'.format(cur_patience, all_patience))
                            if cur_patience == all_patience:
                                train_logger.info('Early Stopping!')
                                break
                            cur_patience += 1
                        else:
                            train_logger.info("Saving Model.")
                            torch.save(self.state_dict(), self.save_model_path)
                            best_epoch = epoch
                            cur_patience = 1
                    else:
                        torch.save(self.state_dict(), self.save_model_path)
        else:
            # train model
            self.train()
            kl_weight = self.kl_start
            if self.kl_warmup > 0:
                anneal_rate = (1.0 - self.kl_start) / (
                            self.kl_warmup * len(train_input) / self.batch_size)
            else:
                anneal_rate = 0
            epoch_losses = []
            epoch_valid_losses = []
            for epoch in range(self.epochs):
                kl_weight = min(1.0, kl_weight + anneal_rate)
                train_losses = []
                # opt.zero_grad()
                for i, (batch_x, batch_y) in enumerate(train_data):
                    opt.zero_grad()
                    batch_x = batch_x.to(device)
                    batch_x_reconstruct, kl_loss = self.forward(batch_x)
                    batch_loss = loss_fn(batch_x_reconstruct, batch_x)
                    batch_loss = batch_loss + kl_weight * kl_loss.mean()
                    batch_loss.backward()
                    if self.use_clip_norm:
                        torch.nn.utils.clip_grad_norm_(list(self.parameters()), self.gradient_clip_norm)
                    opt.step()
                    sched.step()
                    train_losses.append(batch_loss.item())
                epoch_losses.append(mean(train_losses))
                if epoch % self.display_epoch == 0:
                    train_logger.info('epoch = {} , train loss = {}'.format(epoch, epoch_losses[-1]))

                valid_losses = []
                # opt.zero_grad()
                self.eval()
                with torch.no_grad():
                    for i, (val_batch_x, val_batch_y) in enumerate(valid_data):
                        val_batch_x = val_batch_x.to(device)
                        val_batch_x_reconstruct, _ = self.forward(val_batch_x)
                        val_batch_loss = loss_fn(val_batch_x_reconstruct, val_batch_x)
                        valid_losses.append(val_batch_loss.item())
                epoch_valid_losses.append(mean(valid_losses))
                if epoch % self.display_epoch == 0:
                    train_logger.info('epoch = {} , valid loss = {}'.format(epoch, epoch_valid_losses[-1]))

                if self.early_stopping:
                    if len(epoch_valid_losses) > 1:
                        if epoch_valid_losses[best_epoch] - epoch_valid_losses[-1] < 3e-4:
                            train_logger.info('EarlyStopping counter: {} out of {}'.format(cur_patience, all_patience))
                            if cur_patience == all_patience:
                                train_logger.info('Early Stopping!')
                                break
                            cur_patience += 1
                        else:
                            train_logger.info("Saving Model.")
                            torch.save(self.state_dict(), self.save_model_path)
                            best_epoch = epoch
                            cur_patience = 1
                    else:
                        torch.save(self.state_dict(), self.save_model_path)
        #if self.save_model:
            #torch.save(self.state_dict(), self.save_model_path)
        min_valid_loss = min(epoch_valid_losses)
        self.load_state_dict(torch.load(self.save_model_path))
        # test model
        self.eval()
        with torch.no_grad():
            cat_xs = []
            for i, (batch_x, batch_y) in enumerate(test_data):
                batch_x = batch_x.to(device)
                batch_x_reconstruct, _ = self.forward(batch_x)
                cat_xs.append(batch_x_reconstruct)

            cat_xs = torch.cat(cat_xs)
            hifi_output = HIFIOutput(dec_means=cat_xs, best_TN=None, best_FP=None,
                                     best_FN=None, best_TP=None, best_precision=None, best_recall=None,
                                     best_fbeta=None, best_pr_auc=None, best_roc_auc=None, best_cks=None,
                                     min_valid_loss=min_valid_loss)
            return hifi_output


def RunModel(train_filename, test_filename, label_filename, config, ratio):
    negative_sample = True if "noise" in config.dataset else False
    train_data, abnormal_data, abnormal_label = read_dataset(train_filename, test_filename, label_filename,
                                                             normalize=True, file_logger=file_logger, negative_sample=negative_sample, ratio=ratio)
    original_x_dim = abnormal_data.shape[1]

    rolling_train_data = None
    rolling_valid_data = None
    if config.preprocessing:
        if config.use_overlapping:
            if train_data is not None:
                rolling_train_data, rolling_abnormal_data, rolling_abnormal_label = rolling_window_2D(train_data,
                                                                                                      config.rolling_size), rolling_window_2D(
                    abnormal_data, config.rolling_size), rolling_window_2D(abnormal_label, config.rolling_size)
                train_split_idx = int(rolling_train_data.shape[0] * 0.7)
                rolling_train_data, rolling_valid_data = rolling_train_data[:train_split_idx], rolling_train_data[
                                                                                               train_split_idx:]
            else:
                rolling_abnormal_data, rolling_abnormal_label = rolling_window_2D(abnormal_data,
                                                                                  config.rolling_size), rolling_window_2D(
                    abnormal_label, config.rolling_size)
        else:
            if train_data is not None:
                rolling_train_data, rolling_abnormal_data, rolling_abnormal_label = cutting_window_2D(train_data,
                                                                                                      config.rolling_size), cutting_window_2D(
                    abnormal_data, config.rolling_size), cutting_window_2D(abnormal_label, config.rolling_size)
                train_split_idx = int(rolling_train_data.shape[0] * 0.7)
                rolling_train_data, rolling_valid_data = rolling_train_data[:train_split_idx], rolling_train_data[
                                                                                               train_split_idx:]
            else:
                rolling_abnormal_data, rolling_abnormal_label = cutting_window_2D(abnormal_data,
                                                                                  config.rolling_size), cutting_window_2D(
                    abnormal_label, config.rolling_size)
    else:
        if train_data is not None:
            rolling_train_data, rolling_abnormal_data, rolling_abnormal_label = np.expand_dims(train_data,
                                                                                               axis=0), np.expand_dims(
                abnormal_data, axis=0), np.expand_dims(abnormal_label, axis=0)
            train_split_idx = int(rolling_train_data.shape[0] * 0.7)
            rolling_train_data, rolling_valid_data = rolling_train_data[:train_split_idx], rolling_train_data[
                                                                                           train_split_idx:]
        else:
            rolling_abnormal_data, rolling_abnormal_label = np.expand_dims(abnormal_data, axis=0), np.expand_dims(
                abnormal_label, axis=0)

    config.x_dim = rolling_abnormal_data.shape[2]

    model = HIFI(file_name=train_filename, config=config)
    model = model.to(device)
    hifi_output = None
    if train_data is not None and config.robustness == False:
        hifi_output = model.fit(train_input=rolling_train_data, train_label=rolling_train_data,
                                valid_input=rolling_valid_data, valid_label=rolling_valid_data,
                               test_input=rolling_abnormal_data, test_label=rolling_abnormal_label,
                               abnormal_data=abnormal_data, abnormal_label=abnormal_label,
                               original_x_dim=original_x_dim)
    elif train_data is None or config.robustness == True:
        hifi_output = model.fit(train_input=rolling_abnormal_data, train_label=rolling_abnormal_data,
                                valid_input=rolling_valid_data, valid_label=rolling_valid_data,
                                test_input=rolling_abnormal_data, test_label=rolling_abnormal_label,
                               abnormal_data=abnormal_data, abnormal_label=abnormal_label,
                               original_x_dim=original_x_dim)
    # %%
    min_max_scaler = preprocessing.MinMaxScaler()
    if config.preprocessing:
        if config.use_overlapping:
            if config.use_last_point:
                dec_mean_unroll = hifi_output.dec_means.detach().cpu().numpy()[:, -1]
                dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
                x_original_unroll = abnormal_data[config.rolling_size - 1:]
            else:
                dec_mean_unroll = unroll_window_3D(hifi_output.dec_means.detach().cpu().numpy())[::-1]
                dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
                x_original_unroll = abnormal_data[: dec_mean_unroll.shape[0]]

        else:
            dec_mean_unroll = np.reshape(hifi_output.dec_means.detach().cpu().numpy(), (-1, original_x_dim))
            dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
            x_original_unroll = abnormal_data[: dec_mean_unroll.shape[0]]
    else:
        dec_mean_unroll = hifi_output.dec_means.detach().cpu().numpy()
        dec_mean_unroll = np.squeeze(dec_mean_unroll, axis=0)
        dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
        x_original_unroll = abnormal_data

    if config.save_output:
        if not os.path.exists('./outputs/NPY/{}/'.format(config.dataset)):
            os.makedirs('./outputs/NPY/{}/'.format(config.dataset))
        np.save('./outputs/NPY/{}/Dec_HIFI_hdim_{}_rollingsize_{}_{}_pid={}.npy'.format(config.dataset, config.h_dim, config.rolling_size, train_filename.stem, config.pid),
                dec_mean_unroll)

    error = np.sum(x_original_unroll - np.reshape(dec_mean_unroll, [-1, original_x_dim]), axis=1) ** 2
    # final_zscore = zscore(error)
    # np_decision = create_label_based_on_zscore(final_zscore, 2.5, True)
    # np_decision = create_label_based_on_quantile(error, quantile=99)
    SD_Tmin, SD_Tmax = SD_autothreshold(error)
    SD_y_hat = get_labels_by_threshold(error, Tmax=SD_Tmax, use_max=True, use_min=False)
    MAD_Tmin, MAD_Tmax = MAD_autothreshold(error)
    MAD_y_hat = get_labels_by_threshold(error, Tmax=MAD_Tmax, use_max=True, use_min=False)
    IQR_Tmin, IQR_Tmax = IQR_autothreshold(error)
    IQR_y_hat = get_labels_by_threshold(error, Tmax=IQR_Tmax, use_max=True, use_min=False)
    np_decision = {}
    np_decision["SD"] = SD_y_hat
    np_decision["MAD"] = MAD_y_hat
    np_decision["IQR"] = IQR_y_hat

    # TODO metrics computation.

    # %%
    if config.save_figure:
        if original_x_dim == 1:
            plt.figure(figsize=(9, 3))
            plt.plot(x_original_unroll, color='blue', lw=1.5)
            plt.title('Original Data')
            plt.grid(True)
            plt.tight_layout()
            # plt.show()
            plt.savefig('./figures/{}/Ori_HIFI_hdim_{}_rollingsize_{}_{}_pid={}.png'.format(config.dataset, config.h_dim, config.rolling_size, train_filename.stem, config.pid),
                        dpi=600)
            plt.close()

            # Plot decoder output
            plt.figure(figsize=(9, 3))
            plt.plot(dec_mean_unroll, color='blue', lw=1.5)
            plt.title('Decoding Output')
            plt.grid(True)
            plt.tight_layout()
            # plt.show()
            plt.savefig('./figures/{}/Dec_HIFI_hdim_{}_rollingsize_{}_{}_pid={}.png'.format(config.dataset, config.h_dim, config.rolling_size, train_filename.stem, config.pid),
                        dpi=600)
            plt.close()

            t = np.arange(0, abnormal_data.shape[0])
            # markercolors = ['blue' if i == 1 else 'red' for i in abnormal_label[: dec_mean_unroll.shape[0]]]
            # markersize = [4 if i == 1 else 25 for i in abnormal_label[: dec_mean_unroll.shape[0]]]
            # plt.figure(figsize=(9, 3))
            # ax = plt.axes()
            # plt.yticks([0, 0.25, 0.5, 0.75, 1])
            # ax.set_xlim(t[0] - 10, t[-1] + 10)
            # ax.set_ylim(-0.10, 1.10)
            # plt.xlabel('$t$')
            # plt.ylabel('$s$')
            # plt.grid(True)
            # plt.tight_layout()
            # plt.margins(0.1)
            # plt.plot(np.squeeze(abnormal_data[: dec_mean_unroll.shape[0]]), alpha=0.7)
            # plt.scatter(t[: dec_mean_unroll.shape[0]], x_original_unroll[: dec_mean_unroll.shape[0]], s=markersize, c=markercolors)
            # # plt.show()
            # plt.savefig('./figures/{}/VisInp_RAE_{}_pid={}.png'.format(config.dataset, Path(file_name).stem, config.pid), dpi=600)
            # plt.close()

            markercolors = ['blue' for i in range(config.rolling_size - 1)] + ['blue' if i == 1 else 'red' for i in
                                                                               np_decision["SD"]]
            markersize = [4 for i in range(config.rolling_size - 1)] + [4 if i == 1 else 25 for i in np_decision["SD"]]
            plt.figure(figsize=(9, 3))
            ax = plt.axes()
            plt.yticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_xlim(t[0] - 10, t[-1] + 10)
            ax.set_ylim(-0.10, 1.10)
            plt.xlabel('$t$')
            plt.ylabel('$s$')
            plt.grid(True)
            plt.tight_layout()
            plt.margins(0.1)
            plt.plot(np.squeeze(abnormal_data), alpha=0.7)
            plt.scatter(t, abnormal_data, s=markersize, c=markercolors)
            # plt.show()
            plt.savefig(
                './figures/{}/VisOut_HIFI_hdim_{}_rollingsize_{}_SD_{}_pid={}.png'.format(config.dataset, config.h_dim, config.rolling_size, train_filename.stem, config.pid),
                dpi=300)
            plt.close()

            markercolors = ['blue' for i in range(config.rolling_size - 1)] + ['blue' if i == 1 else 'red' for i in
                                                                               np_decision["MAD"]]
            markersize = [4 for i in range(config.rolling_size - 1)] + [4 if i == 1 else 25 for i in np_decision["MAD"]]
            plt.figure(figsize=(9, 3))
            ax = plt.axes()
            plt.yticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_xlim(t[0] - 10, t[-1] + 10)
            ax.set_ylim(-0.10, 1.10)
            plt.xlabel('$t$')
            plt.ylabel('$s$')
            plt.grid(True)
            plt.tight_layout()
            plt.margins(0.1)
            plt.plot(np.squeeze(abnormal_data), alpha=0.7)
            plt.scatter(t, abnormal_data, s=markersize, c=markercolors)
            # plt.show()
            plt.savefig(
                './figures/{}/VisOut_HIFI_hdim_{}_rollingsize_{}_MAD_{}_pid={}.png'.format(config.dataset, config.h_dim, config.rolling_size, train_filename.stem, config.pid),
                dpi=300)
            plt.close()

            markercolors = ['blue' for i in range(config.rolling_size - 1)] + ['blue' if i == 1 else 'red' for i in
                                                                               np_decision["IQR"]]
            markersize = [4 for i in range(config.rolling_size - 1)] + [4 if i == 1 else 25 for i in np_decision["IQR"]]
            plt.figure(figsize=(9, 3))
            ax = plt.axes()
            plt.yticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_xlim(t[0] - 10, t[-1] + 10)
            ax.set_ylim(-0.10, 1.10)
            plt.xlabel('$t$')
            plt.ylabel('$s$')
            plt.grid(True)
            plt.tight_layout()
            plt.margins(0.1)
            plt.plot(np.squeeze(abnormal_data), alpha=0.7)
            plt.scatter(t, abnormal_data, s=markersize, c=markercolors)
            # plt.show()
            plt.savefig(
                './figures/{}/VisOut_HIFI_hdim_{}_rollingsize_{}_IQR_{}_pid={}.png'.format(config.dataset, config.h_dim, config.rolling_size, train_filename.stem, config.pid),
                dpi=300)
            plt.close()
        else:
            file_logger.info('cannot plot image with x_dim > 1')

    if config.use_spot:
        pass
    else:
        pos_label = -1
        TN, FP, FN, TP, precision, recall, f1 = {}, {}, {}, {}, {}, {}, {}
        for threshold_method in np_decision:
            cm = confusion_matrix(y_true=abnormal_label, y_pred=np_decision[threshold_method],
                                  labels=[1, -1])
            TN[threshold_method] = cm[0][0]
            FP[threshold_method] = cm[0][1]
            FN[threshold_method] = cm[1][0]
            TP[threshold_method] = cm[1][1]
            precision[threshold_method] = precision_score(y_true=abnormal_label,
                                                          y_pred=np_decision[threshold_method], pos_label=pos_label)
            recall[threshold_method] = recall_score(y_true=abnormal_label,
                                                    y_pred=np_decision[threshold_method], pos_label=pos_label)
            f1[threshold_method] = f1_score(y_true=abnormal_label,
                                            y_pred=np_decision[threshold_method], pos_label=pos_label)

        fpr, tpr, _ = roc_curve(y_true=abnormal_label, y_score=np.nan_to_num(error),
                                pos_label=pos_label)
        roc_auc = auc(fpr, tpr)
        pre, re, _ = precision_recall_curve(y_true=abnormal_label,
                                            probas_pred=np.nan_to_num(error),
                                            pos_label=pos_label)
        pr_auc = auc(re, pre)
        metrics_result = MetricsResult(TN=TN, FP=FP, FN=FN, TP=TP, precision=precision,
                                       recall=recall, fbeta=f1, pr_auc=pr_auc, roc_auc=roc_auc,
                                       best_TN=hifi_output.best_TN, best_FP=hifi_output.best_FP,
                                       best_FN=hifi_output.best_FN, best_TP=hifi_output.best_TP,
                                       best_precision=hifi_output.best_precision, best_recall=hifi_output.best_recall,
                                       best_fbeta=hifi_output.best_fbeta, best_pr_auc=hifi_output.best_pr_auc,
                                       best_roc_auc=hifi_output.best_roc_auc, best_cks=hifi_output.best_cks,
                                       min_valid_loss=hifi_output.min_valid_loss)
        return metrics_result


if __name__ == '__main__':

    # %%
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, default=0)
    parser.add_argument('--x_dim', type=int, default=1)
    parser.add_argument('--h_dim', type=int, default=128)
    parser.add_argument('--d_inner', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--d_k', type=int, default=16)
    parser.add_argument('--d_v', type=int, default=16)
    parser.add_argument('--dropout', type=int, default=0.1)
    parser.add_argument('--gcn_layers', type=int, default=2)
    parser.add_argument('--gcn_alpha', type=int, default=0.2)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--preprocessing', type=str2bool, default=True)
    parser.add_argument('--use_overlapping', type=str2bool, default=True)
    # rolling_size means window size.
    parser.add_argument('--rolling_size', type=int, default=16)
    # parser.add_argument('--epochs', type=int, default=1000)
    # recommanded by the original paper, we set the epochs to 200.
    parser.add_argument('--epochs', type=int, default=200)
    # milestone_epochs is used to reduce the learning rate.
    parser.add_argument('--milestone_epochs', type=int, default=50)
    parser.add_argument('--kl_start', type=float, default=0.5)
    parser.add_argument('--kl_warmup', type=float, default=10)
    parser.add_argument('--ratio', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--weight_decay', type=float, default=1e-8)
    parser.add_argument('--early_stopping', type=str2bool, default=True)
    parser.add_argument('--loss_function', type=str, default='mse')
    parser.add_argument('--use_clip_norm', type=str2bool, default=True)
    parser.add_argument('--gradient_clip_norm', type=float, default=10)
    parser.add_argument('--use_bidirection', type=str2bool, default=False)
    parser.add_argument('--display_epoch', type=int, default=5)
    parser.add_argument('--save_output', type=str2bool, default=True)
    parser.add_argument('--save_figure', type=str2bool, default=False)
    parser.add_argument('--save_model', type=str2bool, default=True)  # save model
    parser.add_argument('--save_results', type=str2bool, default=True)  # save results
    parser.add_argument('--load_model', type=str2bool, default=False)  # load model
    parser.add_argument('--continue_training', type=str2bool, default=False)
    parser.add_argument('--use_spot', type=str2bool, default=False)
    parser.add_argument('--use_last_point', type=str2bool, default=False)
    parser.add_argument('--save_config', type=str2bool, default=True)
    parser.add_argument('--load_config', type=str2bool, default=False)
    parser.add_argument('--server_run', type=str2bool, default=False)
    parser.add_argument('--robustness', type=str2bool, default=False)
    parser.add_argument('--pid', type=int, default=0)
    args = parser.parse_args()

    #for registered_dataset in dataset2path:
    # HIFI only runs on the datasets, on which the time series is multivariate.
    #for registered_dataset in ["MSL", "SMD", "SMAP"]:
    #for registered_dataset in ["SWAT"]:
    #for registered_dataset in ["MSL", "SMD", "SMAP", "SWAT", "NAB", "AIOps", "Credit", "ECG", "nyc_taxi", "Yahoo"]:
    #for registered_dataset in ["Yahoo"]:
    #for registered_dataset in ["nyc_taxi"]:
    #for registered_dataset in ["ECG"]:
    #for registered_dataset in ["MSL_noise", "NAB_noise"]:
    for registered_dataset in ["MSL", "SMAP", "SMD", "NAB", "AIOps", "Credit", "ECG", "nyc_taxi", "SWAT", "Yahoo"]:

        # the dim in args is useless, which should be deleted in the future version.
        if "noise" in registered_dataset:
            args.dataset = registered_dataset + "_{:.2f}".format(args.ratio)
        else:
            args.dataset = registered_dataset

        if args.load_config:
            config = HIFIConfig(dataset=None, x_dim=None, h_dim=None, d_inner=None, n_layers=None, n_head=None,
                                d_k=None, d_v=None, gcn_layers=None, gcn_alpha=None, k=None, preprocessing=None,
                                use_overlapping=None, rolling_size=None, epochs=None, milestone_epochs=None,
                                lr=None, gamma=None, batch_size=None, weight_decay=None, early_stopping=None,
                                loss_function=None, use_clip_norm=None, gradient_clip_norm=None, display_epoch=None,
                                save_output=None, save_figure=None, save_model=None, load_model=None, continue_training=None,
                                dropout=None, use_spot=None, use_last_point=None, save_config=None, load_config=None,
                                server_run=None, robustness=None, pid=None, save_results=None, kl_start=None, kl_warmup=None)

            try:
                config.import_config('./config/{}/Config_HIFI_hdim_{}_rollingsize_{}_pid={}.json'.format(config.dataset, config.h_dim, config.rolling_size, config.pid))
            except:
                print('There is no config.')
        else:
            config = HIFIConfig(dataset=args.dataset, x_dim=args.x_dim, h_dim=args.h_dim, d_inner=args.d_inner,
                                n_layers=args.n_layers, n_head=args.n_head, d_k=args.d_k, d_v=args.d_v,
                                gcn_layers=args.gcn_layers, gcn_alpha=args.gcn_alpha, k=args.k, preprocessing=args.preprocessing,
                                use_overlapping=args.use_overlapping, rolling_size=args.rolling_size,
                                epochs=args.epochs, milestone_epochs=args.milestone_epochs, lr=args.lr, gamma=args.gamma,
                                batch_size=args.batch_size, weight_decay=args.weight_decay,
                                early_stopping=args.early_stopping, loss_function=args.loss_function,
                                use_clip_norm=args.use_clip_norm, gradient_clip_norm=args.gradient_clip_norm,
                                display_epoch=args.display_epoch, save_output=args.save_output,
                                save_figure=args.save_figure, save_model=args.save_model, load_model=args.load_model,
                                continue_training=args.continue_training, dropout=args.dropout, use_spot=args.use_spot,
                                use_last_point=args.use_last_point, save_config=args.save_config,
                                load_config=args.load_config, server_run=args.server_run, robustness=args.robustness,
                                pid=args.pid, save_results=args.save_results, kl_start=args.kl_start, kl_warmup=args.kl_warmup)

        if args.save_config:
            if not os.path.exists('./config/{}/'.format(config.dataset)):
                os.makedirs('./config/{}/'.format(config.dataset))
            config.export_config('./config/{}/Config_HIFI_hdim_{}_rollingsize_{}_pid={}.json'.format(config.dataset, config.h_dim, config.rolling_size, config.pid))
        # %%
        if config.dataset not in dataset2path:
            raise ValueError("dataset {} is not registered.".format(config.dataset))
        else:
            train_path = dataset2path[config.dataset]["train"]
            test_path = dataset2path[config.dataset]["test"]
            label_path = dataset2path[config.dataset]["test_label"]
        # %%
        #device = torch.device(get_free_device())
        device = torch.device("cuda:3")

        train_logger, file_logger, meta_logger = create_logger(dataset=args.dataset,
                                                               h_dim=config.h_dim,
                                                               rolling_size=config.rolling_size,
                                                               train_logger_name='hifi_train_logger',
                                                               file_logger_name='hifi_file_logger',
                                                               meta_logger_name='hifi_meta_logger',
                                                               model_name='HIFI',
                                                               pid=args.pid)

        # logging setting
        file_logger.info('============================')
        for key, value in vars(args).items():
            file_logger.info(key + ' = {}'.format(value))
        file_logger.info('============================')

        meta_logger.info('============================')
        for key, value in vars(args).items():
            meta_logger.info(key + ' = {}'.format(value))
        meta_logger.info('============================')


        #for train_file in [Path('../datasets/train/SMD/machine-1-1.pkl')]:
        #for train_file in [Path('../datasets/train/MSL/M-1.pkl')]:
        for train_file in train_path.iterdir():
            test_file = test_path / train_file.name
            label_file = label_path / train_file.name
            file_logger.info('============================')
            file_logger.info(train_file)

            metrics_result = RunModel(train_filename=train_file, test_filename=test_file, label_filename=label_file,
                                      config=config, ratio=args.ratio)
            result_dataframe = make_result_dataframe(metrics_result)

            if config.save_results == True:
                if not os.path.exists('./results/{}/'.format(config.dataset)):
                    os.makedirs('./results/{}/'.format(config.dataset))
                result_dataframe.to_csv(
                    './results/{}/Results_HIFI_hdim_{}_rollingsize_{}_{}_pid={}.csv'.format(config.dataset, config.h_dim, config.rolling_size, train_file.stem, config.pid),
                    index=False)
