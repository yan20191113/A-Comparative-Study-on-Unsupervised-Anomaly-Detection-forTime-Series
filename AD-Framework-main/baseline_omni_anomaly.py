import argparse
import os
import matplotlib as mpl
from sklearn import preprocessing
from torch.autograd import Variable
from utils.outputs import VRAEOutput
mpl.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from statistics import mean
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, auc, precision_recall_curve, roc_curve, confusion_matrix
from torch.optim import Adam, lr_scheduler
from utils.config import VRAEConfig
from utils.data_provider import rolling_window_2D, get_loader, cutting_window_2D, unroll_window_3D
from utils.logger import create_logger
from utils.metrics import MetricsResult
from utils.utils import str2bool
from utils.data_provider import dataset2path, read_dataset
from utils.metrics import SD_autothreshold, MAD_autothreshold, IQR_autothreshold, get_labels_by_threshold
from utils.utils import make_result_dataframe
from sklearn.metrics import f1_score


class NormalizingFlow(nn.Module):
    def __init__(self, latent_dim, K):
        super(NormalizingFlow, self).__init__()
        self.latent_dim = latent_dim
        self.K = K
        w_k = []
        b_k = []
        u_k = []
        for k in range(self.K):
            w_k.append(nn.Parameter(torch.rand([1, self.latent_dim])).to(device))
            b_k.append(nn.Parameter(torch.rand(1)).to(device))
            u_k.append(nn.Parameter(torch.rand([1, self.latent_dim])).to(device))
        self.w = w_k
        self.b = b_k
        self.u = u_k

    def forward(self, z):
        for k in range(self.K):
            z = z + self.u[k] * torch.tanh(torch.mm(z, self.w[k].T) + self.b[k])
        return z

# class NormalizingFlow(nn.Module):
#     def __init__(self, latent_dim=20, K=16):
#         super().__init__()
#         self.transforms = nn.ModuleList([PlanarTransform(dim=latent_dim) for k in range(K)])
#
#     def forward(self, z, logdet=False):
#         zK = z
#         SLDJ = 0.
#         for transform in self.transforms:
#             out = transform(zK, logdet=logdet)
#             if logdet:
#                 SLDJ += out[1]
#                 zK = out[0]
#             else:
#                 zK = out
#
#         if logdet:
#             return zK, SLDJ
#         return zK
#
#
# class PlanarTransform(nn.Module):
#     def __init__(self, latent_dim=20):
#         super().__init__()
#         self.u = nn.Parameter(torch.randn(1, latent_dim) * 0.01)
#         self.w = nn.Parameter(torch.randn(1, latent_dim) * 0.01)
#         self.b = nn.Parameter(torch.randn(()) * 0.01)
#
#     def m(self, x):
#         return -1 + torch.log(1 + torch.exp(x))
#
#     def h(self, x):
#         return torch.tanh(x)
#
#     def h_prime(self, x):
#         return 1 - torch.tanh(x) ** 2
#
#     def forward(self, z, logdet=False):
#         # z.size() = batch x dim
#         u_dot_w = (self.u @ self.w.t()).view(())
#         w_hat = self.w / torch.norm(self.w, p=2)  # Unit vector in the direction of w
#         u_hat = (self.m(u_dot_w) - u_dot_w) * (w_hat) + self.u  # 1 x dim
#         affine = z @ self.w.t() + self.b
#         z_next = z + u_hat * self.h(affine)  # batch x dim
#         if logdet:
#             psi = self.h_prime(affine) * self.w  # batch x dim
#             LDJ = -torch.log(torch.abs(psi @ u_hat.t() + 1) + 1e-8)  # batch x 1
#             return z_next, LDJ
#         return z_next

class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim, dense_dim, z_dim, rolling_size, use_PNF, PNF_layers):
        super(Encoder, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.dense_dim = dense_dim
        self.z_dim = z_dim
        self.rolling_size = rolling_size

        self.gru_cell = nn.GRUCell(input_size=self.x_dim, hidden_size=self.h_dim)
        self.use_PNF = use_PNF
        self.PNF_layers = PNF_layers

        if self.use_PNF:
            self.PNF = nn.ModuleList(NormalizingFlow(latent_dim=self.z_dim, K=self.PNF_layers).to(device) for _ in range(self.rolling_size))

        self.phi_enc = nn.Sequential(
            nn.Linear(self.h_dim + self.z_dim, self.dense_dim),
            nn.ReLU())

        self.enc_means = nn.Sequential(
            nn.Linear(self.dense_dim, self.z_dim),
            nn.Sigmoid())

        self.enc_stds = nn.Sequential(
            nn.Linear(self.dense_dim, self.z_dim),
            nn.Softplus())

    def reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps).to(device)
        return eps.mul(std).add_(mean)

    def forward(self, input):
        h_i = Variable(torch.zeros(input.shape[0], self.h_dim), requires_grad=True).to(device)
        z_i = Variable(torch.zeros(input.shape[0], self.z_dim), requires_grad=True).to(device)
        # hs = []
        z_means = []
        z_stds = []
        zs = []
        for i in range(input.shape[1]):
            h_i = self.gru_cell(input[:, i], (h_i))
            h_z = torch.cat([h_i, z_i], dim=1)
            phi_z = self.phi_enc(h_z)
            z_mean = self.enc_means(phi_z)
            z_std = self.enc_stds(phi_z)
            z = self.reparameterized_sample(mean=z_mean, std=z_std)
            if self.use_PNF:
                z = self.PNF[i](z)
            # hs.append(h_i)
            z_means.append(z_mean)
            z_stds.append(z_std)
            zs.append(z)
        # return the list to tensor
        zs = torch.stack(zs)
        z_means = torch.stack(z_means)
        z_stds = torch.stack(z_stds)
        return zs.permute(1, 0, 2), z_means.permute(1, 0, 2), z_stds.permute(1, 0, 2)


class Decoder(nn.Module):
    def __init__(self, x_dim, h_dim, dense_dim, z_dim, rolling_size):
        super(Decoder, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.dense_dim = dense_dim
        self.z_dim = z_dim
        self.rolling_size = rolling_size

        self.gru_cell = nn.GRUCell(input_size=self.z_dim, hidden_size=self.h_dim)

        self.phi_dec = nn.Sequential(
            nn.Linear(self.h_dim, self.dense_dim),
            nn.ReLU())

        self.dec_means = nn.Sequential(
            nn.Linear(self.dense_dim, self.x_dim),
            nn.Sigmoid())

        self.dec_stds = nn.Sequential(
            nn.Linear(self.dense_dim, self.x_dim),
            nn.Softplus())

    def reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps).to(device)
        return eps.mul(std).add_(mean)

    def forward(self, input):
        h_i = Variable(torch.zeros(input.shape[0], self.h_dim), requires_grad=True).to(device)
        # hs = []
        x_means = []
        x_stds = []
        xs = []
        for i in range(input.shape[1]):
            h_i = self.gru_cell(input[:, i], (h_i))
            phi_x = self.phi_dec(h_i)
            x_mean = self.dec_means(phi_x)
            x_std = self.dec_stds(phi_x)
            x = self.reparameterized_sample(mean=x_mean, std=x_std)
            # hs.append(h_i)
            x_means.append(x_mean)
            x_stds.append(x_std)
            xs.append(x)
            # return the list to tensor
        xs = torch.stack(xs)
        x_means = torch.stack(x_means)
        x_stds = torch.stack(x_stds)
        return xs.permute(1, 0, 2), x_means.permute(1, 0, 2), x_stds.permute(1, 0, 2)


class OMNIANOMALY(nn.Module):
    def __init__(self, file_name, config):
        super(OMNIANOMALY, self).__init__()
        # file info
        self.dataset = config.dataset
        self.file_name = file_name

        # dim info
        self.x_dim = config.x_dim
        self.h_dim = config.h_dim
        self.dense_dim = config.h_dim
        self.z_dim = config.z_dim

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
        self.lmbda = config.lmbda
        self.use_clip_norm = config.use_clip_norm
        self.gradient_clip_norm = config.gradient_clip_norm

        # dropout
        self.dropout = config.dropout
        self.continue_training = config.continue_training

        self.robustness = config.robustness

        # layers
        self.rnn_layers = config.rnn_layers
        self.use_PNF = config.use_PNF
        self.PNF_layers = config.PNF_layers
        self.use_bidirection = config.use_bidirection
        self.use_seq2seq = config.use_seq2seq
        self.force_teaching = config.force_teaching
        self.force_teaching_threshold = config.force_teaching_threshold
        self.flexible_h = config.flexible_h
        self.alpha = config.alpha

        # pid
        self.pid = config.pid

        self.encoder = Encoder(x_dim=self.x_dim, h_dim=self.h_dim, dense_dim=self.dense_dim, z_dim=self.z_dim,
                               rolling_size=self.rolling_size, use_PNF=self.use_PNF, PNF_layers=self.PNF_layers)
        self.decoder = Decoder(x_dim=self.x_dim, h_dim=self.h_dim, dense_dim=self.dense_dim, z_dim=self.z_dim,
                               rolling_size=self.rolling_size,)

        self.save_model = config.save_model
        if self.save_model:
            if not os.path.exists('./save_model/{}/'.format(self.dataset)):
                os.makedirs('./save_model/{}/'.format(self.dataset))
            self.save_model_path = \
                './save_model/{}/OMNIANOMALY_hdim_{}_rollingsize_{}' \
                '_{}_pid={}.pt'.format(self.dataset, config.h_dim, config.rolling_size, Path(self.file_name).stem, self.pid)
        else:
            self.save_model_path = None

        self.load_model = config.load_model
        if self.load_model:
            self.load_model_path = \
                './save_model/{}/OMNIANOMALY_hdim_{}_rollingsize_{}' \
                '_{}_pid={}.pt'.format(self.dataset, config.h_dim, config.rolling_size, Path(self.file_name).stem, self.pid)
        else:
            self.load_model_path = None

    def nll_bernoulli(self, theta, x):
        return - torch.sum(x * torch.log(theta) + (1 - x) * torch.log(1 - theta))

    def nll_gaussian_1(self, mean, std, x):
        return 0.5 * (torch.sum(std) + torch.sum(((x - mean) / std.mul(0.5).exp_()) ** 2))  # Owned definition

    def nll_gaussian_2(self, mean, std, x):
        return torch.sum(torch.log(std) + (x - mean).pow(2) / (2 * std.pow(2)))

    def mse(self, mean, x):
        return torch.nn.functional.mse_loss(input=mean, target=x, reduction='sum')

    def kld_gaussian(self, mean_1, std_1, mean_2, std_2):
        if mean_2 is not None and std_2 is not None:
            kl_loss = 0.5 * torch.sum(
                2 * torch.log(std_2) - 2 * torch.log(std_1) + (std_1.pow(2) + (mean_1 - mean_2).pow(2)) / std_2.pow(2) - 1)
        else:
            kl_loss = -0.5 * torch.sum(1 + std_1 - mean_1.pow(2) - std_1.exp())
        return kl_loss

    def kld_gaussian_1(self, mean, std):
        """Using std to compute KLD"""
        return -0.5 * torch.sum(1 + torch.log(std) - mean.pow(2) - std)

    def kld_gaussian_2(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""
        kld_element = (2 * torch.log(std_2) - 2 * torch.log(std_1) + (std_1.pow(2) + (mean_1 - mean_2).pow(2)) / std_2.pow(2) - 1)
        return 0.5 * torch.sum(kld_element)

    def forward(self, input):
        kld_loss = 0  # KL in ELBO
        nll_loss = 0  # -loglikihood in ELBO
        z, z_mean, z_std = self.encoder(input)
        x, x_mean, x_std = self.decoder(z)
        if self.loss_function == 'nll':
            nll_loss += self.nll_gaussian_1(x=input, mean=x_mean, std=x_std)
        elif self.loss_function == 'mse':
            nll_loss += self.mse(x=input, mean=x_mean)
        # nll_loss += self.mse(mean=x_mean, x=input)
        kld_loss += self.kld_gaussian(mean_1=z_mean, std_1=z_std, mean_2=None, std_2=None) + self.kld_gaussian(mean_1=x_mean, std_1=x_std, mean_2=None, std_2=None)
        return nll_loss, kld_loss, z, z_mean, z_std, x, x_mean, x_std

    def fit(self, train_input, train_label, valid_input, valid_label, test_input, test_label, abnormal_data, abnormal_label, original_x_dim):
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
            epoch_losses = []
            epoch_nll_losses = []
            epoch_kld_losses = []
            epoch_valid_losses = []
            epoch_valid_nll_losses = []
            epoch_valid_kld_losses = []
            for epoch in range(self.epochs):
                batch_train_losses = []
                batch_nll_losses = []
                batch_kld_losses = []
                # opt.zero_grad()
                self.train()
                for i, (batch_x, batch_y) in enumerate(train_data):
                    opt.zero_grad()
                    nll_loss, kld_loss, batch_z, batch_z_mean, batch_z_std, batch_x_reconstruct, batch_x_mean, batch_x_std = self.forward(input=batch_x.to(device))
                    batch_loss = nll_loss + self.lmbda * epoch * kld_loss
                    batch_loss.backward()
                    if self.use_clip_norm:
                        torch.nn.utils.clip_grad_norm_(list(self.parameters()), self.gradient_clip_norm)
                    opt.step()
                    sched.step()
                    batch_nll_losses.append(nll_loss.item())
                    batch_kld_losses.append(kld_loss.item())
                    batch_train_losses.append(batch_loss.item())
                epoch_losses.append(mean(batch_train_losses))
                epoch_nll_losses.append(mean(batch_nll_losses))
                epoch_kld_losses.append(mean(batch_kld_losses))
                if epoch % self.display_epoch == 0:
                    train_logger.info('epoch = {} , train loss = {} , nll loss = {}, kld loss = {}'.format(epoch, epoch_losses[-1], epoch_nll_losses[-1], epoch_kld_losses[-1]))

                batch_valid_losses = []
                batch_valid_nll_losses = []
                batch_valid_kld_losses = []
                with torch.no_grad():
                    self.eval()
                    for i, (val_batch_x, val_batch_y) in enumerate(valid_data):
                        val_nll_loss, val_kld_loss, val_batch_z, val_batch_z_mean, val_batch_z_std, val_batch_x_reconstruct, val_batch_x_mean, val_batch_x_std = self.forward(
                            input=val_batch_x.to(device))
                        val_batch_loss = val_nll_loss + self.lmbda * epoch * val_kld_loss
                        batch_valid_nll_losses.append(val_nll_loss.item())
                        batch_valid_kld_losses.append(val_kld_loss.item())
                        batch_valid_losses.append(val_batch_loss.item())
                    epoch_valid_losses.append(mean(batch_valid_losses))
                    epoch_valid_nll_losses.append(mean(batch_valid_nll_losses))
                    epoch_valid_kld_losses.append(mean(batch_valid_kld_losses))
                if epoch % self.display_epoch == 0:
                    train_logger.info(
                        'epoch = {} , valid loss = {} , nll loss = {}, kld loss = {}'.format(epoch, epoch_valid_losses[-1],
                                                                                             epoch_valid_nll_losses[-1],
                                                                                             epoch_valid_kld_losses[-1]))

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
            epoch_losses = []
            epoch_nll_losses = []
            epoch_kld_losses = []
            epoch_valid_losses = []
            epoch_valid_nll_losses = []
            epoch_valid_kld_losses = []
            for epoch in range(self.epochs):
                batch_train_losses = []
                batch_nll_losses = []
                batch_kld_losses = []
                # opt.zero_grad()
                self.train()
                for i, (batch_x, batch_y) in enumerate(train_data):
                    # All outputs have shape [sequence_length|rolling_size, batch_size, h_dim|z_dim]
                    opt.zero_grad()
                    nll_loss, kld_loss, batch_z, batch_z_mean, batch_z_std, batch_x_reconstruct, batch_x_mean, batch_x_std = self.forward(input=batch_x.to(device))
                    batch_loss = nll_loss + self.lmbda * epoch * kld_loss
                    batch_loss.backward()
                    if self.use_clip_norm:
                        torch.nn.utils.clip_grad_norm_(list(self.parameters()), self.gradient_clip_norm)
                    opt.step()
                    sched.step()
                    batch_nll_losses.append(nll_loss.item())
                    batch_kld_losses.append(kld_loss.item())
                    batch_train_losses.append(batch_loss.item())
                epoch_losses.append(mean(batch_train_losses))
                epoch_nll_losses.append(mean(batch_nll_losses))
                epoch_kld_losses.append(mean(batch_kld_losses))
                if epoch % self.display_epoch == 0:
                    train_logger.info('epoch = {} , train loss = {} , nll loss = {}, kld loss = {}'.format(epoch, epoch_losses[-1], epoch_nll_losses[-1], epoch_kld_losses[-1]))

                batch_valid_losses = []
                batch_valid_nll_losses = []
                batch_valid_kld_losses = []
                with torch.no_grad():
                    self.eval()
                    for i, (val_batch_x, val_batch_y) in enumerate(valid_data):
                        val_nll_loss, val_kld_loss, val_batch_z, val_batch_z_mean, val_batch_z_std, val_batch_x_reconstruct, val_batch_x_mean, val_batch_x_std = self.forward(
                            input=val_batch_x.to(device))
                        val_batch_loss = val_nll_loss + self.lmbda * epoch * val_kld_loss
                        batch_valid_nll_losses.append(val_nll_loss.item())
                        batch_valid_kld_losses.append(val_kld_loss.item())
                        batch_valid_losses.append(val_batch_loss.item())
                    epoch_valid_losses.append(mean(batch_valid_losses))
                    epoch_valid_nll_losses.append(mean(batch_valid_nll_losses))
                    epoch_valid_kld_losses.append(mean(batch_valid_kld_losses))
                if epoch % self.display_epoch == 0:
                    train_logger.info(
                        'epoch = {} , valid loss = {} , nll loss = {}, kld loss = {}'.format(epoch, epoch_valid_losses[-1],
                                                                                             epoch_valid_nll_losses[-1],
                                                                                             epoch_valid_kld_losses[-1]))

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
        # test model
        min_valid_loss = min(epoch_valid_losses)
        self.load_state_dict(torch.load(self.save_model_path))
        self.eval()
        with torch.no_grad():
            cat_zs = []
            cat_z_means = []
            cat_z_stds = []
            cat_xs = []
            cat_x_means = []
            cat_x_stds = []
            kld_loss = 0
            nll_loss = 0

            for i, (batch_x, batch_y) in enumerate(test_data):
                nll_loss, kld_loss, batch_z, batch_z_mean, batch_z_std, batch_x_reconstruct, batch_x_mean, batch_x_std = self.forward(input=batch_x.to(device))
                cat_zs.append(batch_z.cpu())
                cat_z_means.append(batch_z_mean.cpu())
                cat_z_stds.append(batch_z_std.cpu())
                cat_xs.append(batch_x_reconstruct.cpu())
                cat_x_means.append(batch_x_mean.cpu())
                cat_x_stds.append(batch_x_std.cpu())
                kld_loss += kld_loss
                nll_loss += nll_loss

            cat_zs = torch.cat(cat_zs)
            cat_z_means = torch.cat(cat_z_means)
            cat_z_stds = torch.cat(cat_z_stds)
            cat_xs = torch.cat(cat_xs)
            cat_x_means = torch.cat(cat_x_means)
            cat_x_stds = torch.cat(cat_x_stds)

            vrae_output = VRAEOutput(best_TN=None, best_FP=None, best_FN=None, best_TP=None,
                                     best_precision=None, best_recall=None, best_fbeta=None,
                                     best_pr_auc=None, best_roc_auc=None, best_cks=None, zs=cat_zs,
                                     z_infer_means=cat_z_means, z_infer_stds=cat_z_stds, decs=cat_xs,
                                     dec_means=cat_x_means, dec_stds=cat_x_stds, kld_loss=kld_loss, nll_loss=nll_loss,
                                     min_valid_loss=min_valid_loss)
            return vrae_output

def RunModel(train_filename, test_filename, label_filename, config, ratio):

    negative_sample = True if "noise" in config.dataset else False
    train_data, abnormal_data, abnormal_label = read_dataset(train_filename, test_filename, label_filename,
                                                             normalize=True, file_logger=file_logger,
                                                             negative_sample=negative_sample, ratio=ratio)

    if abnormal_data.shape[0] < config.rolling_size:
        train_logger.warning("test data is less than rolling_size! Ignore the current data!")
        TN, FP, FN, TP, precision, recall, f1 = {}, {}, {}, {}, {}, {}, {}
        for threshold_method in ["SD", "MAD", "IQR"]:
            TN[threshold_method] = -1
            FP[threshold_method] = -1
            FN[threshold_method] = -1
            TP[threshold_method] = -1
            precision[threshold_method] = -1
            recall[threshold_method] = -1
            f1[threshold_method] = -1
        roc_auc = -1
        pr_auc = -1
        metrics_result = MetricsResult(TN=TN, FP=FP, FN=FN, TP=TP, precision=precision,
                                       recall=recall, fbeta=f1, pr_auc=pr_auc, roc_auc=roc_auc)
        return metrics_result

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

    model = OMNIANOMALY(file_name=train_filename, config=config)
    model = model.to(device)
    omnianomaly_output = None
    if train_data is not None and config.robustness == False:
        omnianomaly_output = model.fit(train_input=rolling_train_data, train_label=rolling_train_data,
                                       valid_input=rolling_valid_data, valid_label=rolling_valid_data,
                                       test_input=rolling_abnormal_data, test_label=rolling_abnormal_label,
                                       abnormal_data=abnormal_data, abnormal_label=abnormal_label,
                                       original_x_dim=original_x_dim)
    elif train_data is None or config.robustness == True:
        omnianomaly_output = model.fit(train_input=rolling_abnormal_data, train_label=rolling_abnormal_data,
                                       valid_input=rolling_valid_data, valid_label=rolling_valid_data,
                                       test_input=rolling_abnormal_data, test_label=rolling_abnormal_label,
                                       abnormal_data=abnormal_data, abnormal_label=abnormal_label,
                                       original_x_dim=original_x_dim)
    # %%
    min_max_scaler = preprocessing.MinMaxScaler()
    if config.preprocessing:
        if config.use_overlapping:
            if config.use_last_point:
                dec_mean_unroll = omnianomaly_output.dec_means.detach().cpu().numpy()[:, -1]
                dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
                latent_mean_unroll = omnianomaly_output.zs.detach().cpu().numpy()
                x_original_unroll = abnormal_data[config.rolling_size - 1:]
            else:
                dec_mean_unroll = unroll_window_3D(np.reshape(omnianomaly_output.dec_means.detach().cpu().numpy(), (-1, config.rolling_size, original_x_dim)))[::-1]
                dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
                latent_mean_unroll = omnianomaly_output.zs.detach().cpu().numpy()
                x_original_unroll = abnormal_data[: dec_mean_unroll.shape[0]]

        else:
            dec_mean_unroll = np.reshape(omnianomaly_output.dec_means.detach().cpu().numpy(), (-1, original_x_dim))
            dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
            latent_mean_unroll = omnianomaly_output.zs.detach().cpu().numpy()
            x_original_unroll = abnormal_data[: dec_mean_unroll.shape[0]]
    else:
        dec_mean_unroll = omnianomaly_output.dec_means.detach().cpu().numpy()
        dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
        latent_mean_unroll = omnianomaly_output.zs.detach().cpu().numpy()
        x_original_unroll = abnormal_data

    # %%
    if config.save_output:
        if not os.path.exists('./outputs/NPY/{}/'.format(config.dataset)):
            os.makedirs('./outputs/NPY/{}/'.format(config.dataset))
        np.save('./outputs/NPY/{}/Dec_OMNIANOMALY_hdim_{}_rollingsize_{}_{}_pid={}.npy'.format(config.dataset, config.h_dim, config.rolling_size, train_filename.stem, config.pid), dec_mean_unroll)
        np.save('./outputs/NPY/{}/Latent_OMNIANOMALY_hdim_{}_rollingsize_{}_{}_pid={}.npy'.format(config.dataset, config.h_dim, config.rolling_size, train_filename.stem, config.pid), latent_mean_unroll)
        if config.use_seq2seq:
            file_logger.info('under development')

    error = np.sum(x_original_unroll - np.reshape(dec_mean_unroll, [-1, original_x_dim]), axis=1) ** 2
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
            plt.savefig('./figures/{}/Ori_OMNIANOMALY_hdim_{}_rollingsize_{}_{}_pid={}.png'.format(config.dataset, config.h_dim, config.rolling_size, train_filename.stem, config.pid), dpi=600)
            plt.close()

            # Plot decoder output
            plt.figure(figsize=(9, 3))
            plt.plot(dec_mean_unroll, color='blue', lw=1.5)
            plt.title('Decoding Output')
            plt.grid(True)
            plt.tight_layout()
            # plt.show()
            plt.savefig('./figures/{}/Dec_OMNIANOMALY_hdim_{}_rollingsize_{}_{}_pid={}.png'.format(config.dataset, config.h_dim, config.rolling_size, train_filename.stem, config.pid), dpi=600)
            plt.close()

            t = np.arange(0, abnormal_data.shape[0])
            markercolors = ['blue' if i == 1 else 'red' for i in abnormal_label[: dec_mean_unroll.shape[0]]]
            markersize = [4 if i == 1 else 25 for i in abnormal_label[: dec_mean_unroll.shape[0]]]
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
            plt.plot(np.squeeze(abnormal_data[: dec_mean_unroll.shape[0]]), alpha=0.7)
            plt.scatter(t[: dec_mean_unroll.shape[0]], x_original_unroll[: np_decision["SD"].shape[0]], s=markersize, c=markercolors)
            # plt.show()
            plt.savefig('./figures/{}/VisInp_OMNIANOMALY_hdim_{}_rollingsize_{}_{}_pid={}.png'.format(config.dataset, config.h_dim, config.rolling_size, train_filename.stem, config.pid), dpi=600)
            plt.close()

            markercolors = ['blue' if i == 1 else 'red' for i in np_decision["SD"]]
            markersize = [4 if i == 1 else 25 for i in np_decision["SD"]]
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
            plt.plot(np.squeeze(abnormal_data[: dec_mean_unroll.shape[0]]), alpha=0.7)
            plt.scatter(t[: np_decision["SD"].shape[0]], abnormal_data[: np_decision["SD"].shape[0]], s=markersize, c=markercolors)
            # plt.show()
            plt.savefig('./figures/{}/VisOut_OMNIANOMALY_hdim_{}_rollingsize_{}_SD_{}_pid={}.png'.format(config.dataset, config.h_dim, config.rolling_size, train_filename.stem, config.pid), dpi=600)
            plt.close()

            markercolors = ['blue' if i == 1 else 'red' for i in np_decision["MAD"]]
            markersize = [4 if i == 1 else 25 for i in np_decision["MAD"]]
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
            plt.plot(np.squeeze(abnormal_data[: dec_mean_unroll.shape[0]]), alpha=0.7)
            plt.scatter(t[: np_decision["MAD"].shape[0]], abnormal_data[: np_decision["MAD"].shape[0]], s=markersize, c=markercolors)
            # plt.show()
            plt.savefig('./figures/{}/VisOut_OMNIANOMALY_hdim_{}_rollingsize_{}_MAD_{}_pid={}.png'.format(config.dataset, config.h_dim, config.rolling_size, train_filename.stem, config.pid), dpi=600)
            plt.close()

            markercolors = ['blue' if i == 1 else 'red' for i in np_decision["IQR"]]
            markersize = [4 if i == 1 else 25 for i in np_decision["IQR"]]
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
            plt.plot(np.squeeze(abnormal_data[: dec_mean_unroll.shape[0]]), alpha=0.7)
            plt.scatter(t[: np_decision["IQR"].shape[0]], abnormal_data[: np_decision["IQR"].shape[0]], s=markersize, c=markercolors)
            # plt.show()
            plt.savefig('./figures/{}/VisOut_OMNIANOMALY_hdim_{}_rollingsize_{}_IQR_{}_pid={}.png'.format(config.dataset, config.h_dim, config.rolling_size, train_filename.stem, config.pid), dpi=600)
            plt.close()
        else:
            file_logger.info('cannot plot image with x_dim > 1')

    if config.use_spot:
        pass
    else:
        pos_label = -1
        TN, FP, FN, TP, precision, recall, f1 = {}, {}, {}, {}, {}, {}, {}
        for threshold_method in np_decision:
            cm = confusion_matrix(y_true=abnormal_label, y_pred=np_decision[threshold_method], labels=[1, -1])
            TN[threshold_method] = cm[0][0]
            FP[threshold_method] = cm[0][1]
            FN[threshold_method] = cm[1][0]
            TP[threshold_method] = cm[1][1]
            precision[threshold_method] = precision_score(y_true=abnormal_label, y_pred=np_decision[threshold_method], pos_label=pos_label)
            recall[threshold_method] = recall_score(y_true=abnormal_label, y_pred=np_decision[threshold_method], pos_label=pos_label)
            f1[threshold_method] = f1_score(y_true=abnormal_label, y_pred=np_decision[threshold_method], pos_label=pos_label)

        fpr, tpr, _ = roc_curve(y_true=abnormal_label, y_score=np.nan_to_num(error), pos_label=pos_label)
        roc_auc = auc(fpr, tpr)
        pre, re, _ = precision_recall_curve(y_true=abnormal_label, probas_pred=np.nan_to_num(error),
                                            pos_label=pos_label)
        pr_auc = auc(re, pre)
        metrics_result = MetricsResult(TN=TN, FP=FP, FN=FN, TP=TP, precision=precision,
                                       recall=recall, fbeta=f1, pr_auc=pr_auc, roc_auc=roc_auc,
                                       best_TN=omnianomaly_output.best_TN, best_FP=omnianomaly_output.best_FP,
                                       best_FN=omnianomaly_output.best_FN, best_TP=omnianomaly_output.best_TP,
                                       best_precision=omnianomaly_output.best_precision, best_recall=omnianomaly_output.best_recall,
                                       best_fbeta=omnianomaly_output.best_fbeta, best_pr_auc=omnianomaly_output.best_pr_auc,
                                       best_roc_auc=omnianomaly_output.best_roc_auc, best_cks=omnianomaly_output.best_cks,
                                       min_valid_loss=omnianomaly_output.min_valid_loss)
        return metrics_result

if __name__ == '__main__':
    # %%
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, default=0)
    parser.add_argument('--x_dim', type=int, default=1)
    parser.add_argument('--h_dim', type=int, default=16)
    parser.add_argument('--z_dim', type=int, default=16)
    parser.add_argument('--preprocessing', type=str2bool, default=True)
    parser.add_argument('--ratio', type=float, default=0.05)
    parser.add_argument('--use_overlapping', type=str2bool, default=True)
    parser.add_argument('--rolling_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--milestone_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--weight_decay', type=float, default=1e-8)
    parser.add_argument('--early_stopping', type=str2bool, default=True)
    parser.add_argument('--loss_function', type=str, default='mse')
    parser.add_argument('--rnn_layers', type=int, default=1)
    parser.add_argument('--lmbda', type=float, default=0.0001)
    parser.add_argument('--use_clip_norm', type=str2bool, default=True)
    parser.add_argument('--gradient_clip_norm', type=float, default=10)
    parser.add_argument('--use_PNF', type=str2bool, default=True)
    parser.add_argument('--PNF_layers', type=int, default=10)
    parser.add_argument('--use_bidirection', type=str2bool, default=False)
    parser.add_argument('--use_seq2seq', type=str2bool, default=False)
    parser.add_argument('--force_teaching', type=str2bool, default=False)
    parser.add_argument('--force_teaching_threshold', type=float, default=0.75)
    parser.add_argument('--flexible_h', type=str2bool, default=False)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.005)
    parser.add_argument('--display_epoch', type=int, default=5)
    parser.add_argument('--save_output', type=str2bool, default=True)
    parser.add_argument('--save_figure', type=str2bool, default=False)
    parser.add_argument('--save_model', type=str2bool, default=True)  # save model
    parser.add_argument('--save_results', type=str2bool, default=True)  # save results
    parser.add_argument('--load_model', type=str2bool, default=False)  # load model
    parser.add_argument('--continue_training', type=str2bool, default=False)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--use_spot', type=str2bool, default=False)
    parser.add_argument('--use_last_point', type=str2bool, default=False)
    parser.add_argument('--save_config', type=str2bool, default=True)
    parser.add_argument('--load_config', type=str2bool, default=False)
    parser.add_argument('--server_run', type=str2bool, default=False)
    parser.add_argument('--robustness', type=str2bool, default=False)
    parser.add_argument('--pid', type=int, default=0)
    args = parser.parse_args()
    #["MSL", "SMAP", "SMD", "NAB", "AIOps", "Credit", "ECG", "nyc_taxi", "Yahoo", "SWAT"]
    #for registered_dataset in ["MSL_noise", "NAB_noise"]:
    for registered_dataset in ["Credit", "NAB", "ECG", "SMAP"]:

        # the dim in args is useless, which should be deleted in the future version.
        if "noise" in registered_dataset:
            args.dataset = registered_dataset + "_{:.2f}".format(args.ratio)
        else:
            args.dataset = registered_dataset

        if args.load_config:
            config = VRAEConfig(dataset=None, x_dim=None, h_dim=None, z_dim=None, preprocessing=None, use_overlapping=None,
                                rolling_size=None, epochs=None, milestone_epochs=None, lr=None, gamma=None, batch_size=None,
                                weight_decay=None, early_stopping=None, loss_function=None, rnn_layers=None, lmbda=None,
                                use_clip_norm=None, gradient_clip_norm=None, use_PNF=None, PNF_layers=None,
                                use_bidirection=None, use_seq2seq=None, force_teaching=None, force_teaching_threshold=None,
                                flexible_h=None, alpha=None, beta=None, display_epoch=None, save_output=None,
                                save_figure=None, save_model=None, load_model=None, continue_training=None, dropout=None,
                                use_spot=None, use_last_point=None, save_config=None,  load_config=None, server_run=None,
                                robustness=None, pid=None, save_results=None)
            try:
                config.import_config('./config/{}/Config_OMNIANOMALY_hdim_{}_rollingsize_{}_pid={}.json'.format(config.dataset, config.h_dim, config.rolling_size, config.pid))
            except:
                print('There is no config.')
        else:
            config = VRAEConfig(dataset=args.dataset, x_dim=args.x_dim, h_dim=args.h_dim, z_dim=args.z_dim,
                                preprocessing=args.preprocessing, use_overlapping=args.use_overlapping,
                                rolling_size=args.rolling_size, epochs=args.epochs, milestone_epochs=args.milestone_epochs,
                                lr=args.lr, gamma=args.gamma, batch_size=args.batch_size, weight_decay=args.weight_decay,
                                early_stopping=args.early_stopping, loss_function=args.loss_function,
                                lmbda=args.lmbda, use_clip_norm=args.use_clip_norm,
                                gradient_clip_norm=args.gradient_clip_norm, rnn_layers=args.rnn_layers,
                                use_PNF=args.use_PNF, PNF_layers=args.PNF_layers, use_bidirection=args.use_bidirection,
                                use_seq2seq=args.use_seq2seq, force_teaching=args.force_teaching,
                                force_teaching_threshold=args.force_teaching_threshold, flexible_h=args.flexible_h,
                                alpha=args.alpha, beta=args.beta, display_epoch=args.display_epoch,
                                save_output=args.save_output, save_figure=args.save_figure, save_model=args.save_model,
                                load_model=args.load_model, continue_training=args.continue_training, dropout=args.dropout,
                                use_spot=args.use_spot, use_last_point=args.use_last_point, save_config=args.save_config,
                                load_config=args.load_config, server_run=args.server_run, robustness=args.robustness,
                                pid=args.pid, save_results=args.save_results)
        if args.save_config:
            if not os.path.exists('./config/{}/'.format(config.dataset)):
                os.makedirs('./config/{}/'.format(config.dataset))
            config.export_config('./config/{}/Config_OMNIANOMALY_hdim_{}_rollingsize_{}_pid={}.json'.format(config.dataset, config.h_dim, config.rolling_size, config.pid))
        # %%
        if config.dataset not in dataset2path:
            raise ValueError("dataset {} is not registered.".format(config.dataset))
        else:
            train_path = dataset2path[config.dataset]["train"]
            test_path = dataset2path[config.dataset]["test"]
            label_path = dataset2path[config.dataset]["test_label"]
        # %%
        #device = torch.device(get_free_device())
        gpu_id = 1
        device = torch.device("cuda:{}".format(gpu_id))

        train_logger, file_logger, meta_logger = create_logger(dataset=args.dataset,
                                                               h_dim=config.h_dim,
                                                               rolling_size=config.rolling_size,
                                                               train_logger_name='omnianomaly_train_logger',
                                                               file_logger_name='omnianomaly_file_logger',
                                                               meta_logger_name='omnianomaly_meta_logger',
                                                               model_name='OMNIANOMALY',
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

        for train_file in train_path.iterdir():
            # for train_file in [Path('../datasets/train/MSL/M-1.pkl')]:
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
                    './results/{}/Results_OMNIANOMALY_hdim_{}_rollingsize_{}_{}_pid={}.csv'.format(config.dataset, config.h_dim, config.rolling_size, train_file.stem, config.pid),
                    index=False)
