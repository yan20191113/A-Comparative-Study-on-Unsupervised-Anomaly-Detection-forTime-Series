from pathlib import Path
import matplotlib as mpl
from sklearn import preprocessing
from torch.autograd import Variable
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams["font.size"] = 16
from sklearn.metrics import recall_score, precision_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from utils.config import RAEConfig
from utils.device import get_free_device
from utils.outputs import RAEOutput
from utils.utils import str2bool
import os
import torch
import torch.nn as nn
import argparse
from utils.logger import create_logger
from utils.metrics import MetricsResult
import numpy as np
from utils.data_provider import get_loader, rolling_window_2D, cutting_window_2D, unroll_window_3D
from statistics import mean
from torch.optim import Adam, lr_scheduler
from utils.data_provider import dataset2path, read_dataset
from utils.metrics import SD_autothreshold, MAD_autothreshold, IQR_autothreshold, get_labels_by_threshold
from utils.utils import make_result_dataframe
from sklearn.metrics import f1_score
import time


class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim, sequence_length, use_bidirection=False, batch_size=1, rnn_layers=1):
        super(Encoder, self).__init__()
        # dim info
        self.x_dim = x_dim
        self.h_dim = h_dim

        # sequence info
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.use_bidirection = use_bidirection

        # layers
        self.rnn_layers = rnn_layers

        # units
        self.gru_cell = nn.GRUCell(self.x_dim, self.h_dim, bias=True)

    def forward(self, input):
        h_i = Variable(torch.zeros(input.shape[0], self.h_dim), requires_grad=True).to(device)
        for i in range(input.shape[1]):
            h_i = self.gru_cell(input[:, i], (h_i))
        return h_i


class Decoder(nn.Module):
    def __init__(self, x_dim, h_dim, sequence_length, use_bidirection=False, batch_size=1, rnn_layers=1):
        super(Decoder, self).__init__()
        # dim info
        self.x_dim = x_dim
        self.h_dim = h_dim

        # sequence info
        self.batch_size = batch_size
        self.sequence_length = sequence_length

        # layers
        self.gru_layers = rnn_layers
        self.use_bidirection = use_bidirection

        # units
        self.gru_cell = nn.GRUCell(self.x_dim, self.h_dim, bias=True)
        # self.list_Linear = nn.ModuleList([nn.Linear(in_features=self.h_dim, out_features=self.x_dim) for _ in range(self.sequence_length)])
        self.linear = nn.Linear(in_features=self.h_dim, out_features=self.x_dim)

    def forward(self, input, h_enc):
        GO = Variable(torch.zeros(h_enc.shape[0], self.x_dim), requires_grad=False).to(device)
        output = []
        h_i = h_enc
        for i in range(input.shape[1]):
            if i == 0:
                h_i = self.gru_cell(GO, h_i)
                # projected_h_i = self.list_Linear[i](h_i)
                projected_h_i = self.linear(h_i)
                output.append(projected_h_i)
            else:
                h_i = self.gru_cell(output[i - 1], h_i)
                # projected_h_i = self.list_Linear[i](h_i)
                projected_h_i = self.linear(h_i)
                output.append(projected_h_i)
        decoded_output = torch.stack(output)
        return decoded_output.permute(1, 0, 2)


class RAE(nn.Module):
    def __init__(self, file_name, config):
        super(RAE, self).__init__()
        # file info
        self.dataset = config.dataset
        self.file_name = file_name

        # dim info
        self.x_dim = config.x_dim
        self.h_dim = config.h_dim

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

        # dropout
        self.dropout = config.dropout
        self.continue_training = config.continue_training

        self.robustness = config.robustness

        # layers
        self.rnn_layers = config.rnn_layers
        self.use_bidirection = config.use_bidirection
        self.force_teaching = config.force_teaching
        self.force_teaching_threshold = config.force_teaching_threshold

        # pid
        self.pid = config.pid

        self.save_model = config.save_model
        if self.save_model:
            if not os.path.exists('./save_model/{}/'.format(self.dataset)):
                os.makedirs('./save_model/{}/'.format(self.dataset))
            self.save_model_path = \
                './save_model/{}/RAE_hdim_{}_rollingsize_{}' \
                '_{}_pid={}.pt'.format(self.dataset, config.h_dim, config.rolling_size, Path(self.file_name).stem, self.pid)
        else:
            self.save_model_path = None

        self.load_model = config.load_model
        if self.load_model:
            self.load_model_path = \
                './save_model/{}/RAE_hdim_{}_rollingsize_{}' \
                '_{}_pid={}.pt'.format(self.dataset, config.h_dim, config.rolling_size, Path(self.file_name).stem, self.pid)
        else:
            self.load_model_path = None

        # units
        self.encoder = Encoder(x_dim=self.x_dim, h_dim=self.h_dim, sequence_length=self.rolling_size, use_bidirection=False, batch_size=self.batch_size, rnn_layers=self.rnn_layers)
        self.decoder = Decoder(x_dim=self.x_dim, h_dim=self.h_dim, sequence_length=self.rolling_size, use_bidirection=False, batch_size=self.batch_size, rnn_layers=self.rnn_layers)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, Encoder):
                self.encoder = Encoder(use_bidirection=False, batch_size=self.batch_size, rnn_layers=self.rnn_layers)
            elif isinstance(m, Decoder):
                self.decoder = Decoder(use_bidirection=False, batch_size=self.batch_size, rnn_layers=self.rnn_layers)

    def forward(self, input):
        h_enc = self.encoder(input)
        decoded_output = self.decoder(input, h_enc)
        return decoded_output

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
        training_time, testing_time = 0, 0
        if self.load_model == True and self.continue_training == False:
            epoch_valid_losses = [-1]
            start_time = time.time()
            self.load_state_dict(torch.load(self.load_model_path))
        elif self.load_model == True and self.continue_training == True:
            self.load_state_dict(torch.load(self.load_model_path))
            # train model
            start_time = time.time()
            epoch_losses = []
            epoch_valid_losses = []
            for epoch in range(self.epochs):
                train_losses = []
                # opt.zero_grad()
                self.train()
                for i, (batch_x, batch_y) in enumerate(train_data):
                    opt.zero_grad()
                    batch_x = batch_x.to(device)
                    batch_x_reconstruct = self.forward(batch_x)
                    batch_loss = loss_fn(batch_x_reconstruct, batch_x)
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
                        val_batch_x_reconstruct = self.forward(val_batch_x)
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
            epoch_losses = []
            epoch_valid_losses = []
            start_time = time.time()
            for epoch in range(self.epochs):
                train_losses = []
                # opt.zero_grad()
                self.train()
                for i, (batch_x, batch_y) in enumerate(train_data):
                    opt.zero_grad()
                    batch_x = batch_x.to(device)
                    batch_x_reconstruct = self.forward(batch_x)
                    batch_loss = loss_fn(batch_x_reconstruct, batch_x)
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
                        val_batch_x_reconstruct = self.forward(val_batch_x)
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
        training_time = time.time() - start_time
        min_valid_loss = min(epoch_valid_losses)
        self.load_state_dict(torch.load(self.save_model_path))
        # load the trained model
        self.eval()
        start_time = time.time()
        with torch.no_grad():
            cat_xs = []
            for i, (batch_x, batch_y) in enumerate(test_data):
                batch_x = batch_x.to(device)
                batch_x_reconstruct = self.forward(batch_x)
                cat_xs.append(batch_x_reconstruct)

            cat_xs = torch.cat(cat_xs)
            testing_time = time.time() - start_time
            rae_output = RAEOutput(dec_means=cat_xs, best_TN=None, best_FP=None, best_FN=None, best_TP=None,
                                   best_precision=None, best_recall=None, best_fbeta=None,
                                   best_pr_auc=None, best_roc_auc=None, best_cks=None, min_valid_loss=min_valid_loss,
                                   training_time=training_time, testing_time=testing_time, memory_usage_nvidia=None)
            return rae_output


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
                rolling_train_data, rolling_abnormal_data, rolling_abnormal_label = rolling_window_2D(train_data, config.rolling_size), rolling_window_2D(abnormal_data, config.rolling_size), rolling_window_2D(abnormal_label, config.rolling_size)
                train_split_idx = int(rolling_train_data.shape[0] * 0.7)
                rolling_train_data, rolling_valid_data = rolling_train_data[:train_split_idx], rolling_train_data[train_split_idx:]
            else:
                rolling_abnormal_data, rolling_abnormal_label = rolling_window_2D(abnormal_data, config.rolling_size), rolling_window_2D(abnormal_label, config.rolling_size)
        else:
            if train_data is not None:
                rolling_train_data, rolling_abnormal_data, rolling_abnormal_label = cutting_window_2D(train_data, config.rolling_size), cutting_window_2D(abnormal_data, config.rolling_size), cutting_window_2D(abnormal_label, config.rolling_size)
                train_split_idx = int(rolling_train_data.shape[0] * 0.7)
                rolling_train_data, rolling_valid_data = rolling_train_data[:train_split_idx], rolling_train_data[train_split_idx:]
            else:
                rolling_abnormal_data, rolling_abnormal_label = cutting_window_2D(abnormal_data, config.rolling_size), cutting_window_2D(abnormal_label, config.rolling_size)
    else:
        if train_data is not None:
            rolling_train_data, rolling_abnormal_data, rolling_abnormal_label = np.expand_dims(train_data, axis=0), np.expand_dims(abnormal_data, axis=0), np.expand_dims(abnormal_label, axis=0)
            train_split_idx = int(rolling_train_data.shape[0] * 0.7)
            rolling_train_data, rolling_valid_data = rolling_train_data[:train_split_idx], rolling_train_data[train_split_idx:]
        else:
            rolling_abnormal_data, rolling_abnormal_label = np.expand_dims(abnormal_data, axis=0), np.expand_dims(abnormal_label, axis=0)

    config.x_dim = rolling_abnormal_data.shape[2]

    model = RAE(file_name=train_filename, config=config)
    model = model.to(device)
    rae_output = None
    if train_data is not None and config.robustness == False:
        rae_output = model.fit(train_input=rolling_train_data, train_label=rolling_train_data,
                               valid_input=rolling_valid_data, valid_label=rolling_valid_data,
                               test_input=rolling_abnormal_data, test_label=rolling_abnormal_label,
                               abnormal_data=abnormal_data, abnormal_label=abnormal_label, original_x_dim=original_x_dim)
    elif train_data is None or config.robustness == True:
        rae_output = model.fit(train_input=rolling_abnormal_data, train_label=rolling_abnormal_data,
                               valid_input=rolling_valid_data, valid_label=rolling_valid_data,
                               test_input=rolling_abnormal_data, test_label=rolling_abnormal_label,
                               abnormal_data=abnormal_data, abnormal_label=abnormal_label, original_x_dim=original_x_dim)
    # %%
    min_max_scaler = preprocessing.MinMaxScaler()
    if config.preprocessing:
        if config.use_overlapping:
            if config.use_last_point:
                dec_mean_unroll = rae_output.dec_means.detach().cpu().numpy()[:, -1]
                dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
                x_original_unroll = abnormal_data[config.rolling_size - 1:]
            else:
                # the unroll_window_3D will recover the shape as abnormal_data
                # and we only use the [config.rolling_size-1:] to calculate the error, in which we ignore
                # the first config.rolling_size time steps.
                dec_mean_unroll = unroll_window_3D(rae_output.dec_means.detach().cpu().numpy())[::-1]
                dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
                x_original_unroll = abnormal_data[: dec_mean_unroll.shape[0]]

        else:
            dec_mean_unroll = np.reshape(rae_output.dec_means.detach().cpu().numpy(), (-1, original_x_dim))
            dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
            x_original_unroll = abnormal_data[: dec_mean_unroll.shape[0]]
    else:
        dec_mean_unroll = rae_output.dec_means.detach().cpu().numpy()
        dec_mean_unroll = np.squeeze(dec_mean_unroll, axis=0)
        dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
        x_original_unroll = abnormal_data

    if config.save_output:
        if not os.path.exists('./outputs/NPY/{}/'.format(config.dataset)):
            os.makedirs('./outputs/NPY/{}/'.format(config.dataset))
        np.save('./outputs/NPY/{}/Dec_RAE_hdim_{}_rollingsize_{}_{}_pid={}.npy'.format(config.dataset, config.h_dim, config.rolling_size, train_filename.stem, config.pid), dec_mean_unroll)

    error = np.sum(x_original_unroll - np.reshape(dec_mean_unroll, [-1, original_x_dim]), axis=1) ** 2
    # final_zscore = zscore(error)
    # np_decision = create_label_based_on_zscore(final_zscore, 2.5, True)
    #np_decision = create_label_based_on_quantile(error, quantile=99)
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
            plt.savefig('./figures/{}/Ori_RAE_hdim_{}_rollingsize_{}_{}_pid={}.png'.format(config.dataset, config.h_dim, config.rolling_size, train_filename.stem, config.pid), dpi=600)
            plt.close()

            # Plot decoder output
            plt.figure(figsize=(9, 3))
            plt.plot(dec_mean_unroll, color='blue', lw=1.5)
            plt.title('Decoding Output')
            plt.grid(True)
            plt.tight_layout()
            # plt.show()
            plt.savefig('./figures/{}/Dec_RAE_hdim_{}_rolling_size_{}_{}_pid={}.png'.format(config.dataset, config.h_dim, config.rolling_size, train_filename.stem, config.pid), dpi=600)
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

            markercolors = ['blue' for i in range(config.rolling_size - 1)] + ['blue' if i == 1 else 'red' for i in np_decision["SD"]]
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
            plt.savefig('./figures/{}/VisOut_RAE_hdim_{}_rollingsize_{}_SD_{}_pid={}.png'.format(config.dataset, config.h_dim, config.rolling_size, train_filename.stem, config.pid), dpi=300)
            plt.close()

            markercolors = ['blue' for i in range(config.rolling_size - 1)] + ['blue' if i == 1 else 'red' for i in np_decision["MAD"]]
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
            plt.savefig('./figures/{}/VisOut_RAE_hdim_{}_rollingsize_{}_MAD_{}_pid={}.png'.format(config.dataset, config.h_dim, config.rolling_size, train_filename.stem, config.pid), dpi=300)
            plt.close()

            markercolors = ['blue' for i in range(config.rolling_size - 1)] + ['blue' if i == 1 else 'red' for i in np_decision["IQR"]]
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
            plt.savefig('./figures/{}/VisOut_RAE_hdim_{}_rollingsize_{}_IQR_{}_pid={}.png'.format(config.dataset, config.h_dim, config.rolling_size, train_filename.stem, config.pid), dpi=300)
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
                                       best_TN=rae_output.best_TN, best_FP=rae_output.best_FP,
                                       best_FN=rae_output.best_FN, best_TP=rae_output.best_TP,
                                       best_precision=rae_output.best_precision, best_recall=rae_output.best_recall,
                                       best_fbeta=rae_output.best_fbeta, best_pr_auc=rae_output.best_pr_auc,
                                       best_roc_auc=rae_output.best_roc_auc, best_cks=rae_output.best_cks,
                                       min_valid_loss=rae_output.min_valid_loss)
        return metrics_result

if __name__ == '__main__':

    # %%
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, default=0)
    parser.add_argument('--x_dim', type=int, default=1)
    parser.add_argument('--h_dim', type=int, default=64)
    parser.add_argument('--preprocessing', type=str2bool, default=True)
    parser.add_argument('--use_overlapping', type=str2bool, default=True)
    # rolling_size means window size.
    parser.add_argument('--rolling_size', type=int, default=32)
    #parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=200)
    # milestone_epochs is used to reduce the learning rate.
    parser.add_argument('--milestone_epochs', type=int, default=50)
    parser.add_argument('--ratio', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--weight_decay', type=float, default=1e-8)
    parser.add_argument('--early_stopping', type=str2bool, default=True)
    parser.add_argument('--loss_function', type=str, default='mse')
    parser.add_argument('--rnn_layers', type=int, default=1)
    parser.add_argument('--use_clip_norm', type=str2bool, default=True)
    parser.add_argument('--gradient_clip_norm', type=float, default=10)
    parser.add_argument('--use_bidirection', type=str2bool, default=False)
    parser.add_argument('--force_teaching', type=str2bool, default=False)
    parser.add_argument('--force_teaching_threshold', type=float, default=0.75)
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

    for registered_dataset in ["MSL", "SMAP", "SMD", "NAB", "AIOps", "Credit", "ECG", "nyc_taxi", "SWAT", "Yahoo"]:
    #for registered_dataset in ["NAB"]:
        # the dim in args is useless, which should be deleted in the future version.
        if "noise" in registered_dataset:
            args.dataset = registered_dataset + "_{:.2f}".format(args.ratio)
        else:
            args.dataset = registered_dataset

        if args.load_config:
            config = RAEConfig(dataset=None, x_dim=None, h_dim=None, preprocessing=None, use_overlapping=None,
                               rolling_size=None, epochs=None, milestone_epochs=None, lr=None, gamma=None, batch_size=None,
                               weight_decay=None, early_stopping=None, loss_function=None, rnn_layers=None,
                               use_clip_norm=None, gradient_clip_norm=None, use_bidirection=None, force_teaching=None,
                               force_teaching_threshold=None, display_epoch=None, save_output=None, save_figure=None,
                               save_model=None, load_model=None, continue_training=None, dropout=None, use_spot=None,
                               use_last_point=None, save_config=None, load_config=None, server_run=None, robustness=None,
                               pid=None, save_results=None)
            try:
                config.import_config('./config/{}/Config_RAE_hdim_{}_rollingsize_{}_pid={}.json'.format(config.dataset, config.h_dim, config.rolling_size, config.pid))
            except:
                print('There is no config.')
        else:
            config = RAEConfig(dataset=args.dataset, x_dim=args.x_dim, h_dim=args.h_dim, preprocessing=args.preprocessing,
                               use_overlapping=args.use_overlapping, rolling_size=args.rolling_size, epochs=args.epochs,
                               milestone_epochs=args.milestone_epochs, lr=args.lr, gamma=args.gamma,
                               batch_size=args.batch_size, weight_decay=args.weight_decay,
                               early_stopping=args.early_stopping, loss_function=args.loss_function,
                               use_clip_norm=args.use_clip_norm, gradient_clip_norm=args.gradient_clip_norm,
                               rnn_layers=args.rnn_layers, use_bidirection=args.use_bidirection,
                               force_teaching=args.force_teaching, force_teaching_threshold=args.force_teaching_threshold,
                               display_epoch=args.display_epoch, save_output=args.save_output, save_figure=args.save_figure,
                               save_model=args.save_model, load_model=args.load_model,
                               continue_training=args.continue_training, dropout=args.dropout, use_spot=args.use_spot,
                               use_last_point=args.use_last_point, save_config=args.save_config,
                               load_config=args.load_config, server_run=args.server_run, robustness=args.robustness,
                               pid=args.pid, save_results=args.save_results)
        if args.save_config:
            if not os.path.exists('./config/{}/'.format(config.dataset)):
                os.makedirs('./config/{}/'.format(config.dataset))
            config.export_config('./config/{}/Config_RAE_hdim_{}_rollingsize_{}_pid={}.json'.format(config.dataset, config.h_dim, config.rolling_size, config.pid))
        # %%
        if config.dataset not in dataset2path:
            raise ValueError("dataset {} is not registered.".format(config.dataset))
        else:
            train_path = dataset2path[config.dataset]["train"]
            test_path = dataset2path[config.dataset]["test"]
            label_path = dataset2path[config.dataset]["test_label"]
        # %%
        device = torch.device(get_free_device())

        train_logger, file_logger, meta_logger = create_logger(dataset=args.dataset,
                                                               h_dim=config.h_dim,
                                                               rolling_size=config.rolling_size,
                                                               train_logger_name='rae_train_logger',
                                                               file_logger_name='rae_file_logger',
                                                               meta_logger_name='rae_meta_logger',
                                                               model_name='RAE',
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
        #for train_file in [Path('../datasets/train/MSL/M-1.pkl')]:
            test_file = test_path / train_file.name
            label_file = label_path / train_file.name
            file_logger.info('============================')
            file_logger.info(train_file)

            metrics_result = RunModel(train_filename=train_file, test_filename=test_file, label_filename=label_file, config=config, ratio=args.ratio)
            result_dataframe = make_result_dataframe(metrics_result)

            if config.save_results == True:
                if not os.path.exists('./results/{}/'.format(config.dataset)):
                    os.makedirs('./results/{}/'.format(config.dataset))
                result_dataframe.to_csv('./results/{}/Results_RAE_hdim_{}_rollingsize_{}_{}_pid={}.csv'.format(config.dataset, config.h_dim, config.rolling_size, train_file.stem, config.pid),
                                        index=False)


