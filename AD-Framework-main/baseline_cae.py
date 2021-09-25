import argparse
import os
import sqlite3
import matplotlib as mpl
from sklearn import preprocessing

from utils.outputs import CAEOutput
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams["font.size"] = 16
from pathlib import Path
from statistics import mean
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, auc, precision_recall_curve, roc_curve, confusion_matrix
from torch.optim import Adam, lr_scheduler
from utils.config import CAEConfig
from utils.data_provider import get_loader, rolling_window_2D, cutting_window_2D, unroll_window_3D
from utils.device import get_free_device
from utils.logger import create_logger
from utils.metrics import MetricsResult
from utils.utils import str2bool

from utils.data_provider import dataset2path, read_dataset
from utils.metrics import SD_autothreshold, MAD_autothreshold, IQR_autothreshold, get_labels_by_threshold
from utils.utils import make_result_dataframe
from sklearn.metrics import f1_score

class CAE(nn.Module):
    def __init__(self, file_name, config):
        super(CAE, self).__init__()
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
                './save_model/{}/CAE' \
                '_{}_hdim_{}_rollingsize_{}_pid={}.pt'.format(self.dataset, Path(self.file_name).stem, self.h_dim, self.rolling_size, self.pid)
        else:
            self.save_model_path = None

        self.load_model = config.load_model
        if self.load_model:
            self.load_model_path = \
                './save_model/{}/CAE' \
                '_{}_pid={}.pt'.format(self.dataset, Path(self.file_name).stem, self.pid)
        else:
            self.load_model_path = None

        # units
        self.autoencoder = nn.Sequential(
            nn.Conv1d(in_channels=self.x_dim, out_channels=self.h_dim * 8, kernel_size=5, stride=1, padding=2, bias=True),
            nn.PReLU(),
            nn.Conv1d(in_channels=self.h_dim * 8, out_channels=self.h_dim * 4, kernel_size=5, stride=1, padding=2, bias=True),
            nn.PReLU(),
            nn.Conv1d(in_channels=self.h_dim * 4, out_channels=self.h_dim * 2, kernel_size=5, stride=1, padding=2, bias=True),
            nn.PReLU(),
            nn.Conv1d(in_channels=self.h_dim * 2, out_channels=self.h_dim, kernel_size=5, stride=1, padding=2, bias=True),
            nn.PReLU(),
            nn.ConvTranspose1d(in_channels=self.h_dim, out_channels=self.h_dim * 2, kernel_size=5, stride=1, padding=2, bias=True),
            nn.PReLU(),
            nn.ConvTranspose1d(in_channels=self.h_dim * 2, out_channels=self.h_dim * 4, kernel_size=5, stride=1, padding=2, bias=True),
            nn.PReLU(),
            nn.ConvTranspose1d(in_channels=self.h_dim * 4, out_channels=self.h_dim * 8, kernel_size=5, stride=1, padding=2, bias=True),
            nn.PReLU(),
            nn.ConvTranspose1d(in_channels=self.h_dim * 8, out_channels=self.x_dim, kernel_size=5, stride=1, padding=2, bias=True)
        )

    def reset_parameters(self):
        for p in self.autoencoder.parameters():
            torch.nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.autoencoder(x)
        return x

    def fit(self, train_input, train_label, valid_input, valid_label, test_input, test_label, abnormal_data, abnormal_label, original_x_dim):
        loss_fn = nn.MSELoss()
        opt = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = lr_scheduler.StepLR(optimizer=opt, step_size=self.milestone_epochs, gamma=self.gamma)
        # get batch data
        if self.preprocessing == False or self.use_overlapping == False:
            train_input, test_input, test_label = np.expand_dims(np.transpose(train_input), axis=0), np.expand_dims(np.transpose(test_input), axis=0), np.expand_dims(test_label, axis=0)
        train_data = get_loader(input=train_input, label=None, batch_size=self.batch_size, from_numpy=True,
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
            epoch_valid_losses = []
            for epoch in range(self.epochs):
                self.train()
                train_losses = []
                # opt.zero_grad()
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
                    if epoch > 1:
                        if -1e-7 < epoch_losses[-1] - epoch_losses[-2] < 1e-7:
                            train_logger.info('early break')
                            break
        else:
            # train model
            epoch_losses = []
            epoch_valid_losses = []
            for epoch in range(self.epochs):
                self.train()
                train_losses = []
                # opt.zero_grad()
                for i, (batch_x, batch_y) in enumerate(train_data):
                    opt.zero_grad()
                    batch_x = batch_x.to(device)
                    batch_x_reconstruct = self.forward(batch_x)
                    batch_loss = loss_fn(batch_x_reconstruct, batch_x)
                    batch_loss.backward()
                    opt.step()
                    sched.step()
                    train_losses.append(batch_loss.item())
                epoch_losses.append(mean(train_losses))

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

        min_valid_loss = min(epoch_valid_losses)
        self.load_state_dict(torch.load(self.save_model_path))
        # test model
        self.eval()
        with torch.no_grad():
            cat_xs = []
            for i, (batch_x, batch_y) in enumerate(test_data):
                batch_x = batch_x.to(device)
                batch_x_reconstruct = self.forward(batch_x)
                cat_xs.append(batch_x_reconstruct.detach().cpu())

            cat_xs = torch.cat(cat_xs)
            cae_output = CAEOutput(dec_means=cat_xs, best_TN=None, best_FP=None, best_FN=None, best_TP=None,
                                   best_precision=None, best_recall=None, best_fbeta=None,
                                   best_pr_auc=None, best_roc_auc=None, best_cks=None, min_valid_loss=min_valid_loss,
                                   memory_usage_nvidia=None, training_time=None, testing_time=None)
            return cae_output


def RunModel(train_filename, test_filename, label_filename, config, ratio):
    negative_sample = True if "noise" in config.dataset else False
    train_data, abnormal_data, abnormal_label = read_dataset(train_filename, test_filename, label_filename,
                                                             normalize=True, file_logger=file_logger, negative_sample=negative_sample, ratio=ratio)

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

    config.x_dim = rolling_abnormal_data.shape[1]

    model = CAE(file_name=train_filename, config=config)
    model = model.to(device)
    cae_output = None
    if train_data is not None and config.robustness == False:
        cae_output = model.fit(train_input=rolling_train_data, train_label=rolling_train_data,
                               valid_input=rolling_valid_data, valid_label=rolling_valid_data,
                               test_input=rolling_abnormal_data, test_label=rolling_abnormal_label,
                               abnormal_data=abnormal_data, abnormal_label=abnormal_label, original_x_dim=original_x_dim)
    elif train_data is None or config.robustness == True:
        cae_output = model.fit(train_input=rolling_abnormal_data, train_label=rolling_abnormal_data,
                               valid_input=rolling_valid_data, valid_label=rolling_valid_data,
                               test_input=rolling_abnormal_data, test_label=rolling_abnormal_label,
                               abnormal_data=abnormal_data, abnormal_label=abnormal_label, original_x_dim=original_x_dim)
    # %%
    min_max_scaler = preprocessing.MinMaxScaler()
    if config.preprocessing:
        if config.use_overlapping:
            if config.use_last_point:
                dec_mean_unroll = cae_output.dec_means.detach().cpu().numpy()[:, -1]
                dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
                x_original_unroll = abnormal_data[config.rolling_size-1:]
            else:
                dec_mean_unroll = unroll_window_3D(np.reshape(cae_output.dec_means.detach().cpu().numpy(), (-1, config.rolling_size, original_x_dim)))[::-1]
                dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
                x_original_unroll = abnormal_data[: dec_mean_unroll.shape[0]]

        else:
            dec_mean_unroll = np.reshape(cae_output.dec_means.detach().cpu().numpy(), (-1, original_x_dim))
            dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
            x_original_unroll = abnormal_data[: dec_mean_unroll.shape[0]]
    else:
        dec_mean_unroll = cae_output.dec_means.detach().cpu().numpy()
        dec_mean_unroll = np.transpose(np.squeeze(dec_mean_unroll, axis=0))
        dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
        x_original_unroll = abnormal_data

    if config.save_output:
        if not os.path.exists('./outputs/NPY/{}/'.format(config.dataset)):
            os.makedirs('./outputs/NPY/{}/'.format(config.dataset))
        np.save('./outputs/NPY/{}/Dec_CAE_hdim_{}_rollingsize_{}_{}_pid={}.npy'.format(config.dataset, config.h_dim, config.rolling_size, Path(train_filename).stem, config.pid), dec_mean_unroll)

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
        file_logger.info('save_figure has been dropped.')

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
            precision[threshold_method] = precision_score(y_true=abnormal_label, y_pred=np_decision[threshold_method],
                                                          pos_label=pos_label)
            recall[threshold_method] = recall_score(y_true=abnormal_label, y_pred=np_decision[threshold_method],
                                                    pos_label=pos_label)
            f1[threshold_method] = f1_score(y_true=abnormal_label, y_pred=np_decision[threshold_method],
                                            pos_label=pos_label)

        fpr, tpr, _ = roc_curve(y_true=abnormal_label, y_score=np.nan_to_num(error), pos_label=pos_label)
        roc_auc = auc(fpr, tpr)
        pre, re, _ = precision_recall_curve(y_true=abnormal_label, probas_pred=np.nan_to_num(error),
                                            pos_label=pos_label)
        pr_auc = auc(re, pre)
        metrics_result = MetricsResult(TN=TN, FP=FP, FN=FN, TP=TP, precision=precision,
                                       recall=recall, fbeta=f1, pr_auc=pr_auc, roc_auc=roc_auc,
                                       best_TN=cae_output.best_TN, best_FP=cae_output.best_FP,
                                       best_FN=cae_output.best_FN, best_TP=cae_output.best_TP,
                                       best_precision=cae_output.best_precision, best_recall=cae_output.best_recall,
                                       best_fbeta=cae_output.best_fbeta, best_pr_auc=cae_output.best_pr_auc,
                                       best_roc_auc=cae_output.best_roc_auc, best_cks=cae_output.best_cks,
                                       min_valid_loss=cae_output.min_valid_loss)
        return metrics_result

if __name__ == '__main__':
    conn = sqlite3.connect('./experiments.db')
    cursor_obj = conn.cursor()

    # %%
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, default=0)
    parser.add_argument('--x_dim', type=int, default=1)
    parser.add_argument('--h_dim', type=int, default=16)
    parser.add_argument('--preprocessing', type=str2bool, default=True)
    parser.add_argument('--use_overlapping', type=str2bool, default=True)
    parser.add_argument('--ratio', type=float, default=0.05)
    parser.add_argument('--rolling_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--milestone_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--weight_decay', type=float, default=1e-8)
    parser.add_argument('--early_stopping', type=str2bool, default=True)
    parser.add_argument('--loss_function', type=str, default='mse')
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

    #for registered_dataset in ["MSL", "SMAP", "SMD", "NAB", "AIOps", "Credit", "ECG", "nyc_taxi", "SWAT", "Yahoo"]:
    #for registered_dataset in ["Credit", "ECG", "nyc_taxi", "SWAT", "Yahoo"]:
    #for registered_dataset in ["SMAP", "SMD", "NAB", "AIOps", "Credit", "ECG", "nyc_taxi", "SWAT", "Yahoo"]:
    #for registered_dataset in ["SMD", "NAB", "AIOps", "Credit", "ECG", "nyc_taxi", "SWAT", "Yahoo"]:
    #for registered_dataset in ["MSL"]:
    for registered_dataset in ["Yahoo"]:

        # the dim in args is useless, which should be deleted in the future version.
        if "noise" in registered_dataset:
            args.dataset = registered_dataset + "_{:.2f}".format(args.ratio)
        else:
            args.dataset = registered_dataset

        if args.load_config:
            config = CAEConfig(dataset=None, x_dim=None, h_dim=None, preprocessing=None, use_overlapping=None,
                               rolling_size=None, epochs=None, milestone_epochs=None, lr=None, gamma=None, batch_size=None,
                               weight_decay=None, early_stopping=None, loss_function=None, display_epoch=None,
                               save_output=None, save_figure=None, save_model=None, load_model=None, continue_training=None,
                               dropout=None, use_spot=None, use_last_point=None, save_config=None, load_config=None,
                               server_run=None, robustness=None, pid=None, save_results=None)
            try:
                config.import_config('./config/{}/Config_CAE_pid={}.json'.format(config.dataset, config.pid))
            except:
                print('There is no config.')
        else:
            config = CAEConfig(dataset=args.dataset, x_dim=args.x_dim, h_dim=args.h_dim, preprocessing=args.preprocessing,
                               use_overlapping=args.use_overlapping, rolling_size=args.rolling_size, epochs=args.epochs,
                               milestone_epochs=args.milestone_epochs, lr=args.lr, gamma=args.gamma,
                               batch_size=args.batch_size, weight_decay=args.weight_decay,
                               early_stopping=args.early_stopping, loss_function=args.loss_function,
                               display_epoch=args.display_epoch, save_output=args.save_output, save_figure=args.save_figure,
                               save_model=args.save_model, load_model=args.load_model,
                               continue_training=args.continue_training, dropout=args.dropout, use_spot=args.use_spot,
                               use_last_point=args.use_last_point, save_config=args.save_config,
                               load_config=args.load_config, server_run=args.server_run, robustness=args.robustness,
                               pid=args.pid, save_results=args.save_results)
        if args.save_config:
            if not os.path.exists('./config/{}/'.format(config.dataset)):
                os.makedirs('./config/{}/'.format(config.dataset))
            config.export_config('./config/{}/Config_CAE_pid={}.json'.format(config.dataset, config.pid))
        # %%
        if config.dataset not in dataset2path:
            raise ValueError("dataset {} is not registered.".format(config.dataset))
        else:
            train_path = dataset2path[config.dataset]["train"]
            test_path = dataset2path[config.dataset]["test"]
            label_path = dataset2path[config.dataset]["test_label"]
        # %%
        #device = torch.device(get_free_device())
        device = torch.device("cuda:0")

        train_logger, file_logger, meta_logger = create_logger(dataset=args.dataset,
                                                               h_dim=config.h_dim,
                                                               rolling_size=config.rolling_size,
                                                               train_logger_name='cae_train_logger',
                                                               file_logger_name='cae_file_logger',
                                                               meta_logger_name='cae_meta_logger',
                                                               model_name='CAE',
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

        #for train_file in train_path.iterdir():
        for train_file in train_path.iterdir():
            #if train_file.name != "real_62.pkl":
            #    continue
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
                    './results/{}/Results_CAE_hdim_{}_rollingsize_{}_{}_pid={}.csv'.format(config.dataset, config.h_dim,
                                                                                           config.rolling_size,
                                                                                           train_file.stem, config.pid), index=False)
