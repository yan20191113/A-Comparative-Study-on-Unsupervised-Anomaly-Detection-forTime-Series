import argparse
import os
import matplotlib as mpl
from sklearn import preprocessing
from utils.device import get_free_device
from utils.outputs import BEATGANOutput
mpl.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from statistics import mean
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, auc, precision_recall_curve, roc_curve, confusion_matrix
from torch.optim import Adam, lr_scheduler
from utils.config import BEATGANConfig
from utils.data_provider import rolling_window_2D, get_loader, cutting_window_2D, unroll_window_3D
from utils.logger import create_logger
from utils.metrics import MetricsResult
from utils.utils import str2bool

from utils.data_provider import dataset2path, read_dataset
from utils.metrics import SD_autothreshold, MAD_autothreshold, IQR_autothreshold, get_labels_by_threshold
from utils.utils import make_result_dataframe
from sklearn.metrics import f1_score


class GeneratorCNN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, sequence_length):
        super(GeneratorCNN, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.sequence_length = sequence_length
        self.enc = nn.Sequential(
            # input is (nc) x 320
            nn.Conv1d(in_channels=self.x_dim, out_channels=self.h_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(num_features=self.h_dim),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 160
            nn.Conv1d(in_channels=self.h_dim, out_channels=self.h_dim * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(num_features=self.h_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 80
            nn.Conv1d(in_channels=self.h_dim * 2, out_channels=self.h_dim * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(num_features=self.h_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 40
            nn.Conv1d(in_channels=self.h_dim * 4, out_channels=self.h_dim * 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(num_features=self.h_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 20
            nn.Conv1d(in_channels=self.h_dim * 8, out_channels=self.h_dim * 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(num_features=self.h_dim * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 10
            nn.Conv1d(in_channels=self.h_dim * 16, out_channels=z_dim, kernel_size=3, stride=1, padding=1, bias=False),
            # state size. (nz) x 1
        )
        self.dec = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose1d(in_channels=z_dim, out_channels=self.h_dim * 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(num_features=self.h_dim * 16),
            nn.ReLU(inplace=True),
            # state size. (ngf*16) x10
            nn.ConvTranspose1d(in_channels=h_dim * 16, out_channels=self.h_dim * 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(num_features=self.h_dim * 8),
            nn.ReLU(inplace=True),
            # state size. (ngf*8) x 20
            nn.ConvTranspose1d(in_channels=h_dim * 8, out_channels=self.h_dim * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(num_features=self.h_dim * 4),
            nn.ReLU(inplace=True),
            # state size. (ngf*2) x 40
            nn.ConvTranspose1d(in_channels=h_dim * 4, out_channels=self.h_dim * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(num_features=self.h_dim * 2),
            nn.ReLU(inplace=True),
            # state size. (ngf) x 80
            nn.ConvTranspose1d(in_channels=h_dim * 2, out_channels=self.h_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(num_features=self.h_dim),
            nn.ReLU(inplace=True),
            # state size. (ngf) x 160
            nn.ConvTranspose1d(in_channels=h_dim, out_channels=self.x_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
            # state size. (nc) x 320
        )

    def forward(self, input):
        input_reshape = input.permute(0, 2, 1)
        z = self.enc(input_reshape)
        output = self.dec(z)
        return output.permute(0, 2, 1)


class DiscriminatorCNN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, sequence_length):
        super(DiscriminatorCNN, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.sequence_length = sequence_length
        self.features = nn.Sequential(
            # input is (nc) x 64
            nn.Conv1d(in_channels=self.x_dim, out_channels=self.h_dim, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 160
            nn.Conv1d(in_channels=self.h_dim, out_channels=self.h_dim * 2, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm1d(num_features=self.h_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 80
            nn.Conv1d(in_channels=self.h_dim * 2, out_channels=self.h_dim * 4, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm1d(num_features=self.h_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 40
            nn.Conv1d(in_channels=self.h_dim * 4, out_channels=self.h_dim * 8, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm1d(num_features=self.h_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 20
            nn.Conv1d(in_channels=self.h_dim * 8, out_channels=self.h_dim * 16, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm1d(num_features=self.h_dim * 16),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.h_dim * 16 * self.sequence_length, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        input_reshape = input.permute(0, 2, 1)
        features = self.features(input_reshape)
        features_reshape = features.view(features.shape[0], -1)
        classifier = self.classifier(features_reshape)
        return classifier, features


class GeneratorFC(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, sequence_length):
        super(GeneratorFC, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.sequence_length = sequence_length
        self.enc = nn.Sequential(
            # input is (nc) x 64
            nn.Linear(in_features=self.x_dim, out_features=self.h_dim * 8),
            nn.Tanh(),
            nn.Linear(in_features=self.h_dim * 8, out_features=self.h_dim * 4),
            nn.Tanh(),
            nn.Linear(in_features=self.h_dim * 4, out_features=self.h_dim * 2),
            nn.Tanh(),
            nn.Linear(in_features=self.h_dim * 2, out_features=self.h_dim),
            nn.Tanh(),
            nn.Linear(in_features=self.h_dim, out_features=self.z_dim),
        )
        self.dec = nn.Sequential(
            # input is (nc) x 64
            nn.Linear(in_features=self.z_dim, out_features=self.h_dim),
            nn.Tanh(),
            nn.Linear(in_features=self.h_dim, out_features=self.h_dim * 2),
            nn.Tanh(),
            nn.Linear(in_features=self.h_dim * 2, out_features=self.h_dim * 4),
            nn.Tanh(),
            nn.Linear(in_features=self.h_dim * 4, out_features=self.h_dim * 8),
            nn.Tanh(),
            nn.Linear(in_features=self.h_dim * 8, out_features=self.x_dim),
            nn.Tanh(),
        )

    def forward(self, input):
        input_reshape = input.view(input.shape[0], -1)
        z = self.enc(input_reshape)
        output = self.dec(z)
        output = output.view(input.shape[0], input.shape[1], input.shape[2])
        return output


class DiscriminatorFC(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, sequence_length):
        super(DiscriminatorFC, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.sequence_length = sequence_length
        self.features = nn.Sequential(
            # input is (nc) x 64
            nn.Linear(in_features=self.x_dim, out_features=self.h_dim * 8),
            nn.Tanh(),
            nn.Linear(in_features=self.h_dim * 8, out_features=self.h_dim * 4),
            nn.Tanh(),
            nn.Linear(in_features=self.h_dim * 4, out_features=self.h_dim * 2),
            nn.Tanh(),
            nn.Linear(in_features=self.h_dim * 2, out_features=self.h_dim),
            nn.Tanh(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.h_dim, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        input = input.view(input.shape[0], -1)
        features = self.features(input)
        # features = self.feat(features.view(features.shape[0],-1))
        # features = features.view(out_features.shape[0],-1)
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)
        return classifier, features


class BeatGAN(nn.Module):
    def __init__(self, file_name, config):
        super(BeatGAN, self).__init__()
        # file info
        self.dataset = config.dataset
        self.file_name = file_name

        # model type
        self.model_type = config.model_type

        # dim info
        self.x_dim = config.x_dim
        self.h_dim = config.h_dim
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
        self.batch_size = config.batch_size
        self.weight_decay = config.weight_decay
        self.gamma = config.gamma
        self.early_stopping = config.early_stopping
        self.display_epoch = config.display_epoch
        self.lmbda = config.lmbda
        self.use_clip_norm = config.use_clip_norm
        self.gradient_clip_norm = config.gradient_clip_norm

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
                './save_model/{}/BEATGAN_hdim_{}_rollingsize_{}' \
                '_{}_pid={}.pt'.format(self.dataset, config.h_dim, config.rolling_size, Path(self.file_name).stem, self.pid)
        else:
            self.save_model_path = None

        self.load_model = config.load_model
        if self.load_model:
            self.load_model_path = \
                './save_model/{}/BEATGAN_hdim_{}_rollingsize_{}' \
                '_{}_pid={}.pt'.format(self.dataset, config.h_dim, config.rolling_size, Path(self.file_name).stem, self.pid)
        else:
            self.load_model_path = None

        assert (self.model_type == 1 or self.model_type == 2), 'Model type wrong!'
        if self.model_type == 1:
            self.G = GeneratorFC(x_dim=self.x_dim, h_dim=self.h_dim, z_dim=self.z_dim, sequence_length=self.rolling_size)
            self.D = DiscriminatorFC(x_dim=self.x_dim, h_dim=self.h_dim, z_dim=self.z_dim, sequence_length=self.rolling_size)
        elif self.model_type == 2:
            self.G = GeneratorCNN(x_dim=self.x_dim, h_dim=self.h_dim, z_dim=self.z_dim, sequence_length=self.rolling_size)
            self.D = DiscriminatorCNN(x_dim=self.x_dim, h_dim=self.h_dim, z_dim=self.z_dim, sequence_length=self.rolling_size)

    def forward_G(self, input):
        out_G = self.G(input)
        return out_G

    def forward_D(self, input):
        out_D_classifier, out_D_features = self.D(input)
        return out_D_classifier, out_D_features

    def fit(self, train_input, train_label, valid_input, valid_label, test_input, test_label, abnormal_data, abnormal_label, original_x_dim):
        opt_D = Adam(self.D.parameters(), lr=self.lr / 2, weight_decay=self.weight_decay)
        opt_G = Adam(self.G.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched_D = lr_scheduler.StepLR(optimizer=opt_D, step_size=self.milestone_epochs, gamma=self.gamma)
        sched_G = lr_scheduler.StepLR(optimizer=opt_G, step_size=self.milestone_epochs, gamma=self.gamma)
        loss_D = nn.BCELoss()
        loss_G = nn.MSELoss()
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
            self.train()
            epoch_D_losses = []
            epoch_G_losses = []
            epoch_valid_losses = []
            for epoch in range(self.epochs):
                train_D_losses = []
                train_G_losses = []
                opt_D.zero_grad()
                opt_G.zero_grad()
                for i, (batch_x, batch_y) in enumerate(train_data):
                    # opt_D.zero_grad()
                    batch_x = batch_x.to(device)
                    # Train D with real
                    out_D_real_classifier, _ = self.forward_D(batch_x)
                    # Train D with fake
                    fake = self.forward_G(batch_x)
                    out_d_fake_classifier, _ = self.forward_D(fake)
                    err_d_real = loss_D(out_D_real_classifier, torch.ones(batch_x.shape[0]).to(device))
                    err_d_fake = loss_D(out_d_fake_classifier, torch.zeros(batch_x.shape[0]).to(device))
                    err_d = err_d_real + err_d_fake
                    err_d.backward(retain_graph=True)
                    opt_D.step()
                    sched_D.step()
                    train_D_losses.append(err_d.item())
                    # Train G with real
                    # opt_G.zero_grad()
                    _, out_D_fake_features = self.forward_D(fake)
                    _, out_D_real_features = self.forward_D(batch_x)
                    err_g_adv = loss_G(out_D_fake_features, out_D_real_features)  # loss for feature matching
                    err_g_rec = loss_G(fake, batch_x)  # constrain x' to look like x
                    err_g = err_g_rec + self.lmbda * err_g_adv
                    err_g.backward()
                    opt_G.step()
                    sched_G.step()
                    train_G_losses.append(err_g.item())
                epoch_D_losses.append(mean(train_D_losses))
                epoch_G_losses.append(mean(train_G_losses))
                if epoch % self.display_epoch == 0:
                    train_logger.info('epoch = {} , train D loss = {}'.format(epoch, train_D_losses[-1]))
                    train_logger.info('epoch = {} , train G loss = {}'.format(epoch, train_G_losses[-1]))
                    # train_logger.info('epoch = {} , train rec loss = {}'.format(epoch, train_rec_losses[-1]))

                valid_losses = []
                with torch.no_grad():
                    self.eval()
                    for i, (val_batch_x, val_batch_y) in enumerate(valid_data):
                        val_batch_x = val_batch_x.to(device)
                        val_fake = self.forward_G(val_batch_x)
                        err_g_rec = loss_G(val_fake, val_batch_x)  # constrain x' to look like x
                        valid_losses.append(err_g_rec.item())
                    epoch_valid_losses.append(mean(valid_losses))
                    if epoch % self.display_epoch == 0:
                        train_logger.info('epoch = {} , train rec loss = {}'.format(epoch, epoch_valid_losses[-1]))

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
            epoch_D_losses = []
            epoch_G_losses = []
            epoch_rec_losses = []
            epoch_valid_losses = []
            for epoch in range(self.epochs):
                train_D_losses = []
                train_G_losses = []
                self.train()
                # Use the reconstruction loss as validation loss
                train_rec_losses = []
                opt_D.zero_grad()
                opt_G.zero_grad()
                for i, (batch_x, batch_y) in enumerate(train_data):
                    # opt_D.zero_grad()
                    batch_x = batch_x.to(device)
                    # Train D with real
                    out_D_real_classifier, _ = self.forward_D(batch_x)
                    # Train D with fake
                    fake = self.forward_G(batch_x)
                    out_d_fake_classifier, _ = self.forward_D(fake)
                    err_d_real = loss_D(out_D_real_classifier, torch.ones(batch_x.shape[0]).to(device))
                    err_d_fake = loss_D(out_d_fake_classifier, torch.zeros(batch_x.shape[0]).to(device))
                    err_d = err_d_real + err_d_fake
                    err_d.backward(retain_graph=True)
                    opt_D.step()
                    sched_D.step()
                    train_D_losses.append(err_d.item())
                    # Train G with real
                    # opt_G.zero_grad()
                    _, out_D_fake_features = self.forward_D(fake)
                    _, out_D_real_features = self.forward_D(batch_x)
                    err_g_adv = loss_G(out_D_fake_features, out_D_real_features)  # loss for feature matching
                    err_g_rec = loss_G(fake, batch_x)  # constrain x' to look like x
                    err_g = err_g_rec + self.lmbda * err_g_adv
                    err_g.backward()
                    opt_G.step()
                    sched_G.step()
                    train_G_losses.append(err_g.item())
                    train_rec_losses.append(err_g_rec.item())
                epoch_D_losses.append(mean(train_D_losses))
                epoch_G_losses.append(mean(train_G_losses))
                epoch_rec_losses.append(mean(train_rec_losses))
                if epoch % self.display_epoch == 0:
                    train_logger.info('epoch = {} , train D loss = {}'.format(epoch, train_D_losses[-1]))
                    train_logger.info('epoch = {} , train G loss = {}'.format(epoch, train_G_losses[-1]))
                    train_logger.info('epoch = {} , train rec loss = {}'.format(epoch, train_rec_losses[-1]))

                valid_losses = []
                with torch.no_grad():
                    self.eval()
                    for i, (val_batch_x, val_batch_y) in enumerate(valid_data):
                        val_batch_x = val_batch_x.to(device)
                        val_fake = self.forward_G(val_batch_x)
                        err_g_rec = loss_G(val_fake, val_batch_x)  # constrain x' to look like x
                        valid_losses.append(err_g_rec.item())
                    epoch_valid_losses.append(mean(valid_losses))
                    if epoch % self.display_epoch == 0:
                        train_logger.info('epoch = {} , train rec loss = {}'.format(epoch, epoch_valid_losses[-1]))

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
            cat_fakes = []
            for i, (batch_x, batch_y) in enumerate(test_data):
                batch_x = batch_x.to(device)
                fake = self.forward_G(batch_x)
                cat_fakes.append(fake)
        cat_fakes = torch.cat(cat_fakes)
        beat_gan_output = BEATGANOutput(best_TN=None, best_FP=None, best_FN=None, best_TP=None,
                                        best_precision=None, best_recall=None, best_fbeta=None,
                                        best_pr_auc=None, best_roc_auc=None, best_cks=None,
                                        dec_means=cat_fakes, min_valid_loss=min_valid_loss)

        return beat_gan_output


def RunModel(train_filename, test_filename, label_filename, config):

    train_data, abnormal_data, abnormal_label = read_dataset(train_filename, test_filename, label_filename,
                                                             normalize=True, file_logger=file_logger)

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

    if config.model_type == 1:
        config.x_dim = rolling_abnormal_data.shape[1] * rolling_abnormal_data.shape[2]
    elif config.model_type == 2:
        config.x_dim = rolling_abnormal_data.shape[2]

    model = BeatGAN(file_name=train_filename, config=config)
    model = model.to(device)
    beat_gan_output = None
    if train_data is not None and config.robustness == False:
        beat_gan_output = model.fit(train_input=rolling_train_data, train_label=rolling_train_data,
                                    valid_input=rolling_valid_data, valid_label=rolling_valid_data,
                                    test_input=rolling_abnormal_data, test_label=rolling_abnormal_label,
                                    abnormal_data=abnormal_data, abnormal_label=abnormal_label,
                                    original_x_dim=original_x_dim)
    elif train_data is None or config.robustness == True:
        beat_gan_output = model.fit(train_input=rolling_abnormal_data, train_label=rolling_abnormal_data,
                                    valid_input=rolling_valid_data, valid_label=rolling_valid_data,
                                    test_input=rolling_abnormal_data, test_label=rolling_abnormal_label,
                                    abnormal_data=abnormal_data, abnormal_label=abnormal_label,
                                    original_x_dim=original_x_dim)
    # %%
    # %%
    min_max_scaler = preprocessing.MinMaxScaler()
    if config.preprocessing:
        if config.use_overlapping:
            if config.use_last_point:
                dec_mean_unroll = beat_gan_output.dec_means.detach().cpu().numpy()[:, -1]
                dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
                x_original_unroll = abnormal_data[config.rolling_size - 1:]
            else:
                # the unroll_window_3D will recover the shape as abnormal_data
                # and we only use the [config.rolling_size-1:] to calculate the error, in which we ignore
                # the first config.rolling_size time steps.
                dec_mean_unroll = unroll_window_3D(beat_gan_output.dec_means.detach().cpu().numpy())[::-1]
                dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
                x_original_unroll = abnormal_data[: dec_mean_unroll.shape[0]]

        else:
            dec_mean_unroll = np.reshape(beat_gan_output.dec_means.detach().cpu().numpy(), (-1, original_x_dim))
            dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
            x_original_unroll = abnormal_data[: dec_mean_unroll.shape[0]]
    else:
        dec_mean_unroll = beat_gan_output.dec_means.detach().cpu().numpy()
        dec_mean_unroll = np.squeeze(dec_mean_unroll, axis=0)
        dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
        x_original_unroll = abnormal_data

    if config.save_output:
        if not os.path.exists('./outputs/NPY/{}/'.format(config.dataset)):
            os.makedirs('./outputs/NPY/{}/'.format(config.dataset))
        np.save('./outputs/NPY/{}/Dec_BEATGAN_hdim_{}_rollingsize_{}_{}_pid={}.npy'.format(config.dataset, config.h_dim, config.rolling_size, train_filename.stem, config.pid), dec_mean_unroll)

    error = np.sum(x_original_unroll - np.reshape(dec_mean_unroll, [-1, original_x_dim]), axis=1) ** 2
    #final_zscore = zscore(error)
    #np_decision = create_label_based_on_zscore(final_zscore, 2.5, True)
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
            plt.savefig('./figures/{}/Ori_BEATGAN_hdim_{}_rollingsize_{}_{}_pid={}.png'.format(config.dataset, config.h_dim, config.rolling_size, train_filename.stem, config.pid), dpi=600)
            plt.close()

            # Plot decoder output
            plt.figure(figsize=(9, 3))
            plt.plot(dec_mean_unroll, color='blue', lw=1.5)
            plt.title('Decoding Output')
            plt.grid(True)
            plt.tight_layout()
            # plt.show()
            plt.savefig('./figures/{}/Dec_BEATGAN_hdim_{}_rollingsize_{}_{}_pid={}.png'.format(config.dataset, config.h_dim, config.rolling_size, train_filename.stem, config.pid), dpi=600)
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
            plt.savefig('./figures/{}/VisInp_BEATGAN_hdim_{}_rollingsize_{}_SD_{}_pid={}.png'.format(config.dataset, config.h_dim, config.rolling_size, train_filename.stem, config.pid), dpi=600)
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
            plt.savefig('./figures/{}/VisOut_BEATGAN_hdim_{}_rollingsize_{}_SD_{}_pid={}.png'.format(config.dataset, config.h_dim, config.rolling_size, train_filename.stem, config.pid), dpi=600)
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
            plt.savefig('./figures/{}/VisOut_BEATGAN_hdim_{}_rollingsize_{}_MAD_{}_pid={}.png'.format(config.dataset, config.h_dim, config.rolling_size, train_filename.stem, config.pid), dpi=600)
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
            plt.savefig('./figures/{}/VisOut_BEATGAN_hdim_{}_rollingsize_{}_IQR_{}_pid={}.png'.format(config.dataset, config.h_dim, config.rolling_size, train_filename.stem, config.pid), dpi=600)
            plt.close()
        else:
            file_logger.info('cannot plot image with x_dim > 1')

    if config.use_spot:
        pass
    else:
        pos_label = -1
        TN, FP, FN, TP, precision, recall, f1 = {}, {}, {}, {}, {}, {}, {}
        for threshold_method in np_decision:
            cm = confusion_matrix(y_true=abnormal_label,
                                  y_pred=np_decision[threshold_method], labels=[1, -1])
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
        fpr, tpr, _ = roc_curve(y_true=abnormal_label, y_score=np.nan_to_num(error), pos_label=pos_label)
        roc_auc = auc(fpr, tpr)
        pre, re, _ = precision_recall_curve(y_true=abnormal_label, probas_pred=np.nan_to_num(error),
                                            pos_label=pos_label)
        pr_auc = auc(re, pre)
        metrics_result = MetricsResult(TN=TN, FP=FP, FN=FN, TP=TP, precision=precision,
                                       recall=recall, fbeta=f1, pr_auc=pr_auc, roc_auc=roc_auc,
                                       best_TN=beat_gan_output.best_TN, best_FP=beat_gan_output.best_FP,
                                       best_FN=beat_gan_output.best_FN, best_TP=beat_gan_output.best_TP,
                                       best_precision=beat_gan_output.best_precision, best_recall=beat_gan_output.best_recall,
                                       best_fbeta=beat_gan_output.best_fbeta, best_pr_auc=beat_gan_output.best_pr_auc,
                                       best_roc_auc=beat_gan_output.best_roc_auc, best_cks=beat_gan_output.best_cks,
                                       min_valid_loss=beat_gan_output.min_valid_loss)
        return metrics_result

if __name__ == '__main__':

    # %%
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, default=0)
    parser.add_argument('--model_type', type=int, default=2)
    parser.add_argument('--x_dim', type=int, default=1)
    parser.add_argument('--h_dim', type=int, default=64)
    parser.add_argument('--z_dim', type=int, default=16)
    parser.add_argument('--preprocessing', type=str2bool, default=True)
    parser.add_argument('--use_overlapping', type=str2bool, default=True)
    parser.add_argument('--rolling_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--milestone_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--weight_decay', type=float, default=1e-8)
    parser.add_argument('--early_stopping', type=str2bool, default=True)
    parser.add_argument('--lmbda', type=float, default=1)
    parser.add_argument('--use_clip_norm', type=str2bool, default=True)
    parser.add_argument('--gradient_clip_norm', type=float, default=10)
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

        # the dim in args is useless, which should be deleted in the future version.
        args.dataset = registered_dataset

        if args.load_config:
            config = BEATGANConfig(dataset=None, model_type=None, x_dim=None, h_dim=None, z_dim=None, preprocessing=None,
                                   use_overlapping=None, rolling_size=None, epochs=None, milestone_epochs=None, lr=None,
                                   gamma=None, batch_size=None, weight_decay=None, early_stopping=None, lmbda=None,
                                   use_clip_norm=None, gradient_clip_norm=None, display_epoch=None, save_output=None,
                                   save_figure=None, save_model=None, load_model=None, continue_training=None, dropout=None,
                                   use_spot=None, use_last_point=None, save_config=None, load_config=None, server_run=None,
                                   robustness=None, pid=None, save_results=None)
            try:
                config.import_config('./config/{}/Config_BEATGAN_hdim_{}_rollingsize_{}_pid={}.json'.format(config.dataset, config.h_dim, config.rolling_size, config.pid))
            except:
                print('There is no config.')
        else:
            config = BEATGANConfig(dataset=args.dataset, model_type=args.model_type, x_dim=args.x_dim, h_dim=args.h_dim,
                                   z_dim=args.z_dim, preprocessing=args.preprocessing, use_overlapping=args.use_overlapping,
                                   rolling_size=args.rolling_size, epochs=args.epochs,
                                   milestone_epochs=args.milestone_epochs, lr=args.lr, gamma=args.gamma,
                                   batch_size=args.batch_size, weight_decay=args.weight_decay,
                                   early_stopping=args.early_stopping, lmbda=args.lmbda, use_clip_norm=args.use_clip_norm,
                                   gradient_clip_norm=args.gradient_clip_norm, display_epoch=args.display_epoch,
                                   save_output=args.save_output, save_figure=args.save_figure, save_model=args.save_model,
                                   load_model=args.load_model, continue_training=args.continue_training,
                                   dropout=args.dropout, use_spot=args.use_spot, use_last_point=args.use_last_point,
                                   save_config=args.save_config, load_config=args.load_config, server_run=args.server_run,
                                   robustness=args.robustness, pid=args.pid, save_results=args.save_results)
        if args.save_config:
            if not os.path.exists('./config/{}/'.format(config.dataset)):
                os.makedirs('./config/{}/'.format(config.dataset))
            config.export_config('./config/{}/Config_BEATGAN_hdim_{}_rollingsize_{}_pid={}.json'.format(config.dataset, config.h_dim, config.rolling_size, config.pid))
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
                                                               train_logger_name='beat_gan_train_logger',
                                                               file_logger_name='beat_gan_file_logger',
                                                               meta_logger_name='beat_gan_meta_logger',
                                                               model_name='BEATGAN',
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
                                      config=config)
            result_dataframe = make_result_dataframe(metrics_result)

            if config.save_results == True:
                if not os.path.exists('./results/{}/'.format(config.dataset)):
                    os.makedirs('./results/{}/'.format(config.dataset))
                result_dataframe.to_csv(
                    './results/{}/Results_BEATGAN_hdim_{}_rollingsize_{}_{}_pid={}.csv'.format(config.dataset, config.h_dim, config.rolling_size, train_file.stem, config.pid),
                    index=False)
