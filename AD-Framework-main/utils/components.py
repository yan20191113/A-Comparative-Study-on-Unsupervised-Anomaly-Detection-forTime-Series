import logging
from statistics import mean, median
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from utils.data_provider import get_loader
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class FullyConnectedLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=False):
        super(FullyConnectedLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc1 = nn.Linear(self.in_features, self.out_features)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        return x


class ConvolutionalLayer1D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(ConvolutionalLayer1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1d_1 = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=5, padding=2, stride=1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1d_1(x))
        return x


class RobustPCA(nn.Module):
    def __init__(self, mu=None, lmbda=None, max_iter=100):
        super(RobustPCA, self).__init__()
        if mu:
            self.mu = mu
        else:
            self.mu = None
        if lmbda:
            self.lmbda = lmbda
        else:
            self.lmbda = None
        if max_iter:
            self.max_iter = max_iter

    def frobenius_norm(self, M):
        return torch.norm(M, p='fro')

    def shrink(self, M, tau):
        return torch.sign(M) * torch.max((torch.abs(M) - tau), torch.zeros(M.shape).to(device))

    def svd_threshold(self, M, tau):
        U, S, V = torch.svd(M)
        return torch.mm(U, torch.mm(torch.diag(self.shrink(S, tau)), torch.transpose(V, dim0=1, dim1=0)))

    def forward(self, X, train_logger):
        S = torch.zeros(X.shape).to(device)
        Y = torch.zeros(X.shape).to(device)
        if self.mu is None:
            self.mu = np.prod(X.shape) / (4 * self.frobenius_norm(X))
        self.mu_inv = 1 / self.mu
        if self.lmbda is None:
            self.lmbda = 1 / np.sqrt(np.max(X.shape))
        _tol = 1E-8 * self.frobenius_norm(X)
        epoch = 0
        epoch_losses = np.Inf
        Sk = S
        Yk = Y
        Lk = torch.zeros(X.shape).to(device)

        while (epoch_losses > _tol) and epoch < self.max_iter:
            Lk = self.svd_threshold(X - Sk + self.mu_inv * Yk, self.mu_inv)
            Sk = self.shrink(X - Lk + (self.mu_inv * Yk), self.mu_inv * self.lmbda)
            Yk = Yk + self.mu * (X - Lk - Sk)
            epoch_losses = self.frobenius_norm(X - Lk - Sk)
            epoch += 1
            if (epoch % 10) == 0 or epoch == 1 or epoch > self.max_iter or epoch_losses <= _tol:
                train_logger.info('RPCA epoch={}, RPCA loss={}'.format(epoch, epoch_losses))
        return Lk, Sk


class AutoEncoder(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=False, display_epoch=10,
                 epochs=500, batch_size=128, lr=0.001):
        super(AutoEncoder, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.dropout = dropout
        self.fc_ae = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=self.hidden_features * 4, bias=True),
            nn.PReLU(),
            nn.Linear(in_features=self.hidden_features * 4, out_features=self.hidden_features * 2, bias=True),
            nn.PReLU(),
            nn.Linear(in_features=self.hidden_features * 2, out_features=self.hidden_features, bias=True),
            nn.PReLU(),
            nn.Linear(in_features=self.hidden_features, out_features=self.hidden_features * 2, bias=True),
            nn.PReLU(),
            nn.Linear(in_features=self.hidden_features * 2, out_features=self.hidden_features * 4, bias=True),
            nn.PReLU(),
            nn.Linear(in_features=self.hidden_features * 4, out_features=self.out_features, bias=True),
            # nn.Sigmoid(),
        )

        self.display_epoch = display_epoch
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    def weights_init(self):
        for p in self.fc_ae.parameters():
            torch.nn.init.xavier_uniform_(p)

    def forward(self, input):
        output = self.fc_ae(input)
        return output, 0

    def fit(self, X, Y, train_logger):
        # init optimizer
        opt = Adam(self.parameters(), lr=self.lr, weight_decay=1e-8)
        sched = lr_scheduler.StepLR(optimizer=opt, step_size=50, gamma=0.95)
        loss = nn.MSELoss()

        # get batch data
        train_data = get_loader(X, Y, batch_size=self.batch_size)
        epoch_losses = []

        # train model
        for epoch in range(self.epochs):
            self.train()
            train_losses = []
            for i, (batch_X, batch_Y) in enumerate(train_data):
                opt.zero_grad()
                recontructed_batch_X, _ = self.forward(batch_X)
                batch_loss = loss(batch_X, recontructed_batch_X)
                batch_loss.backward()
                opt.step()
                sched.step()
                train_losses.append(batch_loss.item())
            epoch_losses.append(mean(train_losses))
            if epoch % self.display_epoch == 0:
                train_logger.info('inner AE epoch = {}, inner AE loss = {}'.format(epoch, epoch_losses[-1]))
            # if epoch > 1:
            #     if -1e-8 < epoch_losses[-1] - epoch_losses[-2] < 1e-8:
            #         train_logger.info('early break')
            #         break
        # test model
        with torch.no_grad():
            self.eval()
            reconstructed_X = []
            for i, (batch_X, batch_Y) in enumerate(train_data):
                recontructed_batch_X, _ = self.forward(batch_X)
                reconstructed_X.append(recontructed_batch_X)
            reconstructed_X = torch.stack(reconstructed_X, dim=0)
            return torch.squeeze(reconstructed_X, dim=0), mean(epoch_losses)


class Conv1DAutoEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, epochs, lr, batch_size, dropout=False, display_epoch=10):
        super(Conv1DAutoEncoder, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.conv1d_ae =nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, out_channels=self.hidden_channels * 8, kernel_size=5, stride=1, padding=2, bias=True),
            nn.PReLU(),
            nn.Conv1d(in_channels=self.hidden_channels * 8, out_channels=self.hidden_channels * 4, kernel_size=5, stride=1, padding=2, bias=True),
            nn.PReLU(),
            nn.Conv1d(in_channels=self.hidden_channels * 4, out_channels=self.hidden_channels * 2, kernel_size=5, stride=1, padding=2, bias=True),
            nn.PReLU(),
            nn.Conv1d(in_channels=self.hidden_channels * 2, out_channels=self.hidden_channels, kernel_size=5, stride=1, padding=2, bias=True),
            nn.PReLU(),
            nn.ConvTranspose1d(in_channels=self.hidden_channels, out_channels=self.hidden_channels * 2, kernel_size=5, stride=1, padding=2, bias=True),
            nn.PReLU(),
            nn.ConvTranspose1d(in_channels=self.hidden_channels * 2, out_channels=self.hidden_channels * 4, kernel_size=5, stride=1, padding=2, bias=True),
            nn.PReLU(),
            nn.ConvTranspose1d(in_channels=self.hidden_channels * 4, out_channels=self.hidden_channels * 8, kernel_size=5, stride=1, padding=2, bias=True),
            nn.PReLU(),
            nn.ConvTranspose1d(in_channels=self.hidden_channels * 8, out_channels=self.in_channels, kernel_size=5, stride=1, padding=2, bias=True),
            # nn.Sigmoid(),
        )
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.display_epoch = display_epoch

    def forward(self, input):
        output = self.conv1d_ae(input)
        return output, 0

    def weights_init(self):
        for p in self.conv1d_ae.parameters():
            torch.nn.init.xavier_uniform_(p)

    def fit(self, X, Y, train_logger):
        # init optimizer
        opt = Adam(self.parameters(), lr=self.lr, weight_decay=1e-8)
        sched = lr_scheduler.StepLR(optimizer=opt, step_size=50, gamma=0.95)
        loss = nn.MSELoss()
        # get batch data
        train_data = get_loader(X, Y, batch_size=self.batch_size)
        epoch_losses = []
        # train model
        self.train()
        for epoch in range(self.epochs):
            train_losses = []
            for i, (batch_X, batch_Y) in enumerate(train_data):
                opt.zero_grad()
                recontructed_batch_X, _ = self.forward(batch_X)
                batch_loss = loss(batch_X, recontructed_batch_X)
                batch_loss.backward()
                opt.step()
                sched.step()
                train_losses.append(batch_loss.item())
            epoch_losses.append(mean(train_losses))
            if epoch % self.display_epoch == 0:
                train_logger.info('outter AE epoch = {} , outter AE loss = {}'.format(epoch, epoch_losses[-1]))
            # if epoch > 1:
            #     if -1e-8 < epoch_losses[-1] - epoch_losses[-2] < 1e-8:
            #         train_logger.info('early break')
            #         break
        # test model
        with torch.no_grad():
            self.eval()
            reconstructed_X = []
            for i, (batch_X, batch_Y) in enumerate(train_data):
                recontructed_batch_X, _ = self.forward(batch_X)
                reconstructed_X.append(recontructed_batch_X)
            reconstructed_X = torch.stack(reconstructed_X, dim=0)
            return torch.squeeze(reconstructed_X, dim=0), mean(epoch_losses)


class Conv2DAutoEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, epochs, lr, batch_size, dropout=False, display_epoch=10):
        super(Conv2DAutoEncoder, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.conv2d_ae = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=256, kernel_size=5, stride=1, padding=2, bias=True),
            nn.PReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=5, stride=1, padding=2, bias=True),
            nn.PReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=1, padding=2, bias=True),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2, bias=True),
            nn.PReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2, bias=True),
            nn.PReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2, bias=True),
            nn.PReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2, bias=True),
            nn.PReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=self.in_channels, kernel_size=5, stride=1, padding=2, bias=True),
        )
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.display_epoch = display_epoch

    def forward(self, input):
        output = self.conv2d_ae(input)
        return output, 0

    def weights_init(self):
        for p in self.conv2d_ae.parameters():
            torch.nn.init.xavier_uniform_(p)

    def fit(self, X, Y, train_logger):
        # init optimizer
        opt = Adam(self.parameters(), lr=self.lr, weight_decay=1e-8)
        sched = lr_scheduler.StepLR(optimizer=opt, step_size=100, gamma=0.95)
        loss = nn.MSELoss()
        # get batch data
        train_data = get_loader(X, Y, batch_size=self.batch_size)
        epoch_losses = []
        # train model
        self.train()
        for epoch in range(self.epochs):
            train_losses = []
            for i, (batch_X, batch_Y) in enumerate(train_data):
                opt.zero_grad()
                recontructed_batch_X, _ = self.forward(batch_X)
                batch_loss = loss(batch_X, recontructed_batch_X)
                batch_loss.backward()
                opt.step()
                sched.step()
                train_losses.append(batch_loss.item())
            epoch_losses.append(mean(train_losses))
            if epoch % self.display_epoch == 0:
                train_logger.info('inner AE epoch = {} , inner AE loss = {}'.format(epoch, epoch_losses[-1]))
            # if epoch > 1:
            #     if -1e-8 < epoch_losses[-1] - epoch_losses[-2] < 1e-8:
            #         train_logger.info('early break')
            #         break
        # test model
        with torch.no_grad():
            self.eval()
            reconstructed_X = []
            for i, (batch_X, batch_Y) in enumerate(train_data):
                recontructed_batch_X, _ = self.forward(batch_X)
                reconstructed_X.append(recontructed_batch_X)
            reconstructed_X = torch.stack(reconstructed_X, dim=0)
            return torch.squeeze(reconstructed_X, dim=0), mean(epoch_losses)


class EncodeTimeSeriesToSpectralMatrix(nn.Module):
    def __init__(self, in_features, out_features, epochs, lr, batch_size, display_epoch=10):
        super(EncodeTimeSeriesToSpectralMatrix, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=(self.in_features + self.out_features) * 16),
            nn.PReLU(),
            nn.Linear(in_features=(self.in_features + self.out_features) * 16, out_features=(self.in_features + self.out_features) * 8),
            nn.PReLU(),
            nn.Linear(in_features=(self.in_features + self.out_features) * 8, out_features=(self.in_features + self.out_features) * 4),
            nn.PReLU(),
            nn.Linear(in_features=(self.in_features + self.out_features) * 4, out_features=(self.in_features + self.out_features) * 8),
            nn.PReLU(),
            nn.Linear(in_features=(self.in_features + self.out_features) * 8, out_features=(self.in_features + self.out_features) * 16),
            nn.PReLU(),
            nn.Linear(in_features=(self.in_features + self.out_features) * 16, out_features=self.out_features),
            # nn.Sigmoid(),
        )
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.display_epoch = display_epoch

    def weights_init(self):
        for p in self.encoder.parameters():
            torch.nn.init.xavier_uniform_(p)

    def forward(self, input):
        output = self.encoder(input)
        return output, 0

    def fit(self, X, train_logger):
        # init optimizer
        opt = Adam(self.parameters(), lr=self.lr, weight_decay=1e-8)
        sched = lr_scheduler.StepLR(optimizer=opt, step_size=100, gamma=0.95)
        loss = nn.MSELoss()

        # get batch data
        train_data = get_loader(X, None, batch_size=self.batch_size)
        epoch_losses = []

        # train model
        for epoch in range(self.epochs):
            self.train()
            train_losses = []
            for i, (batch_X, batch_Y) in enumerate(train_data):
                opt.zero_grad()
                recontructed_batch_X, _ = self.forward(batch_X)
                batch_loss = loss(batch_X, recontructed_batch_X)
                batch_loss.backward()
                opt.step()
                sched.step()
                train_losses.append(batch_loss.item())
            epoch_losses.append(mean(train_losses))
            if epoch % self.display_epoch == 0:
                train_logger.info('T encode epoch={}, T loss={}'.format(epoch, epoch_losses[-1]))
            # if epoch > 1:
            #     if -1e-8 < epoch_losses[-1] - epoch_losses[-2] < 1e-8:
            #         train_logger.info('early break')
            #         break
        # test model
        with torch.no_grad():
            reconstructed_X = []
            for i, (batch_X, batch_Y) in enumerate(train_data):
                recontructed_batch_X, _ = self.forward(batch_X)
                reconstructed_X.append(recontructed_batch_X)
            reconstructed_X = torch.stack(reconstructed_X, dim=0)
            return torch.squeeze(reconstructed_X, dim=0), mean(epoch_losses)


class DecodeSpectralMatrixToTimeSeries(nn.Module):
    def __init__(self, in_features, out_features, epochs, lr, batch_size, display_epoch=10):
        '''
        Time series will be here with shape [batch, channel, observation]
        :param in_features: in feature
        :param out_features: out feature
        :param display_epoch: display epoch
        '''
        super(DecodeSpectralMatrixToTimeSeries, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=(self.in_features + self.out_features) * 16),
            nn.PReLU(),
            nn.Linear(in_features=(self.in_features + self.out_features) * 16, out_features=(self.in_features + self.out_features) * 8),
            nn.PReLU(),
            nn.Linear(in_features=(self.in_features + self.out_features) * 8, out_features=(self.in_features + self.out_features) * 4),
            nn.PReLU(),
            nn.Linear(in_features=(self.in_features + self.out_features) * 4, out_features=(self.in_features + self.out_features) * 8),
            nn.PReLU(),
            nn.Linear(in_features=(self.in_features + self.out_features) * 8, out_features=(self.in_features + self.out_features) * 16),
            nn.PReLU(),
            nn.Linear(in_features=(self.in_features + self.out_features) * 16, out_features=self.out_features),
            # nn.Sigmoid(),
        )
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.display_epoch = display_epoch

    def weights_init(self):
        for p in self.decoder.parameters():
            torch.nn.init.xavier_uniform_(p)

    def forward(self, input):
        output = self.decoder(input)
        return output, 0

    def fit(self, T_L, train_logger):
        T_L = torch.transpose(T_L, dim0=2, dim1=1)
        # init optimizer
        opt = Adam(self.parameters(), lr=self.lr, weight_decay=1e-8)
        sched = lr_scheduler.StepLR(optimizer=opt, step_size=100, gamma=0.95)
        loss = nn.MSELoss()
        epoch_losses = []

        # train model
        for epoch in range(self.epochs):
            self.train()
            train_losses = []
            opt.zero_grad()
            reconstruc_T_L, _ = self.forward(T_L)
            batch_loss = loss(T_L, reconstruc_T_L)
            batch_loss.backward()
            opt.step()
            sched.step()
            train_losses.append(batch_loss.item())
            epoch_losses.append(mean(train_losses))
            if epoch % self.display_epoch == 0:
                train_logger.info('T decode epoch={}, T loss={}'.format(epoch, epoch_losses[-1]))
            # if epoch > 1:
            #     if -1e-8 < epoch_losses[-1] - epoch_losses[-2] < 1e-8:
            #         train_logger.info('early break')
            #         break
        # test model
        with torch.no_grad():
            self.eval()
            T_L, _ = self.forward(T_L)
            T_L = torch.transpose(T_L, dim0=2, dim1=1)
            return T_L, mean(epoch_losses)

