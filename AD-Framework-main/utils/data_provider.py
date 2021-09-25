import json
import os
import random
from math import pi
from pathlib import Path
import ast
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import pickle
import copy

possible_ratio = [0.01, 0.02, 0.03, 0.04, 0.05]
possible_noise_dataset = ["MSL", "NAB"]
datasets_path = Path("../datasets/")
dataset2path = {
    "MSL": {},
    "SMAP": {},
    "SMD": {},
    "NAB": {},
    "AIOps": {},
    "Credit": {},
    "ECG": {},
    "nyc_taxi": {},
    "SWAT": {},
    "Yahoo": {}
}
for dataset in dataset2path:
    dataset2path[dataset]["train"] = datasets_path / "train" / dataset
    dataset2path[dataset]["test"] = datasets_path / "test" / dataset
    dataset2path[dataset]["test_label"] = datasets_path / "test_label" / dataset

for ratio in possible_ratio:
    for noise_dataset in possible_noise_dataset:
        dataset2path[noise_dataset+"_noise_{:.2f}".format(ratio)] = dataset2path[noise_dataset]


def negative_sampling(data, index, length, delta=0.05):

    data1 = copy.deepcopy(data)
    low_val = min(data)
    high_val = max(data)
    flag = np.random.randint(0, 2)

    if flag == 0:  #
        a = np.random.uniform(
            low=low_val - delta,
            high=low_val,
            size=length)
    else:
        a = np.random.uniform(
            low=high_val,
            high=high_val + delta,
            size=length)

    for j in range(len(index)):
        data1[index[j]] = a[j]
    return data1

def read_dataset(train_filename, test_filename, label_filename, normalize=True, file_logger=None, ratio=0.01, negative_sample=False):
    '''
        file_name: the path of the file, such as "./datasets/train/AIOps/phase2_train.pkl"
        normalize: whether to normalize the data.
        ratio: ratio of negative samples
    Returns:

    '''
    train_data = pickle.load(Path(train_filename).open("rb"))
    test_data = pickle.load(Path(test_filename).open("rb"))
    # On the MSL dataset, the type of the dumped labels are bool, which should be converted to int.
    # On the SWaT dataset, the type of the dumped labels are "<U23", which should be converted to float type before
    # converted to int.
    labels = pickle.load(Path(label_filename).open("rb")).astype(dtype="float").astype(dtype="int")

    # label verify
    if labels.sum() == 0:

        if file_logger is not None:
            file_logger.warning("There are no anomalies in {}.".format(test_filename))
        # replace the test_label by all_label
        label_filename_parts = list(label_filename.parts)
        label_filename_parts[label_filename_parts.index("test_label")] = "all_label"
        label_filename = Path(label_filename_parts[0])
        for part in label_filename_parts[1:]:
            label_filename = label_filename / part
        labels = pickle.load(Path(label_filename).open("rb")).astype(dtype="float").astype(dtype="int")
        # concat the train_data and test_data
        train_data = np.concatenate([train_data, test_data], axis=0)
        test_data = copy.deepcopy(train_data)


    # On NAB dataset, the data is 1D array, which should be converted to 2D array.
    if len(train_data.shape) == 1:
        train_data = train_data.reshape(-1, 1)
        test_data = test_data.reshape(-1, 1)
    # For all the dataset, the labels should be 2D array.
    labels = labels.reshape(-1, 1)

    if normalize:
        scaler = MinMaxScaler(feature_range=(0, 1))
        #scaler.fit(np.concatenate([train_data, test_data], axis=0))
        scaler.fit(train_data)
        scaler.fit(test_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)

    assert test_data.shape[0] == labels.shape[0]
    # the normal data is marked as 1, and the abnormal data is marked as -1.
    labels[labels == 1] = -1
    labels[labels == 0] = 1


    paths = train_filename.parts
    dataset_name = paths[-2]

    if negative_sample == True and (dataset_name == "NAB" or dataset_name == "MSL"):
        file_logger.info('negative sampling')

        file_logger.info("ratio of negative samples: {:.4f}".format(ratio))
        (train_len, dim) = train_data.shape

        (test_len, _) = test_data.shape

        train_length = int(train_len * ratio)
        test_length = int(test_len * ratio)

        #make sure there is at least one negative sample
        if train_length < 1:
            train_length = 1
        if test_length < 1:
            test_length = 1

        train_index = np.random.randint(0, train_len, train_length)
        test_index = np.random.randint(0, test_len, test_length)

        for i in range(dim):
           train_data[:, i] = negative_sampling(train_data[:, i], train_index, train_length, delta=0.05)
           test_data[:, i] = negative_sampling(test_data[:, i], test_index, test_length, delta=0.05)


    return train_data.astype(dtype='float32'), test_data.astype(dtype='float32'), labels

def rolling_window_2D(a, n):
    # a: 2D Input array
    # n: Group/sliding window length
    return a[np.arange(a.shape[0] - n + 1)[:, None] + np.arange(n)]


def cutting_window_2D(a, n):
    # a: 2D Input array
    # n: Group/sliding window length
    split_positions = list(range(n, a.shape[0], n))
    split_result = np.array_split(a, split_positions)
    np_result = []
    if split_result[-1].shape[0] == split_result[-2].shape[0]:
        for array in split_result[:-1]:
            np_result.append(array)
    else:
        for array in split_result[:-1]:
            np_result.append(array)
    return np.stack(np_result)


def unroll_window_2D(a):
    '''
    :param a: 2D data, matrix of probability scores of rolling windows
    :return: 1D data, final probability score for points
    '''
    return np.array([a.diagonal(i).mean() for i in range(-a.shape[0] + 1, a.shape[1])])

def unroll_window_3D(a):
    '''
    :param a: 3D data, matrix of probability scores of rolling windows (total_length, rolling_size, features)
    :return: 1D data, final probability score for points
    '''
    multi_ts = []
    for channel in range(a.shape[2]):
        # Calculating the mean of the feature in different rolling windows
        uni_ts = np.array([a[:, :, channel].diagonal(i).mean() for i in range(-a[:, :, channel].shape[0] + 1, a[:, :, channel].shape[1])])
        multi_ts.append(uni_ts)

    multi_ts = np.stack(multi_ts, axis=1)
    return multi_ts


def generate_synthetic_dataset(case=0, N=200, noise=True, verbose=True):
    random.seed(a=2412)
    if case == 0:
        scaler = MinMaxScaler()
        t = np.arange(0, N)
        trend = 8 * np.sin(0.2 * (t - N))
        if verbose:
            plt.figure(figsize=(9, 3))
            ax = plt.axes()

            ax.plot(t, scaler.fit_transform(np.expand_dims(trend, 1)), 'blue', lw=2)

            plt.yticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_xlim(-10, 260)
            ax.set_ylim(-0.10, 1.10)

            plt.xlabel('$t$')
            plt.ylabel('$s$')
            plt.grid(True)
            plt.tight_layout()

            plt.show()
            # plt.savefig('./figures/0/T_0.png')
            # plt.close()
        if noise:
            noise = 3 * (np.random.rand(N) - 0.5)
        output = trend + noise
        injection_1 = [18, 24, 27, 62, 80, 83, 143, 173, 181, 205]
        # injection_2 = [40, 65, 108, 127, 135, 196, 234]
        # injection_3 = [17, 18, 23, 24, 26, 27, 39, 40, 61, 62, 64, 65, 79, 80, 82, 83, 107, 108, 126, 127, 134, 135, 142, 143, 172, 173,
        #                180, 181, 195, 196, 204, 205, 233, 234]
        # for index in injection_1:
        #     # if random.randint(0, 1) == 0:
        #     #     output[index] = random.randint(5, 8)
        #     # else:
        #     #     output[index] = random.randint(-8, -5)
        #     output[index] = 5
        # for index in injection_2:
        #     # if random.randint(0, 1) == 0:
        #     #     output[index] = random.randint(5, 8)
        #     # else:
        #     #     output[index] = random.randint(-8, -5)
        #     output[index] = -5
        output = np.expand_dims(output, 1)
        output = scaler.fit_transform(output)
        # if verbose:
        #     plt.figure(figsize=(9, 3))
        #     plt.plot(t, output, 'k', lw=2)
        #     plt.xlabel('$t$')
        #     plt.ylabel('$s$')
        #     plt.grid(True)
        #     plt.tight_layout()
        #     plt.show()
        #     plt.savefig('./figures/0/T_0.png')
        #     plt.close()
        # if verbose:
        #     linecolors = ['red' if i in injection_3 else 'blue' for i in range(249)]
        #     segments_y = np.r_[output[0], output[1:-1].repeat(2), output[-1]].reshape(-1, 2)
        #     segments_x = np.r_[t[0], t[1:-1].repeat(2), t[-1]].reshape(-1, 2)
        #     segments = [list(zip(x_, y_)) for x_, y_ in zip(segments_x, segments_y)]
        #
        #     plt.figure(figsize=(9, 3))
        #     # plt.margins(0.02)
        #     ax = plt.axes()
        #
        #     # Add a collection of lines
        #     ax.add_collection(LineCollection(segments, colors=linecolors, lw=2))
        #
        #     # Set x and y limits... sadly this is not done automatically for line
        #     # collections
        #     plt.yticks([0, 0.25, 0.5, 0.75, 1])
        #     ax.set_xlim(-10, 260)
        #     ax.set_ylim(-0.10, 1.10)
        #
        #     plt.xlabel('$t$')
        #     plt.ylabel('$s$')
        #     plt.grid(True)
        #     plt.tight_layout()
        #     plt.margins(0.1)
        #
        #     plt.show()
        #     # plt.savefig('./figures/0/T_0.png')
        #     # plt.close()
    elif case == 1:
        t_1 = np.arange(0, N // 2)
        t_2 = np.arange(N // 2, N)
        trend_1 = 8 * np.sin(0.15 * t_1)
        trend_2 = 8 * np.sin(0.4 * t_2)
        t = np.concatenate([t_1, t_2])
        trend = np.concatenate([trend_1, trend_2])
        if noise:
            noise = 2 * (np.random.rand(N) - 0.2)
        output = trend + noise
        output = np.expand_dims(output, 1)
        scaler = MinMaxScaler()
        output = scaler.fit_transform(output)
        if verbose:
            plt.figure(figsize=(9, 3))
            plt.plot(t, output, lw=2)
            plt.xlabel('t')
            plt.ylabel('s')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            # plt.savefig('./figures/0/T_1.png')
            # plt.close()
    elif case == 2:
        t = np.arange(0, N)
        trend = 8 * np.sin(1.5 * t / 200 * np.sin(0.4 * t))
        if noise:
            noise = 2 * (np.random.rand(N) - 0.5)
        output = trend + noise
        output = np.expand_dims(output, 1)
        scaler = MinMaxScaler()
        output = scaler.fit_transform(output)
        if verbose:
            plt.figure(figsize=(9, 3))
            plt.plot(t, output, lw=2)
            plt.xlabel('t')
            plt.ylabel('s')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            # plt.savefig('./figures/0/T_2.png')
            # plt.close()
    elif case == 3:
        t_1 = np.arange(0, 120)
        t_2 = np.arange(120, 140)
        t_3 = np.arange(140, N)
        t = np.concatenate([t_1, t_2, t_3])
        trend_1 = 8 * np.sin(0.2 * t_1)
        trend_2 = 0 * t_2
        trend_3 = 8 * np.sin(0.2 * t_3)
        if noise:
            noise_1 = 3 * (np.random.rand(120-0) - 0.5)
            noise_3 = 3 * (np.random.rand(N - 140) - 0.5)
        output = np.concatenate([trend_1 + noise_1, trend_2, trend_3 + noise_3])
        output = np.expand_dims(output, 1)
        scaler = MinMaxScaler()
        output = scaler.fit_transform(output)
        if verbose:
            plt.figure(figsize=(9, 3))
            plt.plot(t, output, lw=2)
            plt.xlabel('t')
            plt.ylabel('s')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            # plt.savefig('./figures/0/T_3.png')
            # plt.close()
    elif case == 4:
        t = np.arange(0, N)
        trend = 0.001 * (t - N // 2) ** 2
        p1, p2 = 20, 30
        periodic1 = 2 * np.sin(2 * pi * t / p1)
        np.random.seed(123)  # So we generate the same noisy time series every time.
        if noise:
            noise = 2 * (np.random.rand(N) - 0.5)
        output = trend + periodic1 + noise
        output = np.expand_dims(output, 1)
        scaler = MinMaxScaler()
        output = scaler.fit_transform(output)
        if verbose:
            plt.figure(figsize=(9, 3))
            plt.plot(t, output, lw=2)
            plt.xlabel('t')
            plt.ylabel('s')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            # plt.savefig('./figures/0/T_4.png')
            # plt.close()
    return output.astype(np.float32), np.ones(output.shape)


def get_loader(input, label=None, batch_size=128, shuffle=False, from_numpy=False, drop_last=False):
    """Convert input and label Tensors to a DataLoader

        If label is None, use a dummy Tensor
    """
    if label is None:
        label = input
    if from_numpy:
        input = torch.from_numpy(input)
        label = torch.from_numpy(label)
    loader = DataLoader(TensorDataset(input, label), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return loader


def cross_brain_get_loader(input_x, input_y, label=None, batch_size=128, shuffle=False, from_numpy=False, drop_last=False):
    """Convert input and label Tensors to a DataLoader
        If label is None, use a dummy Tensor
    """
    if label is None:
        label = input_x
    if from_numpy:
        input_x = torch.from_numpy(input_x)
        input_y = torch.from_numpy(input_y)
        label = torch.from_numpy(label)
    loader = DataLoader(TensorDataset(input_x, input_y, label), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return loader


def partition_data(ts, label, part_number=10):
    splitted_data = np.array_split(ts, part_number, axis=0)
    splitted_label = np.array_split(label, part_number, axis=0)
    return splitted_data, splitted_label


def create_batch_data(X, y=None, cutting_size=128, shuffle=False, from_numpy=False, drop_last=True):
    '''Convert X and y Tensors to a DataLoader

            If y is None, use a dummy Tensor
    '''
    if y is None:
        y = X
    if from_numpy:
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
    loader = DataLoader(TensorDataset(X, y), batch_size=cutting_size, shuffle=shuffle, drop_last=drop_last)
    b_X = []
    b_Y = []
    for i, (batch_X, batch_y) in enumerate(loader):
        b_X.append(batch_X)
        b_Y.append(batch_y)
    # return tensor
    b_X = torch.stack(b_X)
    b_Y = torch.stack(b_Y)
    return b_X, b_Y


def read_S5_dataset(file_name, normalize=True):
    abnormal = pd.read_csv(file_name, header=0, index_col=None)
    abnormal_data = abnormal['value'].values.astype(dtype='float32')
    abnormal_label = abnormal['is_anomaly'].values
    # Normal = 0, Abnormal = 1 => # Normal = 1, Abnormal = -1

    abnormal_data = np.expand_dims(abnormal_data, axis=1)
    abnormal_label = np.expand_dims(abnormal_label, axis=1)

    if normalize == True:
        scaler = MinMaxScaler(feature_range=(0, 1))
        abnormal_data = scaler.fit_transform(abnormal_data)

    abnormal_label[abnormal_label == 1] = -1
    abnormal_label[abnormal_label == 0] = 1
    return abnormal_data, abnormal_label


def read_NAB_dataset(file_name, normalize=True):
    with open('./data/NAB/labels/combined_windows.json') as data_file:
        json_label = json.load(data_file)
    abnormal = pd.read_csv(file_name, header=0, index_col=0)
    abnormal['label'] = 1
    list_windows = json_label.get(os.path.basename(file_name))
    for window in list_windows:
        start = window[0]
        end = window[1]
        abnormal.loc[start:end, 'label'] = -1

    abnormal_data = abnormal['value'].values.astype(dtype='float32')
    # abnormal_preprocessing_data = np.reshape(abnormal_preprocessing_data, (abnormal_preprocessing_data.shape[0], 1))
    abnormal_label = abnormal['label'].values

    abnormal_data = np.expand_dims(abnormal_data, axis=1)
    abnormal_label = np.expand_dims(abnormal_label, axis=1)

    if normalize == True:
        scaler = MinMaxScaler(feature_range=(0, 1))
        abnormal_data = scaler.fit_transform(abnormal_data)

    # Normal = 1, Abnormal = -1
    return abnormal_data, abnormal_label


def read_UAH_dataset(file_folder, normalize=True):
    def calculate_steering_angle(a):
        b = np.zeros(shape=(a.shape[0], 1))
        for i in range(a.size):
            if i == 0:
                b[i] = a[i]
            else:
                b[i] = (a[i] - a[i - 1])
                if b[i] >= 180:
                    b[i] = 360 - b[i]
                elif -180 < b[i] < 180:
                    b[i] = abs(b[i])
                elif b[i] <= -180:
                    b[i] = b[i] + 360
        return b

    def calculate_by_previous_element(a):
        b = np.zeros(shape=(a.shape[0], 1))
        for i in range(a.size):
            if i == 0:
                b[i] = 0
            else:
                b[i] = (a[i] - a[i - 1])
        return b

    def read_raw_GPS_dataset(folder_name):
        dataset = np.loadtxt(fname=folder_name + '/' + os.path.basename(folder_name) + '_RAW_GPS.txt', delimiter=' ',
                             usecols=(1, 7))
        return dataset

    def read_timestamp_and_label_of_semantic_dataset(folder_name):
        dataset = np.loadtxt(fname=folder_name + '/' + os.path.basename(folder_name) + '_SEMANTIC_ONLINE.txt',
                             delimiter=' ', usecols=(0, 23, 24, 25))
        return dataset

    def preprocess_raw_data(raw_data):
        speed_array = raw_data[:, 0]
        dir_array = raw_data[:, 1]

        # calculate acceleration (diff of speed)
        acceleration_array = calculate_by_previous_element(speed_array)

        # calculate jerk (diff of acceleration)
        jerk_array = calculate_by_previous_element(acceleration_array)

        # calculate steering (diff of direction)
        steering_array = calculate_steering_angle(dir_array)

        add_acceleration = np.c_[speed_array, acceleration_array]
        add_jerk = np.c_[add_acceleration, jerk_array]
        add_steering = np.c_[add_jerk, steering_array]

        return add_steering

    def compute_label_for_semantic(semantic_online_data):
        label = np.zeros(semantic_online_data.shape[0])
        for i in range(semantic_online_data.shape[0]):
            if semantic_online_data[i][0] <= semantic_online_data[i][1] or semantic_online_data[i][0] <= \
                    semantic_online_data[i][2] or semantic_online_data[i][0] <= semantic_online_data[i][1] + \
                    semantic_online_data[i][2]:
                label[i] = -1
            else:
                label[i] = 1
        return label

    abnormal = read_raw_GPS_dataset(file_folder)
    abnormal_data = preprocess_raw_data(abnormal)

    if normalize:
        scaler = MinMaxScaler(feature_range=(0, 1))
        abnormal_data = scaler.fit_transform(abnormal_data)

    abnormal_label = read_timestamp_and_label_of_semantic_dataset(file_folder)
    abnormal_label_data = compute_label_for_semantic(abnormal_label[:, [1, 2, 3]])

    return abnormal_data, abnormal_label_data


def read_2D_dataset(file_name, normalize=True):
    file_name_wo_path = Path(file_name).name
    parent_path = Path(file_name).parent.parent
    train_frame = pd.read_csv(str(parent_path) + '/train/' + file_name_wo_path, header=None, index_col=None, sep=' ')
    test_frame = pd.read_csv(str(parent_path) + '/test/' + file_name_wo_path, skiprows=1, header=None, index_col=None, sep=' ')
    train_data = train_frame.iloc[:, [0, 1]].values.astype(dtype='float32')
    test_data = test_frame.iloc[:, [0, 1]].values.astype(dtype='float32')
    test_label = test_frame.iloc[:, 2].values
    # Normal = 0, Abnormal = 1 => # Normal = 1, Abnormal = -1

    # abnormal_data = np.expand_dims(abnormal_data, axis=1)
    test_label = np.expand_dims(test_label, axis=1)

    if normalize:
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)

    test_label[test_label == 2] = -1
    test_label[test_label == 0] = 1
    return train_data, test_data, test_label


def read_ECG_dataset(file_name, normalize=True):
    abnormal = pd.read_csv(file_name, header=None, index_col=None, skiprows=0, sep=',')
    abnormal_data = abnormal.iloc[:, [1, 2]].values.astype(dtype='float32')
    abnormal_label = abnormal.iloc[:, 3].values
    # Normal = 0, Abnormal = 1 => # Normal = 1, Abnormal = -1

    # abnormal_data = np.expand_dims(abnormal_data, axis=1)
    abnormal_label = np.expand_dims(abnormal_label, axis=1)

    if normalize:
        scaler = MinMaxScaler(feature_range=(0, 1))
        abnormal_data = scaler.fit_transform(abnormal_data)

    abnormal_label[abnormal_label == 1] = -1
    abnormal_label[abnormal_label == 0] = 1
    return abnormal_data, abnormal_label


def read_GD_dataset(file_name, normalize=True):
    abnormal = pd.read_csv(file_name, header=0, index_col=0)
    abnormal_data = abnormal[
        ['MotorData.ActCurrent', 'MotorData.ActPosition', 'MotorData.ActSpeed', 'MotorData.IsAcceleration',
         'MotorData.IsForce', 'MotorData.Motor_Pos1reached', 'MotorData.Motor_Pos2reached',
         'MotorData.Motor_Pos3reached',
         'MotorData.Motor_Pos4reached', 'NVL_Recv_Ind.GL_Metall', 'NVL_Recv_Ind.GL_NonMetall',
         'NVL_Recv_Storage.GL_I_ProcessStarted', 'NVL_Recv_Storage.GL_I_Slider_IN', 'NVL_Recv_Storage.GL_I_Slider_OUT',
         'NVL_Recv_Storage.GL_LightBarrier', 'NVL_Send_Storage.ActivateStorage', 'PLC_PRG.Gripper',
         'PLC_PRG.MaterialIsMetal']].values.astype(dtype='float32')
    if normalize:
        scaler = MinMaxScaler(feature_range=(0, 1))
        abnormal_data = scaler.fit_transform(abnormal_data)

    abnormal_label = abnormal['Label'].values
    # Normal = 0, Abnormal = 2 => # Normal = 1, Abnormal = -1

    abnormal_label = np.expand_dims(abnormal_label, axis=1)

    abnormal_label[abnormal_label != 0] = -1
    abnormal_label[abnormal_label == 0] = 1
    return abnormal_data, abnormal_label


def read_HSS_dataset(file_name, normalize=True):
    abnormal = pd.read_csv(file_name, header=0, index_col=0)
    abnormal_data = abnormal[
        ['I_w_BLO_Weg', 'O_w_BLO_power', 'O_w_BLO_voltage', 'I_w_BHL_Weg', 'O_w_BHL_power', 'O_w_BHL_voltage',
         'I_w_BHR_Weg', 'O_w_BHR_power', 'O_w_BHR_voltage', 'I_w_BRU_Weg', 'O_w_BRU_power', 'O_w_BRU_voltage',
         'I_w_HR_Weg', 'O_w_HR_power', 'O_w_HR_voltage', 'I_w_HL_Weg', 'O_w_HL_power', 'O_w_HL_voltage']].values.astype(dtype='float32')
    if normalize:
        scaler = MinMaxScaler(feature_range=(0, 1))
        abnormal_data = scaler.fit_transform(abnormal_data)

    abnormal_label = abnormal['Labels'].values
    # Normal = 0, Abnormal = 1 => # Normal = 1, Abnormal = -1

    abnormal_label = np.expand_dims(abnormal_label, axis=1)

    abnormal_label[abnormal_label != 0] = -1
    abnormal_label[abnormal_label == 0] = 1
    return abnormal_data, abnormal_label


def read_SMD_dataset(file_name, normalize=True):
    file_name_wo_path = Path(file_name).name
    parent_path = Path(file_name).parent.parent
    train_data = pd.read_csv(str(parent_path) + '/train/' + file_name_wo_path, header=None, index_col=None)
    test_data = pd.read_csv(str(parent_path) + '/test/' + file_name_wo_path, header=None, index_col=None)
    test_label = pd.read_csv(str(parent_path) + '/test_label/' + file_name_wo_path, header=None, index_col=None)
    if normalize:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
    test_label[test_label != 0] = -1
    test_label[test_label == 0] = 1
    return train_data.astype(dtype='float32'), test_data.astype(dtype='float32'), test_label.to_numpy()

def read_SMAP_dataset(file_name, normalize=True):
    file_name_wo_path = Path(file_name).name
    file_name_wo_path_extension = Path(file_name).stem
    parent_path = Path(file_name).parent.parent
    train_data = np.load(str(parent_path) + '/train/' + file_name_wo_path)
    test_data = np.load(str(parent_path) + '/test/' + file_name_wo_path)
    test_label = pd.read_csv(str(parent_path.parent) + '/labeled_anomalies.csv', header=0, index_col=None)
    num_values = test_label.loc[test_label['chan_id'] == file_name_wo_path_extension]['num_values'].item()
    idx_anomalies = ast.literal_eval(test_label.loc[test_label['chan_id'] == file_name_wo_path_extension]['anomaly_sequences'].to_numpy()[0])
    labels = []
    j = 0
    for i in range(num_values):
        for idx in range(j, len(idx_anomalies)):
            if idx_anomalies[idx][0] < i < idx_anomalies[idx][1]:
                labels.append(-1)
                break
            else:
                labels.append(1)
                break
    labels = np.asarray(labels)
    if normalize:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
    assert test_data.shape[0] == labels.shape[0]
    return train_data.astype(dtype='float32'), test_data.astype(dtype='float32'), np.expand_dims(labels, axis=1)

def read_MSL_dataset(file_name, normalize=True):
    file_name_wo_path = Path(file_name).name
    file_name_wo_path_extension = Path(file_name).stem
    parent_path = Path(file_name).parent.parent
    train_data = np.load(str(parent_path) + '/train/' + file_name_wo_path)
    test_data = np.load(str(parent_path) + '/test/' + file_name_wo_path)
    test_label = pd.read_csv(str(parent_path.parent) + '/labeled_anomalies.csv', header=0, index_col=None)
    num_values = test_label.loc[test_label['chan_id'] == file_name_wo_path_extension]['num_values'].item()
    idx_anomalies = ast.literal_eval(test_label.loc[test_label['chan_id'] == file_name_wo_path_extension]['anomaly_sequences'].to_numpy()[0])
    labels = []
    j = 0
    for i in range(num_values):
        for idx in range(j, len(idx_anomalies)):
            if idx_anomalies[idx][0] < i < idx_anomalies[idx][1]:
                labels.append(-1)
                break
            else:
                labels.append(1)
                break
    labels = np.asarray(labels)
    if normalize:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
    return train_data.astype(dtype='float32'), test_data.astype(dtype='float32'), np.expand_dims(labels, axis=1)


def create_low_corr_tuples_h(input, k):
    '''
    :param x: time series [number of window, widow size, number of channel]
    :return: x and y
    '''
    x = []
    y = []
    assert k % 2 == 0, 'k should be even number'
    sample_wise = np.array_split(input, k, axis=0)
    for i in range(k):
        if i < k // 2:
            x.append(sample_wise[i])
            y.append(sample_wise[i + k//2])
        else:
            x.append(sample_wise[i])
            y.append(sample_wise[i - k//2])
    return np.concatenate(x, axis=0), np.concatenate(y, axis=0)


def realign_low_corr_tuples_h(input, k):
    '''
        :param x: time series [number of window, widow size, number of channel]
        :return: x and reconstruct_x align
        '''
    x = []
    reconstruct_x = []
    assert k % 2 == 0, 'k should be even number'
    sample_wise = np.array_split(input, k, axis=0)
    for i in range(k):
        if i < k // 2:
            x.append(sample_wise[i])
            reconstruct_x.append(sample_wise[i + k//2])
        else:
            x.append(sample_wise[i])
            reconstruct_x.append(sample_wise[i - k//2])
    return np.concatenate(x, axis=0), np.concatenate(reconstruct_x, axis=0)


def create_low_corr_tuples_v(input):
    '''
    :param x: time series [number of window, number of channel]
    :return: x and y
    '''
    assert input.shape[1] != 1, 'channel should be > 1'
    x = []
    y = []
    corr_matrix = np.corrcoef(input)
    channel_wise = np.split(input, axis=0)
    return 0

