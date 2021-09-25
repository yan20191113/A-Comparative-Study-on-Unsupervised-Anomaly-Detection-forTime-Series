import argparse
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams["font.size"] = 16
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve, auc, precision_recall_curve
from utils.config import MPConfig
from utils.device import get_free_device
from utils.logger import create_logger
from utils.metrics import MetricsResult
from utils.utils import str2bool
from utils.data_provider import dataset2path, read_dataset
from utils.metrics import SD_autothreshold, MAD_autothreshold, IQR_autothreshold, get_labels_by_threshold
from utils.utils import make_result_dataframe
from sklearn.metrics import f1_score


def sliding_dot_product(q, t):
    n = t.size
    m = q.size

    # Append t with n zeros
    ta = np.append(t, np.zeros(n))

    # Reverse Q
    qr = np.flip(q, 0)

    # Append qra
    qra = np.append(qr, np.zeros(2 * n - m))

    # Compute FFTs
    qraf = np.fft.fft(qra)
    taf = np.fft.fft(ta)

    # Compute the inverse FFT to the element-wise multiplication of qraf and taf
    qt = np.fft.ifft(np.multiply(qraf, taf))
    return qt[m:n]


def sliding_dot_product_stomp(q, t):
    n = t.size
    m = q.size

    # Append t with n zeros
    ta = np.append(t, np.zeros(n))

    # Reverse Q
    qr = np.flip(q, 0)

    # Append qra
    qra = np.append(qr, np.zeros(2 * n - m))

    # Compute FFTs
    qraf = np.fft.fft(qra)
    taf = np.fft.fft(ta)

    # Compute the inverse FFT to the element-wise multiplication of qraf and taf
    qt = np.fft.ifft(np.multiply(qraf, taf))
    return qt[m - 1:n]


def calculate_distance_profile(q, t, qt, a, sum_q, sum_q2, mean_t, sigma_t):
    n = t.size
    m = q.size

    b = np.zeros(n - m)
    dist = np.zeros(n - m)
    for i in range(0, n - m):
        b[i] = -2 * (qt[i].real - sum_q * mean_t[i]) / sigma_t[i]
        dist[i] = a[i] + b[i] + sum_q2
    return np.sqrt(np.abs(dist))


# The code below takes O(m) for each subsequence
# you should replace it for MASS
def compute_mean_std_for_query(Q):
    # Compute Q stats -- O(n)
    sumQ = np.sum(Q)
    sumQ2 = np.sum(np.power(Q, 2))
    return sumQ, sumQ2


def pre_compute_mean_std_for_TS(ta, m):
    na = len(ta)
    sum_t = np.zeros(na - m)
    sum_t2 = np.zeros(na - m)

    # Compute the stats for t
    cumulative_sum_t = np.cumsum(ta)
    cumulative_sum_t2 = np.cumsum(np.power(ta, 2))
    for i in range(na - m):
        sum_t[i] = cumulative_sum_t[i + m] - cumulative_sum_t[i]
        sum_t2[i] = cumulative_sum_t2[i + m] - cumulative_sum_t2[i]
    mean_t = np.divide(sum_t, m)
    mean_t2 = np.divide(sum_t2, m)
    mean_t_p2 = np.power(mean_t, 2)
    sigma_t2 = np.subtract(mean_t2, mean_t_p2)
    sigma_t = np.sqrt(sigma_t2)
    return sum_t, sum_t2, mean_t, mean_t2, mean_t_p2, sigma_t, sigma_t2


def pre_compute_mean_std_for_TS_stomp(ta, m):
    na = len(ta)
    # Compute the stats for t
    cumulative_sum_t = np.cumsum(ta)
    cumulative_sum_t2 = np.cumsum(np.power(ta, 2))
    sum_t = (cumulative_sum_t[m - 1:na] - np.concatenate(([0], cumulative_sum_t[0:na - m])))
    sum_t2 = (cumulative_sum_t2[m - 1:na] - np.concatenate(([0], cumulative_sum_t2[0:na - m])))
    mean_t = np.divide(sum_t, m)
    mean_t2 = np.divide(sum_t2, m)
    mean_t_p2 = np.power(mean_t, 2)
    sigma_t2 = np.subtract(mean_t2, mean_t_p2)
    sigma_t = np.sqrt(sigma_t2)
    return sum_t, sum_t2, mean_t, mean_t2, mean_t_p2, sigma_t, sigma_t2


# MUEEN’S ALGORITHM FOR SIMILARITY SEARCH (MASS)
def mass(Q, T, a, meanT, sigmaT):
    # Z-Normalisation
    if np.std(Q) != 0:
        Q = (Q - np.mean(Q)) / np.std(Q)
    QT = sliding_dot_product(Q, T)
    sumQ, sumQ2 = compute_mean_std_for_query(Q)
    return calculate_distance_profile(Q, T, QT, a, sumQ, sumQ2, meanT, sigmaT)


def element_wise_min(Pab, Iab, D, idx, ignore_trivial, m):
    for i in range(0, len(D)):
        if not ignore_trivial or (
                np.abs(idx - i) > m / 2.0):  # if it's a self-join, ignore trivial matches in [-m/2,m/2]
            if D[i] < Pab[i]:
                Pab[i] = D[i]
                Iab[i] = idx
    return Pab, Iab


def stamp(Ta, Tb, m):
    """
    Compute the Matrix Profile between time-series Ta and Tb.
    If Ta==Tb, the operation is a self-join and trivial matches are ignored.

    :param Ta: time-series, np.array
    :param Tb: time-series, np.array
    :param m: subsequence length
    :return: Matrix Profile, Nearest-Neighbor indexes
    """
    nb = len(Tb)
    na = len(Ta)
    Pab = np.ones(na - m) * np.inf
    Iab = np.zeros(na - m)
    idxes = np.arange(nb - m + 1)

    sumT, sumT2, meanT, meanT_2, meanTP2, sigmaT, sigmaT2 = pre_compute_mean_std_for_TS(Ta, m)

    a = np.zeros(na - m)
    for i in range(0, na - m):
        a[i] = (sumT2[i] - 2 * sumT[i] * meanT[i] + m * meanTP2[i]) / sigmaT2[i]

    ignore_trivial = np.atleast_1d(Ta == Tb).all()
    for idx in idxes:
        D = mass(Tb[idx: idx + m], Ta, a, meanT, sigmaT)
        if (ignore_trivial):
            # ignore trivial minimum and  maximum
            minIdx = int(np.maximum(idx - m / 2.0, 0))
            maxIdx = int(np.minimum(idx + m / 2.0, len(D)))
            D[minIdx:maxIdx:1] = np.inf

        Iab[Pab > D] = i
        Pab = np.minimum(Pab, D)
    return Pab, Iab


def stomp(T, m):
    """
    Compute the Matrix Profile with self join for T
    :param T: time-series, np.array
    :param Tb: time-series, np.array
    :param m: subsequence length
    :return: Matrix Profile, Nearest-Neighbor indexes
    """
    epsilon = 1e-2

    n = len(T)

    seq_l = n - m
    _, _, meanT, _, _, sigmaT, _ = pre_compute_mean_std_for_TS_stomp(T, m)

    Pab = np.full(seq_l + 1, np.inf)
    Iab = np.zeros(n - m + 1)
    ignore_trivial = True
    for idx in range(0, seq_l):
        # There's something with normalization
        Q_std = sigmaT[idx] if sigmaT[idx] > epsilon else epsilon
        if idx == 0:
            QT = sliding_dot_product_stomp(T[0:m], T).real
            QT_first = np.copy(QT)
        else:
            QT[1:] = QT[0:-1] - (T[0:seq_l] * T[idx - 1]) + (T[m:n] * T[idx + m - 1])
            QT[0] = QT_first[idx]

        # Calculate distance profile
        D = (2 * (m - (QT - m * meanT * meanT[idx]) / (Q_std * sigmaT)))
        D[D < epsilon] = 0
        if (ignore_trivial):
            # ignore trivial minimum and  maximum
            minIdx = int(np.maximum(idx - m / 2.0, 0))
            maxIdx = int(np.minimum(idx + m / 2.0, len(D)))
            D[minIdx:maxIdx:1] = np.inf

        Iab[Pab > D] = idx
        np.minimum(Pab, D, Pab)

    np.sqrt(Pab, Pab)
    return Pab, Iab


# Quick Test
# def test_stomp(Ta, m):
#     start_time = time.time()
#
#     Pab, Iab = stomp(Ta, m)
#     print("--- %s seconds ---" % (time.time() - start_time))
#     plot_motif(Ta, Pab, Iab, m)
#     return Pab, Iab


# Quick Test
# def test_stamp(Ta, Tb, m):
#     start_time = time.time()
#
#     Pab, Iab = stamp(Ta, Tb, m)
#     print("--- %s seconds ---" % (time.time() - start_time))
#
#     plot_discord(Ta, Pab, Iab, m, )
#     return Pab, Iab


def plot_motif(Ta, values, indexes, m):
    plt.figure(figsize=(8, 4))
    plt.subplot(211)
    plt.plot(Ta, linestyle='--', alpha=0.5)
    plt.xlim((0, len(Ta)))

    print(np.argmax(values))

    plt.plot(range(np.argmin(values), np.argmin(values) + m), Ta[np.argmin(values):np.argmin(values) + m], c='g',
             label='Top Motif')
    plt.plot(range(np.argmax(values), np.argmax(values) + m), Ta[np.argmax(values):np.argmax(values) + m], c='r',
             label='Top Discord')

    plt.legend(loc='best')
    plt.title('Time-Series')

    plt.subplot(212)
    plt.title('Matrix Profile')
    plt.plot(range(0, len(values)), values, '#ff5722')
    plt.plot(np.argmax(values), np.max(values), marker='x', c='r', ms=10)
    plt.plot(np.argmin(values), np.min(values), marker='^', c='g', ms=10)

    plt.xlim((0, len(Ta)))
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.show()


def plot_discord(Ta, Tb, values, indexes, m):
    from matplotlib import gridspec
    plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[int(len(Ta) / len(Tb)), 1])

    plt.subplot(gs[0])
    plt.plot(Ta, linestyle='--')
    plt.xlim((0, len(Ta)))

    plt.plot(range(np.argmin(values), np.argmin(values) + m), Ta[np.argmin(values):np.argmin(values) + m], c='g',
             label='Best Match')
    plt.legend(loc='best')
    plt.title('Time-Series')
    plt.ylim((-3, 3))

    plt.subplot(gs[1])
    plt.plot(Tb)

    plt.title('Query')
    plt.xlim((0, len(Tb)))
    plt.ylim((-3, 3))

    plt.figure()
    plt.title('Matrix Profile')
    plt.plot(range(0, len(values)), values, '#ff5722')
    plt.plot(np.argmax(values), np.max(values), marker='x', c='r', ms=10)
    plt.plot(np.argmin(values), np.min(values), marker='^', c='g', ms=10)

    plt.xlim((0, len(Ta)))
    plt.xlabel('Index')
    plt.ylabel('Value')

    plt.show()


def plot_match(Ta, Tb, values, indexes, m):
    from matplotlib import gridspec
    plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[int(len(Ta) / len(Tb)), 1])

    plt.subplot(gs[0])
    plt.plot(Ta, linestyle='--')
    plt.xlim((0, len(Ta)))

    print(np.argmax(values))

    plt.plot(range(np.argmin(values), np.argmin(values) + m), Ta[np.argmin(values):np.argmin(values) + m], c='g',
             label='Best Match')
    plt.legend(loc='best')
    plt.title('Time-Series')
    plt.ylim((-3, 3))

    plt.subplot(gs[1])
    plt.plot(Tb)

    plt.title('Query')
    plt.xlim((0, len(Tb)))
    plt.ylim((-3, 3))

    plt.figure()
    plt.title('Matrix Profile')
    plt.plot(range(0, len(values)), values, '#ff5722')
    plt.plot(np.argmax(values), np.max(values), marker='x', c='r', ms=10)
    plt.plot(np.argmin(values), np.min(values), marker='^', c='g', ms=10)

    plt.xlim((0, len(Ta)))
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.show()


def RunModel(train_filename, test_filename, label_filename, config, ratio):
    negative_sample = True if "noise" in config.dataset else False
    train_data, abnormal_data, abnormal_label = read_dataset(train_filename, test_filename, label_filename,
                                                             normalize=True, file_logger=file_logger, negative_sample=negative_sample, ratio=ratio)
    original_x_dim = abnormal_data.shape[1]
    config.x_dim = abnormal_data.shape[1]

    Pab = []
    for i in range(abnormal_data.shape[1]):
        ts = abnormal_data[:, i]
        Pab_i, _ = stomp(ts, config.pattern_size)
        Pab.append(np.nan_to_num(Pab_i))
    Pab = np.sum(Pab, axis=0)
    # final_zscore = zscore(Pab)
    # np_decision = create_label_based_on_zscore(final_zscore, 2.5, True)
    #np_decision = create_label_based_on_quantile(-Pab, quantile=99)

    # higher -Pab is more likely to be anomalies.
    SD_Tmin, SD_Tmax = SD_autothreshold(-Pab)
    SD_y_hat = get_labels_by_threshold(-Pab, Tmax=SD_Tmax, use_max=True, use_min=False)
    MAD_Tmin, MAD_Tmax = MAD_autothreshold(-Pab)
    MAD_y_hat = get_labels_by_threshold(-Pab, Tmax=MAD_Tmax, use_max=True, use_min=False)
    IQR_Tmin, IQR_Tmax = IQR_autothreshold(-Pab)
    IQR_y_hat = get_labels_by_threshold(-Pab, Tmax=IQR_Tmax, use_max=True, use_min=False)
    np_decision = {}
    np_decision["SD"] = SD_y_hat
    np_decision["MAD"] = MAD_y_hat
    np_decision["IQR"] = IQR_y_hat

    if config.save_output:
        if not os.path.exists('./outputs/NPY/{}/'.format(config.dataset)):
            os.makedirs('./outputs/NPY/{}/'.format(config.dataset))
        np.save('./outputs/NPY/{}/MP_hdim_None_rollingsize_{}_{}_pid={}.npy'.format(config.dataset, config.pattern_size, train_filename.stem, config.pid), Pab)

    # TODO metrics computation.

    # %%
    if config.save_figure:
        if original_x_dim == 1:
            plt.figure(figsize=(9, 3))
            plt.plot(ts, color='blue', lw=1.5)
            plt.title('Original Data')
            plt.grid(True)
            plt.tight_layout()
            # plt.show()
            plt.savefig( './figures/{}/Ori_MP_hdim_None_rollingsize_{}_{}_pid={}.png'.format(config.dataset, config.pattern_size, train_filename.stem, config.pid), dpi=300)
            plt.close()

            # Plot decoder output
            plt.figure(figsize=(9, 3))
            plt.plot(Pab, color='blue', lw=1.5)
            plt.title('Profile Output')
            plt.grid(True)
            plt.tight_layout()
            # plt.show()
            plt.savefig('./figures/{}/Profile_MP_hdim_None_rollingsize_{}_{}_pid={}.png'.format(config.dataset, config.pattern_size, train_filename.stem, config.pid), dpi=300)
            plt.close()

            t = np.arange(0, abnormal_data.shape[0])
            markercolors = ['blue' for i in range(config.pattern_size-1)] + ['blue' if i == 1 else 'red' for i in abnormal_label[config.pattern_size-1: ]]
            markersize = [4 for i in range(config.pattern_size-1)] + [4 if i == 1 else 25 for i in abnormal_label[config.pattern_size-1: ]]
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
                './figures/{}/VisInp_MP_hdim_None_rollingsize_{}_{}_pid={}.png'.format(config.dataset, config.pattern_size, train_filename.stem, config.pid),
                dpi=600)
            plt.close()

            markercolors = ['blue' if i == 1 else 'red' for i in np_decision["SD"]] + ['blue' for i in range(config.pattern_size-1)]
            markersize = [4 if i == 1 else 25 for i in np_decision["SD"]] + [4 for i in range(config.pattern_size-1)]
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
                './figures/{}/VisOut_MP_hdim_None_rollingsize_{}_SD_{}_pid={}.png'.format(config.dataset, config.pattern_size, train_filename.stem, config.pid),
                dpi=300)
            plt.close()

            markercolors = ['blue' if i == 1 else 'red' for i in np_decision["MAD"]] + ['blue' for i in range(config.pattern_size-1)]
            markersize = [4 if i == 1 else 25 for i in np_decision["MAD"]] + [4 for i in range(config.pattern_size-1)]
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
                './figures/{}/VisOut_MP_hdim_None_rollingsize_{}_MAD_{}_pid={}.png'.format(config.dataset, config.pattern_size, train_filename.stem, config.pid),
                dpi=300)
            plt.close()

            markercolors = ['blue' if i == 1 else 'red' for i in np_decision["IQR"]] + ['blue' for i in range(config.pattern_size-1)]
            markersize = [4 if i == 1 else 25 for i in np_decision["IQR"]] + [4 for i in range(config.pattern_size-1)]
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
                './figures/{}/VisOut_MP_hdim_None_rollingsize_{}_IQR_{}_pid={}.png'.format(config.dataset, config.pattern_size, train_filename.stem, config.pid),
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
            cm = confusion_matrix(y_true=abnormal_label[config.pattern_size-1: ], y_pred=np_decision[threshold_method], labels=[1, -1])
            TN[threshold_method] = cm[0][0]
            FP[threshold_method] = cm[0][1]
            FN[threshold_method] = cm[1][0]
            TP[threshold_method] = cm[1][1]
            precision[threshold_method] = precision_score(y_true=abnormal_label[config.pattern_size-1: ], y_pred=np_decision[threshold_method], pos_label=pos_label)
            recall[threshold_method] = recall_score(y_true=abnormal_label[config.pattern_size-1: ], y_pred=np_decision[threshold_method], pos_label=pos_label)
            f1[threshold_method] = f1_score(y_true=abnormal_label[config.pattern_size-1: ], y_pred=np_decision[threshold_method], pos_label=pos_label)

        fpr, tpr, _ = roc_curve(y_true=abnormal_label[config.pattern_size-1: ], y_score=-Pab, pos_label=pos_label)
        roc_auc = auc(fpr, tpr)
        pre, re, _ = precision_recall_curve(y_true=abnormal_label[config.pattern_size-1: ], probas_pred=-Pab,
                                            pos_label=pos_label)
        pr_auc = auc(re, pre)
        metrics_result = MetricsResult(TN=TN, FP=FP, FN=FN, TP=TP, precision=precision, recall=recall, fbeta=f1, pr_auc=pr_auc, roc_auc=roc_auc)
        return metrics_result


if __name__ == '__main__':

    # %%
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, default=0)
    parser.add_argument('--x_dim', type=int, default=1)
    parser.add_argument('--ratio', type=float, default=0.05)
    parser.add_argument('--pattern_size', type=int, default=32)
    parser.add_argument('--save_output', type=str2bool, default=True)
    parser.add_argument('--save_results', type=str2bool, default=True)
    parser.add_argument('--save_figure', type=str2bool, default=True)
    parser.add_argument('--use_spot', type=str2bool, default=False)
    parser.add_argument('--save_config', type=str2bool, default=True)
    parser.add_argument('--load_config', type=str2bool, default=False)
    parser.add_argument('--server_run', type=str2bool, default=False)
    parser.add_argument('--robustness', type=str2bool, default=False)
    parser.add_argument('--pid', type=int, default=0)
    args = parser.parse_args()

    #for registered_dataset in dataset2path:
    #for registered_dataset in ["NAB", "Yahoo", "MSL", "SMAP", "nyc_taxi", "AIOps", "Credit"]:
    #for registered_dataset in ["NAB_noise", "MSL_noise"]:
    for registered_dataset in ["MSL", "SMAP", "SMD", "NAB", "AIOps", "Credit", "ECG", "nyc_taxi", "SWAT", "Yahoo"]:
    #for registered_dataset in ["Yahoo"]:

        # the dim in args is useless, which should be deleted in the future version.
        if "noise" in registered_dataset:
            args.dataset = registered_dataset + "_{:.2f}".format(args.ratio)
        else:
            args.dataset = registered_dataset

        if args.load_config:
            config = MPConfig(dataset=None, x_dim=None, pattern_size=None, save_output=None, save_figure=None,
                              use_spot=None, use_last_point=None, save_config=None, load_config=None, server_run=None,
                              robustness=None, pid=None, save_results=None)
            try:
                config.import_config('./config/{}/Config_MP_hdim_None_rollingsize_{}_pid={}.json'.format(config.dataset, config.pattern_size, config.pid))
            except:
                print('There is no config.')
        else:
            config = MPConfig(dataset=args.dataset, x_dim=args.x_dim, pattern_size=args.pattern_size,
                              save_output=args.save_output, save_figure=args.save_figure, use_spot=args.use_spot,
                              use_last_point=True, save_config=args.save_config, load_config=args.load_config,
                              server_run=args.server_run, robustness=args.robustness, pid=args.pid, save_results=args.save_results)
        if args.save_config:
            if not os.path.exists('./config/{}/'.format(config.dataset)):
                os.makedirs('./config/{}/'.format(config.dataset))
            config.export_config('./config/{}/Config_MP_hdim_None_rollingsize_{}_pid={}.json'.format(config.dataset, config.pattern_size, config.pid))
        # %%
        device = torch.device(get_free_device())
        # %%
        train_logger, file_logger, meta_logger = create_logger(dataset=args.dataset,
                                                               train_logger_name='mp_train_logger',
                                                               file_logger_name='mp_file_logger',
                                                               meta_logger_name='mp_meta_logger',
                                                               model_name='MP',
                                                               pid=args.pid)
        # %%
        if config.dataset not in dataset2path:
            raise ValueError("dataset {} is not registered.".format(config.dataset))
        else:
            train_path = dataset2path[config.dataset]["train"]
            test_path = dataset2path[config.dataset]["test"]
            label_path = dataset2path[config.dataset]["test_label"]

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
        #for train_file in [Path("../datasets/train/Yahoo/real_64.pkl")]:
            test_file = test_path / train_file.name
            label_file = label_path / train_file.name
            file_logger.info('============================')
            file_logger.info(train_file)

            metrics_result = RunModel(train_filename=train_file, test_filename=test_file, label_filename=label_file, config=config, ratio=args.ratio)
            result_dataframe = make_result_dataframe(metrics_result)

            if config.save_results == True:
                if not os.path.exists('./results/{}/'.format(config.dataset)):
                    os.makedirs('./results/{}/'.format(config.dataset))
                result_dataframe.to_csv('./results/{}/Results_MP_hdim_None_rollingsize_{}_{}_pid={}.csv'.format(config.dataset, config.pattern_size, train_file.stem, config.pid),
                                        index=False)
