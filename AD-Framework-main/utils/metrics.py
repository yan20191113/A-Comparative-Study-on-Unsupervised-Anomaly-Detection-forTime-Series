import math
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, average_precision_score, \
    roc_curve, auc, cohen_kappa_score
from torch.autograd import Variable
from utils.spot import SPOT


class MetricsResult(object):
    def __init__(self, TN=None, FP=None, FN=None, TP=None, precision=None, recall=None, fbeta=None, pr_auc=None,
                 roc_auc=None, cks=None, best_TN=None, best_FP=None, best_FN=None, best_TP=None, best_precision=None,
                 best_recall=None, best_fbeta=None, best_pr_auc=None, best_roc_auc=None, best_cks=None, min_valid_loss=None,
                 training_time=None, testing_time=None, total_params=None, estimated_total_size=None, memory_usage_nvidia=None):
        self.TN = TN
        self.FP = FP
        self.FN = FN
        self.TP = TP
        self.precision = precision
        self.recall = recall
        self.fbeta = fbeta
        self.pr_auc = pr_auc
        self.roc_auc = roc_auc
        self.cks = cks
        self.best_TN = best_TN
        self.best_FP = best_FP
        self.best_FN = best_FN
        self.best_TP = best_TP
        self.best_precision = best_precision
        self.best_recall = best_recall
        self.best_fbeta = best_fbeta
        self.best_pr_auc = best_pr_auc
        self.best_roc_auc = best_roc_auc
        self.best_cks = best_cks
        self.min_valid_loss = min_valid_loss
        self.training_time = training_time
        self.testing_time = testing_time
        self.total_params = total_params
        self.estimated_total_size = estimated_total_size
        self.memory_usage_nvidia = memory_usage_nvidia



def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.

    Args:
        predict (np.ndarray): the predict label
        actual (np.ndarray): np.ndarray
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return f1, precision, recall, TP, TN, FP, FN

def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.

    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):

    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score < threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict

def calc_seq(score, label, threshold, calc_latency=False):
    """
    Calculate f1 score for a score sequence
    """
    if calc_latency:
        predict, latency = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        t = list(calc_point2point(predict, label))
        t.append(latency)
        return t
    else:
        predict = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        return calc_point2point(predict, label)


def bf_search(score, label, start, end=None, step_num=1, display_freq=1, verbose=True):
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).


    Returns:
        list: list for results
        float: the `threshold` for best-f1
    """
    # convert the label to 0 or 1
    label[label == 1] = 0
    label[label == -1] = 1
    if step_num is None or end is None:
        end = start
        step_num = 1
    search_step, search_range, search_lower_bound = step_num, end - start, start
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
    threshold = search_lower_bound
    m = (-1., -1., -1.)
    m_t = 0.0
    for i in range(search_step):
        threshold += search_range / float(search_step)
        target = calc_seq(score, label, threshold, calc_latency=True)
        if target[0] > m[0]:
            m_t = threshold
            m = target
        if verbose and i % display_freq == 0:
            print("cur thr: ", threshold, target, m, m_t)
    print(m, m_t)
    return m, m_t


def pot_eval(init_score, score, label, q=1e-3, level=0.02):
    """
    Run POT method on given score.
    Args:
        init_score (np.ndarray): The data to get init threshold.
            For `OmniAnomaly`, it should be the anomaly score of train set.
        score (np.ndarray): The data to run POT method.
            For `OmniAnomaly`, it should be the anomaly score of test set.
        label:
        q (float): Detection level (risk)
        level (float): Probability associated with the initial threshold t

    Returns:
        dict: pot result dict
    """
    s = SPOT(q)  # SPOT object
    s.fit(init_score, score)  # data import
    s.initialize(level=level, min_extrema=True)  # initialization step
    ret = s.run(dynamic=False)  # run
    print(len(ret['alarms']))
    print(len(ret['thresholds']))
    pot_th = -np.mean(ret['thresholds'])
    pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=True)
    p_t = calc_point2point(pred, label)
    print('POT result: ', p_t, pot_th, p_latency)
    return {
        'pot-f1': p_t[0],
        'pot-precision': p_t[1],
        'pot-recall': p_t[2],
        'pot-TP': p_t[3],
        'pot-TN': p_t[4],
        'pot-FP': p_t[5],
        'pot-FN': p_t[6],
        'pot-threshold': pot_th,
        'pot-latency': p_latency
    }

def calculate_ensemble_score(ensemble_score):
    final_score = np.median(ensemble_score, axis=0)
    return final_score

def calculate_average_metric(sum_of_score):
    '''
    Calculate average score of a set of multiple dataset
    :param sum_of_score: Python List [] of score
    :return: average score
    '''
    if len(sum_of_score) != 0:
        average_score = sum(sum_of_score) / float(len(sum_of_score))
    else:
        average_score = 0
    return average_score

def zscore(error):
    '''
    Calculate z-score using error
    :param error: error time series
    :return: z-score
    '''
    mu = np.nanmean(error)
    gamma = np.nanstd(error)
    z_score = (error - mu) / gamma
    return z_score

def create_label_based_on_zscore(zscore, threshold, sign=False):
    label = np.full(zscore.shape[0], 1)
    if not sign:
        label[zscore > threshold] = -1
        label[zscore < -threshold] = -1
    else:
        label[zscore > threshold] = -1
    # label[abs(zscore) > abs(threshold)] = -1
    return label

def create_label_based_on_quantile(score, quantile=90):
    # higher scores is more likely to be anomalies
    label = np.full(score.shape[0], 1)
    threshold = np.percentile(score, quantile)
    label[score > threshold] = -1
    return label

def create_top_K_label_based_on_reconstruction_error(error, k):
    label = np.full(error.shape[0], 1)
    outlier_indices = error.argsort()[-k:][::-1]
    for i in outlier_indices:
        label[i] = -1
    return label, outlier_indices

def calculate_precision_at_K(abnormal_label, score, k, type):
    y_pred_at_k = np.full(k, -1)
    if type == 1:  # Local Outlier Factor & Auto-Encoder Type
        # score[score > 2.2] = 1
        outlier_indices = score.argsort()[-k:][::-1]
    if type == 2:  # Isolation Forest & One-class SVM Type
        outlier_indices = score.argsort()[:k]
    abnormal_at_k = []
    for i in outlier_indices:
        abnormal_at_k.append(abnormal_label[i])
    abnormal_at_k = np.asarray(abnormal_at_k)
    precision_at_k = precision_score(abnormal_at_k, y_pred_at_k)
    return precision_at_k

def calculate_metrics(error, score, label):
    if score is None:
        score = zscore(error)

    pos_label = -1
    y_pred = create_label_based_on_zscore(zscore(score), threshold=1.5, sign=False)
    precision = precision_score(label, y_pred, pos_label=pos_label)
    recall = recall_score(label, y_pred, pos_label=pos_label)
    f1 = f1_score(label, y_pred, pos_label=pos_label)
    pc, rc, _ = precision_recall_curve(label, score, pos_label=pos_label)
    pr_auc = average_precision_score(label, score, pos_label=pos_label)
    fpr, tpr, _ = roc_curve(label, score, pos_label=pos_label)
    roc_auc = auc(np.nan_to_num(fpr), np.nan_to_num(tpr))
    cks = cohen_kappa_score(label, y_pred)
    return precision, recall, f1, pr_auc, roc_auc, cks

def kld_gaussian(mean_1, std_1, mean_2, std_2):
    """Using std to compute KLD"""
    if mean_2 is not None and std_2 is not None:
        kl_loss = 0.5 * torch.sum(2 * torch.log(std_2) - 2 * torch.log(std_1) + (std_1.pow(2) + (mean_1 - mean_2).pow(2)) / std_2.pow(2) - 1)
        # kl_loss = -0.5 * torch.sum(1 + 2 * torch.log(std_1) - 2 * torch.log(std_2) - (std_1.pow(2) + (mean_1 - mean_2).pow(2)) / std_2.pow(2))
    else:
        kl_loss = -0.5 * torch.sum(1 + std_1 - mean_1.pow(2) - std_1.exp())
    return kl_loss


def nll_bernoulli(x_original, x_reconstruct):
    return -torch.sum(x_original * torch.log(x_reconstruct) + (1 - x_original) * torch.log(1 - x_reconstruct))


def nll_gaussian(x_original, x_mean, x_std):
    return 0.5 * (torch.sum(x_std) + torch.sum(((x_original - x_mean) / x_std.mul(0.5).exp_()) ** 2))  # Owned definition
    # return torch.sum(torch.log(x_std) + (x_original - x_mean).pow(2) / (2 * x_std.pow(2)))  # VRNN
    # tmp = Variable(torch.ones(x_original.size(0), x_original.size(1)) * np.log(2 * np.pi)) + 2 * torch.log(x_std)
    # return 0.5 * torch.sum(tmp + 1.0 / x_std.pow(2) * (x_original - x_mean).pow(2))

def mse_loss(x_original, x_reconstruct):
    return F.mse_loss(x_original, x_reconstruct)


def beta_bernoulii(beta, x_original, x_reconstruct):
    return ((beta + 1) / (beta)) * (-torch.sum(x_original * torch.log(x_reconstruct).pow(beta) + (1 - x_original) * torch.log(1 - x_reconstruct).pow(beta) - 1)) \
           + torch.sum(torch.log(x_reconstruct).pow(beta + 1) + torch.log(1 - x_reconstruct).pow(beta + 1))


def beta_gaussian(beta, x_original, x_reconstruct):
    return ((beta + 1) / (beta)) * ((torch.sum((x_reconstruct - x_original).exp_()) ** 2) - 1)


def cross_entropy_loss(x_original, x_reconstruct):
    return F.binary_cross_entropy_with_logits(x_original, x_reconstruct, reduction='none')


def reconstructed_probability_np(x_original, x_mean, x_std, sample_times=100):
    reconstructed_prob = np.zeros((x_original.shape[0],), dtype='float32')
    for l in range(sample_times):
        x_mean = x_mean.reshape(x_original.shape)
        x_std = x_std.reshape(x_original.shape) + 0.00001
        for i in range(x_original.shape[0]):
            p_l = scipy.stats.multivariate_normal.pdf(x_original[i, :], x_mean[i, :], np.diag(x_std[i, :]))
            reconstructed_prob[i] += p_l
    reconstructed_prob /= sample_times
    return reconstructed_prob


def reconstructed_probability_torch(x_original, x_mean, x_std, sample_times=100):
    reconstructed_prob = torch.zeros([x_original.shape[0]], dtype=torch.float32)
    for l in range(sample_times):
        x_mean = x_mean.reshape(x_original.shape)
        x_std = x_std.reshape(x_original.shape) + 0.00001
        for i in range(x_original.shape[0]):
            p_l = torch.distributions.multivariate_normal.MultivariateNormal(x_original[i, :], x_mean[i, :], torch.diag(x_std[i, :]))
            reconstructed_prob[i] += p_l
    reconstructed_prob /= sample_times
    return reconstructed_prob


def nll_score(x_original, x_reconstruct, x_mean, x_std, type='mse', reduction='none'):
    def reparameterized_sample(x_mean, x_std):
        """using std to sample"""
        eps = torch.FloatTensor(x_std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(x_std).add_(x_mean)

    if type == 'mse':
        return F.mse_loss(x_original, x_reconstruct, reduction=reduction)
    elif type == 'entropy':
        return F.binary_cross_entropy_with_logits(x_original, x_reconstruct, reduction=reduction)
    elif type == 'bernoulli':
        return -torch.sum(x_original * torch.log(x_reconstruct) + (1 - x_original) * torch.log(1 - x_reconstruct))
    elif type == 'gaussian':
        # return -(-0.5 * torch.sum(((x_reconstruct - x_mean) / (2 * x_std).exp()).pow(2))) + (-torch.sum(x_std)) + (-0.5 * x_reconstruct.shape[1] * np.log(2 * math.pi))
        return 0.5 * (torch.sum(x_std) + torch.sum(((x_original - x_mean) / x_std.mul(0.5).exp_()) ** 2))


def SD_autothreshold(outlier_scores, a=3):
    """
        As shown in "Outlier Detection: How to Threshold Outlier Scores?", SD is used to threshold the outlier scores, where
        SD means the Standard Deviation.
        So the threshold can be calculated by the following equations.
            Tmin = mean - a * SD;
            Tmax = mean + a * SD;
        After Tmin and Tmax are obtained, if a value is larger than Tmax or smaller than Tmin, this value will be considered as
        anomaly.

    """
    outlier_scores = np.array(outlier_scores)
    mean = outlier_scores.mean()
    std = outlier_scores.std()
    Tmin = mean - a * std
    Tmax = mean + a * std
    return Tmin, Tmax


def MAD_autothreshold(outlier_scores, a=3, b=1.4826):
    """
        Tmin = median(outlier_scores) - a * MAD
        Tmax = median(outlier_scores) + a * MAD
        MAD = b * median(|outlier_scores-median(outlier_scores)|)
        The details of MAD can be found in "Outlier Detection: How to Threshold Outlier Scores?".
    """
    outlier_scores = np.array(outlier_scores)
    median = np.median(outlier_scores)
    MAD = b * np.median(np.abs((outlier_scores-median)))
    Tmin = median - a * MAD
    Tmax = median + a * MAD
    return Tmin, Tmax


def IQR_autothreshold(outlier_scores, c=1.5):
    """
        Tmin = Q1 - c * IQR
        Tmax = Q3 + c * IQR
        IQR = Q3 - Q1
        where Q1 and Q3 is the values ranking in 25% and 75%, respectively.
        The details of IQR can be found in "Outlier Detection: How to Threshold Outlier Scores?".
    """
    outlier_scores = np.array(outlier_scores)
    Q1 = np.percentile(outlier_scores, 25)
    Q3 = np.percentile(outlier_scores, 75)
    IQR = Q3 - Q1
    Tmin = Q1 - c * IQR
    Tmax = Q3 + c * IQR
    return Tmin, Tmax


def get_labels_by_threshold(outlier_scores, Tmin=None, Tmax=None, use_min=True, use_max=True):
    """
        if use_min is setted True, the value which is smaller than Tmin will be considered as anomaly.
        if use_max is setted True, the value which is larger than Tmax will be considered as anomaly.
    """
    outlier_scores = np.array(outlier_scores)
    # In the whole experiments, the labels of the normal data is setted as 1.
    labels = np.ones_like(outlier_scores)
    if use_min:
        labels[outlier_scores<Tmin] = -1
    if use_max:
        labels[outlier_scores>Tmax] = -1
    return labels
