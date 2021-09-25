import argparse
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from typing import Union
from skimage.util.shape import view_as_windows as viewW
from utils.metrics import reconstructed_probability_np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_values_of_dict_by_keys(dict, keys):
    values = []
    for key in keys:
        values.append(dict[key])
    return values


def make_result_dataframe(MetricsResult):
    """
        MetricsResult(TN=TN, FP=FP, FN=FN, TP=TP, precision=precision, recall=recall, fbeta=f1, pr_auc=pr_auc, roc_auc=roc_auc)
    """
    autothreshold_methods = list(MetricsResult.TN.keys())
    TN_list = get_values_of_dict_by_keys(MetricsResult.TN, autothreshold_methods)
    FP_list = get_values_of_dict_by_keys(MetricsResult.FP, autothreshold_methods)
    FN_list = get_values_of_dict_by_keys(MetricsResult.FN, autothreshold_methods)
    TP_list = get_values_of_dict_by_keys(MetricsResult.TP, autothreshold_methods)
    precision_list = get_values_of_dict_by_keys(MetricsResult.precision, autothreshold_methods)
    recall_list = get_values_of_dict_by_keys(MetricsResult.recall, autothreshold_methods)
    fbeta_list = get_values_of_dict_by_keys(MetricsResult.fbeta, autothreshold_methods)
    pr_auc = [MetricsResult.pr_auc] * len(autothreshold_methods)
    roc_auc = [MetricsResult.roc_auc] * len(autothreshold_methods)
    training_time = [MetricsResult.training_time] * len(autothreshold_methods)
    testing_time = [MetricsResult.testing_time] * len(autothreshold_methods)
    total_params = [MetricsResult.total_params] * len(autothreshold_methods)
    estimated_total_size = [MetricsResult.estimated_total_size] * len(autothreshold_methods)
    memory_usage_nvidia = [MetricsResult.memory_usage_nvidia] * len(autothreshold_methods)
    if MetricsResult.best_TN is not None:
        best_TN = [MetricsResult.best_TN] * len(autothreshold_methods)
        best_FP = [MetricsResult.best_FP] * len(autothreshold_methods)
        best_FN = [MetricsResult.best_FN] * len(autothreshold_methods)
        best_TP = [MetricsResult.best_TP] * len(autothreshold_methods)
        best_precision = [MetricsResult.best_precision] * len(autothreshold_methods)
        best_recall = [MetricsResult.best_recall] * len(autothreshold_methods)
        best_fbeta = [MetricsResult.best_fbeta] * len(autothreshold_methods)
        result_dict = {
            "autothreshold_methods": autothreshold_methods,
            "TN": TN_list,
            "FP": FP_list,
            "FN": FN_list,
            "TP": TP_list,
            "precision": precision_list,
            "recall": recall_list,
            "fbeta": fbeta_list,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "best_TN": best_TN,
            "best_FP": best_FP,
            "best_FN": best_FN,
            "best_TP": best_TP,
            "best_precision": best_precision,
            "best_recall": best_recall,
            "best_fbeta": best_fbeta,
            "min_valid_loss": [MetricsResult.min_valid_loss]*len(autothreshold_methods),
            "training_time": training_time,
            "testing_time": testing_time,
            "total_params": total_params,
            "estimated_total_size": estimated_total_size,
            "memory_usage_nvidia": memory_usage_nvidia,
        }
        result_dataframe = pd.DataFrame.from_dict(result_dict)
    else:
        result_dict = {
            "autothreshold_methods": autothreshold_methods,
            "TN": TN_list,
            "FP": FP_list,
            "FN": FN_list,
            "TP": TP_list,
            "precision": precision_list,
            "recall": recall_list,
            "fbeta": fbeta_list,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "min_valid_loss": [MetricsResult.min_valid_loss] * len(autothreshold_methods),
            "training_time": training_time,
            "testing_time": testing_time,
            "total_params": total_params,
            "estimated_total_size": estimated_total_size,
            "memory_usage_nvidia": memory_usage_nvidia,
        }
        result_dataframe = pd.DataFrame.from_dict(result_dict)
    return result_dataframe


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def reparameterized_sample(mean, std):
    """using std to sample"""
    eps = torch.FloatTensor(std.size()).normal_().to(device)
    eps = Variable(eps).to(device)
    return eps.mul(std).add_(mean)


def get_reconstruction_probability(x_original, x_mean, x_std):
    '''
    Get reconstruction probability
    :param x_original: original x [batch, length, feature]
    :param x_mean: x_mean [batch, length, feature]
    :param x_std: x_std [batch, length, feature]
    :return: probability score [batch, length, 1]
    '''

    score_all_batches = []
    for x_i, mu_i, std_i in zip(x_original.shape[0], x_mean.shape[0], x_std.shape[0]):  # loop over batch
        score_each_batch = []
        for x_j, mu_j, std_j in zip(x_i.shape[0], mu_i.shape[0], std_i.shape[0]):
            prob_score = reconstructed_probability_np(x_j, mu_j, std_j, sample_times=50)
            score_each_batch.append(prob_score)
        score_all_batches.append(score_each_batch)
    return score_all_batches


def percentile(t: torch.tensor, q: float) -> Union[int, float]:
    """
    Return the ``q``-th percentile of the flattened input tensor's data.

    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.

    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result


def build_trajectory_matrix(TS, window_size):
    """
    Create lagged trajectory matrix, _traj, from time series (_ts) and window size (_window_size)
    """
    # The number of column
    k = TS.shape[0] - window_size + 1
    # Create the trajectory matrix by pulling the relevant subseries of T, and stacking them as columns.
    traj = np.column_stack([TS[i:i + window_size] for i in range(0, k)])
    # Note: the i+window_size above gives us up to i+window_size-1, as numpy array upper bounds are exclusive.
    return traj


def singular_value_decompose(trajectory):
    """
    Decompose _traj matrix using SVD, return traj_elem
    """
    # intrinsic dimensionality of the trajectory space
    d = np.linalg.matrix_rank(trajectory)
    # SVD calculation to NumPy
    U, Sigma, V = np.linalg.svd(trajectory)
    V = V.T

    # Calculate the elementary matrices of X, storing them in a multidimensional NumPy array.
    # This requires calculating sigma_i * U_i * (V_i)^T for each i, or sigma_i * outer_product(U_i, V_i).
    # Note that Sigma is a 1D array of singular values, instead of the full M x K diagonal matrix.
    traj_elem = np.array([Sigma[i] * np.outer(U[:, i], V[:, i]) for i in range(0, d)])

    # Quick sanity check: the sum of all elementary matrices in X_elm should be equal to X, to within a
    # *very small* tolerance:
    if not np.allclose(trajectory, traj_elem.sum(axis=0), atol=1e-10):
        print("WARNING: The sum of X's elementary matrices is not equal to X!")

    return traj_elem


def l21shrink(epsilon, X):
    """
    Args:
        epsilon: the shrinkage parameter
        X: matrix to shrink on
    Ref:
        wiki Regularization: {https://en.wikipedia.org/wiki/Regularization_(mathematics)}
    Returns:
            The shrunk matrix
    """
    output = X.copy()
    norm = np.linalg.norm(X, ord=2, axis=0)
    for i in range(X.shape[1]):
        if norm[i] > epsilon:
            for j in range(X.shape[0]):
                output[j, i] = X[j, i] - epsilon * X[j, i] / norm[i]
        else:
            output[:, i] = 0.
    return output


def l21shrink_torch(epsilon, X):
    """
    Args:
        epsilon: the shrinkage parameter
        X: matrix to shrink on
    Ref:
        wiki Regularization: {https://en.wikipedia.org/wiki/Regularization_(mathematics)}
    Returns:
            The shrunk matrix
    """
    output = X.clone()
    norm = torch.norm(X, p='fro', dim=0)
    for i in range(X.shape[1]):
        if norm[i] > epsilon:
            for j in range(X.shape[0]):
                output[j, i] = X[j, i] - epsilon * X[j, i] / norm[i]
        else:
            output[:, i] = 0.
    return output


def l1shrink(epsilon, X):
    """
    @Original Author: Prof. Randy
    @Modified by: Chong Zhou
    update to python3: 03/15/2019
    Args:
        epsilon: the shrinkage parameter (either a scalar or a vector)
        X: the vector to shrink on

    Returns:
        The shrunk vector
    """
    output = np.array(X * 0.)
    for idx, ele in enumerate(X):
        if ele > epsilon:
            output[idx] = ele - epsilon
        elif ele < -epsilon:
            output[idx] = ele + epsilon
        else:
            output[idx] = 0.
    return output


def l1shrink_torch(epsilon, X):
    """
    @Original Author: Prof. Randy
    @Modified by: Chong Zhou
    update to python3: 03/15/2019
    Args:
        epsilon: the shrinkage parameter (either a scalar or a vector)
        X: the vector to shrink on

    Returns:
        The shrunk vector
    """
    output = X.clone().detach() * 0
    for idx, ele in enumerate(X):
        if ele > epsilon:
            output[idx] = ele - epsilon
        elif ele < -epsilon:
            output[idx] = ele + epsilon
        else:
            output[idx] = 0.
    return output


def components_to_df(n, X, original_TS, TS_comps):
    d = np.linalg.matrix_rank(X)
    """
    Returns all the time series components in a single Pandas DataFrame object.
    """
    if n > 0:
        n = min(n, d)
    else:
        n = d
    # Create list of columns - call them F0, F1, F2, ...
    cols = ["T{}".format(i) for i in range(n)]
    return pd.DataFrame(TS_comps[:, :n], columns=cols, index=original_TS.index)


def search_weight(trajectory_element_i, d):
    L, K = trajectory_element_i.shape

    w = np.array(list(np.arange(L) + 1) + [L] * (K - L - 1) + list(np.arange(L) + 1)[::-1])

    # Get all the components of the toy series, store them as columns in F_elem array.
    F_elem = np.array([reconstruct_time_series(trajectory_element_i[i]) for i in range(d)])

    # Calculate the individual weighted norms, ||F_i||_w, first, then take inverse square-root so we don't have to
    # later.
    F_wnorms = np.array([w.dot(F_elem[i] ** 2) for i in range(d)])
    F_wnorms = F_wnorms ** -0.5

    # Calculate the w-corr matrix. The diagonal elements are equal to 1, so we can start with an identity matrix
    # and iterate over all pairs of i's and j's (i != j), noting that Wij = Wji.
    Wcorr = np.identity(d)
    for i in range(d):
        for j in range(i + 1, d):
            Wcorr[i, j] = abs(w.dot(F_elem[i] * F_elem[j]) * F_wnorms[i] * F_wnorms[j])
            Wcorr[j, i] = Wcorr[i, j]


def strided_indexing_roll(a, r):
    # Concatenate with sliced to cover all rolls
    p = np.full((a.shape[0], a.shape[1] - 1), np.nan)
    a_ext = np.concatenate((p, a, p), axis=1)

    # Get sliding windows; use advanced-indexing to select appropriate ones
    n = a.shape[1]
    return viewW(a_ext, (1, n))[np.arange(len(r)), -r + (n - 1), 0]


def sliding_window(T, window_length, stride):
    # frame_data = _ts[np.arange(_ts.shape[0] - _window_length + 1)[:, None] + np.arange(_window_length)]
    frame_data = T[:, np.arange(T.shape[1] - window_length + 1)[:, None] + np.arange(window_length)]
    frame_data = frame_data[::stride]
    return frame_data


def sliding_window_torch(T, window_length, stride):
    # frame_data = _ts[torch.arange(_ts.shape[0] - _window_length + 1)[:, None] + torch.arange(_window_length)]
    frame_data = T[:, torch.arange(T.shape[1] - window_length + 1)[:, None] + torch.arange(window_length)]
    frame_data = frame_data[::stride]
    return frame_data


def desliding_window(X, window_length, stride):
    overlapping_size = window_length - stride
    ts_length = X.shape[0] + window_length - stride
    reconstructed_ts = np.zeros(shape=(ts_length, 1))
    for i, record_i in enumerate(reconstructed_ts.shape[0]):
        if i == 0:
            reconstructed_ts[i] = X[i][0]
        elif i == reconstructed_ts.shape[0]:
            reconstructed_ts[i] = X[i][X.shape[1]]
        else:
            value = 0
            for j in range(overlapping_size):
                value = value + X[i - j][j]
                if i - j < 0:
                    break
            reconstructed_ts[i] = value
    return reconstructed_ts


def desliding_window_torch(X, window_length, stride):
    overlapping_size = window_length - stride
    ts_length = X.shape[0] + window_length - stride
    reconstructed_ts = torch.zeros(size=(ts_length, 1))
    for i, record_i in enumerate(reconstructed_ts):
        value = 0
        element = 0
        for j in range(overlapping_size + 1):
            if i - j < 0 or i - j > reconstructed_ts.shape[0] - 1:
                break
            else:
                while i - j >= X.shape[0]:
                    j = j + 1
                value = value + X[i - j][j]
                element = element + 1
        reconstructed_ts[i] = value / element
    return reconstructed_ts


def Hankelise(X):
    """
    Hankelises the matrix X, returning H(X).
    """
    L, K = X.shape
    transpose = False
    if L > K:
        # The Hankelisation below only works for matrices where L < K.
        # To Hankelise a L > K matrix, first swap L and K and tranpose X.
        # Set flag for HX to be transposed before returning.
        X = X.T
        L, K = K, L
        transpose = True

    HX = np.zeros((L, K))
    # I know this isn't very efficient...
    for m in range(L):
        for n in range(K):
            s = m + n
            if 0 <= s <= L - 1:
                for l in range(0, s + 1):
                    HX[m, n] += 1 / (s + 1) * X[l, s - l]
            elif L <= s <= K - 1:
                for l in range(0, L - 1):
                    HX[m, n] += 1 / (L - 1) * X[l, s - l]
            elif K <= s <= K + L - 2:
                for l in range(s - K + 1, L):
                    HX[m, n] += 1 / (K + L - s - 1) * X[l, s - l]
    if transpose:
        return HX.T
    else:
        return HX


def reconstruct_time_series(X_i):
    """Averages the anti-diagonals of the given elementary matrix, X_i, and returns a time series."""
    TS = []

    for channel in range(X_i.shape[1]):
        X_channel = X_i[:, channel]
        X_rev = np.flip(X_channel, axis=[1])
        TS_channel = np.array(
            [X_rev.diagonal(i, axis1=1, axis2=2).mean() for i in range(-X_i.shape[2] + 1, X_i.shape[3])])
        TS.append(TS_channel)
    # Full credit to Mark Tolonen at https://stackoverflow.com/a/6313414 for this one:
    TS_tensor = np.stack(TS)
    return TS_tensor


def reconstruct_time_series_torch(X_i):
    '''
    :param X_i: Spectral matrix, [batch, channel, width, height]
    :return: time series
    '''
    """Averages the anti-diagonals of the given elementary matrix, X_i, and returns a time series."""
    TS = []

    # Reverse the column ordering of X_i
    for channel in range(X_i.shape[1]):
        X_channel = X_i[:, channel]
        X_rev = torch.flip(X_channel, dims=[1])
        TS_channel = torch.tensor(
            [X_rev.diagonal(i, dim1=1, dim2=2).mean() for i in range(-X_i.shape[2] + 1, X_i.shape[3])],
            requires_grad=True).to(device)
        TS.append(TS_channel)
    # Full credit to Mark Tolonen at https://stackoverflow.com/a/6313414 for this one:
    TS_tensor = torch.stack(TS)
    return TS_tensor
