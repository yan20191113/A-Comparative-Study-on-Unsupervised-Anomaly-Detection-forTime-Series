import argparse
import os
import matplotlib as mpl
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, roc_curve, auc, confusion_matrix
from sklearn.svm import OneClassSVM
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams["font.size"] = 16
import numpy as np
from utils.config import OCSVMConfig
from utils.logger import create_logger
from utils.outputs import OCSVMOutput
from utils.utils import str2bool
from utils.metrics import MetricsResult
from utils.data_provider import dataset2path, read_dataset
from utils.metrics import SD_autothreshold, MAD_autothreshold, IQR_autothreshold, get_labels_by_threshold
from utils.utils import make_result_dataframe
from sklearn.metrics import f1_score


class OCSVM(object):
    def __init__(self, file_name, config):
        self.dataset = config.dataset
        self.file_name = file_name

        self.x_dim = config.x_dim

        self.kernel = config.kernel
        self.degree = config.degree
        self.gamma = config.gamma
        self.coef0 = config.coef0
        self.tol = config.tol
        self.nu = config.nu

        self.pid = config.pid

        self.model = OneClassSVM(kernel=self.kernel, degree=self.degree, gamma=self.gamma, coef0=self.coef0,
                                 tol=self.tol, nu=self.nu)

    def fit(self, train_input, train_label, test_input, test_label):
        # Perform fit on X and returns labels for X.
        # Returns -1 for outliers and 1 for inliers.
        y_pred = self.model.fit_predict(train_input)
        decision_function = self.model.decision_function(train_input)

        ocsvm_output = OCSVMOutput(y_hat=y_pred, decision_function=decision_function)

        return ocsvm_output


def RunModel(train_filename, test_filename, label_filename, config):

    train_data, abnormal_data, abnormal_label = read_dataset(train_filename, test_filename, label_filename,
                                                             normalize=True)

    original_x_dim = abnormal_data.shape[1]

    config.x_dim = abnormal_data.shape[1]

    model = OCSVM(train_filename, config)
    ocsvm_output = model.fit(train_input=abnormal_data, train_label=abnormal_label, test_input=abnormal_data, test_label=abnormal_label)

    SD_Tmin, SD_Tmax = SD_autothreshold(-ocsvm_output.decision_function)
    SD_y_hat = get_labels_by_threshold(-ocsvm_output.decision_function, Tmax=SD_Tmax, use_max=True, use_min=False)
    MAD_Tmin, MAD_Tmax = MAD_autothreshold(-ocsvm_output.decision_function)
    MAD_y_hat = get_labels_by_threshold(-ocsvm_output.decision_function, Tmax=MAD_Tmax, use_max=True, use_min=False)
    IQR_Tmin, IQR_Tmax = IQR_autothreshold(-ocsvm_output.decision_function)
    IQR_y_hat = get_labels_by_threshold(-ocsvm_output.decision_function, Tmax=IQR_Tmax, use_max=True, use_min=False)
    ocsvm_output.y_hat = {}
    ocsvm_output.y_hat["SD"] = SD_y_hat
    ocsvm_output.y_hat["MAD"] = MAD_y_hat
    ocsvm_output.y_hat["IQR"] = IQR_y_hat

    if config.save_output == True:
        if not os.path.exists('./outputs/NPY/{}/'.format(config.dataset)):
            os.makedirs('./outputs/NPY/{}/'.format(config.dataset))
        np.save('./outputs/NPY/{}/Score_OCSVM_{}_pid={}.npy'.format(config.dataset, train_filename.stem, config.pid), ocsvm_output.decision_function)
        np.save('./outputs/NPY/{}/Pred_OCSVM_{}_pid={}.npy'.format(config.dataset, train_filename.stem, config.pid), ocsvm_output.y_hat)

    # %%
    if config.save_figure:
        if not os.path.exists('./figures/{}/'.format(config.dataset)):
            os.makedirs('./figures/{}/'.format(config.dataset))
        if original_x_dim == 1:
            plt.figure(figsize=(9, 3))
            plt.plot(abnormal_data, color='blue', lw=1.5)
            plt.title('Original Data')
            plt.grid(True)
            plt.tight_layout()
            # plt.show()
            plt.savefig('./figures/{}/Ori_OCSVM_{}_pid={}.png'.format(config.dataset, train_filename.stem, config.pid), dpi=300)
            plt.close()

            t = np.arange(0, abnormal_data.shape[0])
            markercolors = ['blue' if i == 1 else 'red' for i in abnormal_label]
            markersize = [4 if i == 1 else 25 for i in abnormal_label]
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
            plt.plot(abnormal_data, alpha=0.7)
            plt.scatter(t, abnormal_data, s=markersize, c=markercolors)
            # plt.show()
            plt.savefig('./figures/{}/VisInp_OCSVM_{}_pid={}.png'.format(config.dataset, train_filename.stem, config.pid), dpi=300)
            plt.close()

            markercolors = ['blue' if i == 1 else 'red' for i in ocsvm_output.y_hat["SD"]]
            markersize = [4 if i == 1 else 25 for i in ocsvm_output.y_hat["SD"]]
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
            plt.plot(abnormal_data, alpha=0.7)
            plt.scatter(t, abnormal_data, s=markersize, c=markercolors)
            # plt.show()
            plt.savefig('./figures/{}/VisOut_OCSVM_SD_{}_pid={}.png'.format(config.dataset, train_filename.stem, config.pid), dpi=300)
            plt.close()

            markercolors = ['blue' if i == 1 else 'red' for i in ocsvm_output.y_hat["MAD"]]
            markersize = [4 if i == 1 else 25 for i in ocsvm_output.y_hat["MAD"]]
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
            plt.plot(abnormal_data, alpha=0.7)
            plt.scatter(t, abnormal_data, s=markersize, c=markercolors)
            # plt.show()
            plt.savefig(
                './figures/{}/VisOut_OCSVM_MAD_{}_pid={}.png'.format(config.dataset, train_filename.stem, config.pid),
                dpi=300)
            plt.close()

            markercolors = ['blue' if i == 1 else 'red' for i in ocsvm_output.y_hat["IQR"]]
            markersize = [4 if i == 1 else 25 for i in ocsvm_output.y_hat["IQR"]]
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
            plt.plot(abnormal_data, alpha=0.7)
            plt.scatter(t, abnormal_data, s=markersize, c=markercolors)
            # plt.show()
            plt.savefig(
                './figures/{}/VisOut_OCSVM_IQR_{}_pid={}.png'.format(config.dataset, train_filename.stem, config.pid),
                dpi=300)
            plt.close()
        else:
            file_logger.info('cannot plot image with x_dim > 1')

    if config.use_spot:
        pass
    else:
        pos_label = -1
        TN, FP, FN, TP, precision, recall, f1 = {}, {}, {}, {}, {}, {}, {}
        for threshold_method in ocsvm_output.y_hat:
            cm = confusion_matrix(y_true=abnormal_label, y_pred=ocsvm_output.y_hat[threshold_method], labels=[1, -1])
            TN[threshold_method] = cm[0][0]
            FP[threshold_method] = cm[0][1]
            FN[threshold_method] = cm[1][0]
            TP[threshold_method] = cm[1][1]
            precision[threshold_method] = precision_score(y_true=abnormal_label, y_pred=ocsvm_output.y_hat[threshold_method], pos_label=pos_label)
            recall[threshold_method] = recall_score(y_true=abnormal_label, y_pred=ocsvm_output.y_hat[threshold_method], pos_label=pos_label)
            f1[threshold_method] = f1_score(y_true=abnormal_label, y_pred=ocsvm_output.y_hat[threshold_method], pos_label=pos_label)

        fpr, tpr, _ = roc_curve(y_true=abnormal_label, y_score=-ocsvm_output.decision_function, pos_label=pos_label)
        roc_auc = auc(fpr, tpr)
        pre, re, _ = precision_recall_curve(y_true=abnormal_label, probas_pred=-ocsvm_output.decision_function,
                                            pos_label=pos_label)
        pr_auc = auc(re, pre)
        metrics_result = MetricsResult(TN=TN, FP=FP, FN=FN, TP=TP, precision=precision, recall=recall, fbeta=f1, pr_auc=pr_auc, roc_auc=roc_auc)
        return metrics_result

if __name__ == '__main__':

    # %%
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, default=0)
    parser.add_argument('--x_dim', type=int, default=1)
    # Specifies the kernel type to be used in the algorithm.
    # It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable.
    # If none is given, ‘rbf’ will be used. If a callable is given it is used to precompute the kernel matrix.
    parser.add_argument('--kernel', type=str, default='rbf')
    # Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
    parser.add_argument('--degree', type=int, default=5)
    # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
    parser.add_argument('--gamma', type=str, default='auto')
    # Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
    parser.add_argument('--coef0', type=float, default=0.0)
    # Tolerance for stopping criterion.
    parser.add_argument('--tol', type=float, default=1e-4)
    # An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors.
    # Should be in the interval (0, 1]. By default 0.5 will be taken.
    parser.add_argument('--nu', type=float, default=0.1)
    parser.add_argument('--save_output', type=str2bool, default=True)
    parser.add_argument('--save_figure', type=str2bool, default=True)
    parser.add_argument('--save_model', type=str2bool, default=True)  # save model
    parser.add_argument('--save_results', type=str2bool, default=True)  # save results
    parser.add_argument('--load_model', type=str2bool, default=False)  # load model
    parser.add_argument('--use_spot', type=str2bool, default=False)
    parser.add_argument('--save_config', type=str2bool, default=True)
    parser.add_argument('--load_config', type=str2bool, default=False)
    parser.add_argument('--server_run', type=str2bool, default=False)
    parser.add_argument('--robustness', type=str2bool, default=False)
    parser.add_argument('--pid', type=int, default=0)
    args = parser.parse_args()

    for registered_dataset in ["MSL", "SMAP", "SMD", "NAB", "AIOps", "Credit", "ECG", "nyc_taxi", "SWAT", "Yahoo"]:
    #for registered_dataset in ["MSL"]:

        # the dim in args is useless, which should be deleted in the future version.
        if "noise" in registered_dataset:
            args.dataset = registered_dataset + "_{:.2f}".format(args.ratio)
        else:
            args.dataset = registered_dataset

        if args.load_config:
            config = OCSVMConfig(dataset=None, x_dim=None, kernel=None, degree=None, gamma=None, coef0=None, tol=None,
                                 nu=None, save_output=None, save_figure=None, save_model=None, load_model=None,
                                 use_spot=None, save_config=None, load_config=None, server_run=None, robustness=None,
                                 pid=None, save_results=None)
            try:
                config.import_config('./config/{}/Config_OCSVM_pid={}.json'.format(config.dataset, config.pid))
            except:
                print('There is no config.')
        else:
            config = OCSVMConfig(dataset=args.dataset, x_dim=args.x_dim, kernel=args.kernel, degree=args.degree,
                                 gamma=args.gamma, coef0=args.coef0, tol=args.tol, nu=args.nu, save_output=args.save_output,
                                 save_figure=args.save_figure, save_model=args.save_model, load_model=args.load_model,
                                 use_spot=args.use_spot, save_config=args.save_config, load_config=args.load_config,
                                 server_run=args.server_run, robustness=args.robustness, pid=args.pid, save_results=args.save_results)
        if args.save_config:
            if not os.path.exists('./config/{}/'.format(config.dataset)):
                os.makedirs('./config/{}/'.format(config.dataset))
            config.export_config('./config/{}/Config_OCSVM_pid={}.json'.format(config.dataset, config.pid))
        # %%
        if config.dataset not in dataset2path:
            raise ValueError("dataset {} is not registered.".format(config.dataset))
        else:
            train_path = dataset2path[config.dataset]["train"]
            test_path = dataset2path[config.dataset]["test"]
            label_path = dataset2path[config.dataset]["test_label"]
        # %%
        train_logger, file_logger, meta_logger = create_logger(dataset=args.dataset,
                                                               train_logger_name='ocsvm_train_logger',
                                                               file_logger_name='ocsvm_file_logger',
                                                               meta_logger_name='ocsvm_meta_logger',
                                                               model_name='OCSVM',
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

        s_TN = []
        s_FP = []
        s_FN = []
        s_TP = []
        s_precision = []
        s_recall = []
        s_fbeta = []
        s_roc_auc = []
        s_pr_auc = []
        s_cks = []
        for train_file in train_path.iterdir():
            test_file = test_path / train_file.name
            label_file = label_path / train_file.name
            file_logger.info('============================')
            file_logger.info(train_file)

            metrics_result = RunModel(train_filename=train_file, test_filename=test_file, label_filename=label_file, config=config)
            result_dataframe = make_result_dataframe(metrics_result)

            if config.save_results == True:
                if not os.path.exists('./results/{}/'.format(config.dataset)):
                    os.makedirs('./results/{}/'.format(config.dataset))
                result_dataframe.to_csv('./results/{}/Results_OCSVM_{}_pid={}.csv'.format(config.dataset, train_file.stem, config.pid),
                                        index=False)

