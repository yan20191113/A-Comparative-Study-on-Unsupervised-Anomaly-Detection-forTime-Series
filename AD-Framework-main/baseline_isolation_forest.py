import argparse
import os
import matplotlib as mpl
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, roc_curve, auc, confusion_matrix
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from utils.config import ISFConfig
from utils.logger import create_logger
from utils.outputs import ISFOutput
from utils.utils import str2bool
from sklearn.ensemble import IsolationForest
from utils.metrics import MetricsResult

from utils.data_provider import dataset2path, read_dataset
from utils.metrics import SD_autothreshold, MAD_autothreshold, IQR_autothreshold, get_labels_by_threshold
from utils.utils import make_result_dataframe
from sklearn.metrics import f1_score


class ISF(object):
    def __init__(self, file_name, config):
        self.dataset = config.dataset
        self.file_name = file_name

        self.x_dim = config.x_dim

        self.n_estimators = config.n_estimators
        self.max_samples = config.max_samples
        self.bootstrap = config.bootstrap
        self.max_features = config.max_features
        self.contamination = config.contamination

        self.pid = config.pid

        self.model = IsolationForest(n_estimators=self.n_estimators, max_samples=self.max_samples,
                                     bootstrap=self.bootstrap, max_features=self.max_features,
                                     contamination=self.contamination)

    def fit(self, train_input, train_label, test_input, test_label):
        y_pred = self.model.fit_predict(train_input)
        decision_function = self.model.decision_function(train_input)

        isf_output = ISFOutput(y_hat=y_pred, decision_function=decision_function)

        return isf_output


def RunModel(train_filename, test_filename, label_filename, config, ratio):
    negative_sample = True if "noise" in config.dataset else False
    train_data, abnormal_data, abnormal_label = read_dataset(train_filename, test_filename, label_filename,
                                                             normalize=True, file_logger=file_logger, negative_sample=negative_sample, ratio=ratio)

    original_x_dim = abnormal_data.shape[1]

    config.x_dim = abnormal_data.shape[1]

    model = ISF(train_filename, config)
    isf_output = model.fit(train_input=abnormal_data, train_label=abnormal_label, test_input=abnormal_data, test_label=abnormal_label)

    # We only use the max value to threshold the outlier scores
    SD_Tmin, SD_Tmax = SD_autothreshold(-isf_output.decision_function)
    SD_y_hat = get_labels_by_threshold(-isf_output.decision_function, Tmax=SD_Tmax, use_max=True, use_min=False)
    MAD_Tmin, MAD_Tmax = MAD_autothreshold(-isf_output.decision_function)
    MAD_y_hat = get_labels_by_threshold(-isf_output.decision_function, Tmax=MAD_Tmax, use_max=True, use_min=False)
    IQR_Tmin, IQR_Tmax = IQR_autothreshold(-isf_output.decision_function)
    IQR_y_hat = get_labels_by_threshold(-isf_output.decision_function, Tmax=IQR_Tmax, use_max=True, use_min=False)
    isf_output.y_hat = {}
    isf_output.y_hat["SD"] = SD_y_hat
    isf_output.y_hat["MAD"] = MAD_y_hat
    isf_output.y_hat["IQR"] = IQR_y_hat

    if config.save_output == True:
        if not os.path.exists('./outputs/NPY/{}/'.format(config.dataset)):
            os.makedirs('./outputs/NPY/{}/'.format(config.dataset))
        np.save('./outputs/NPY/{}/Score_ISF_hdim_{}_rollingsize_{}_{}_pid={}.npy'.format(config.dataset, config.n_estimators, 1, train_filename.stem, config.pid), isf_output.decision_function)
        np.save('./outputs/NPY/{}/Pred_ISF_hdim_{}_rollingsize_{}_{}_pid={}.npy'.format(config.dataset, config.n_estimators, 1, train_filename.stem, config.pid), isf_output.y_hat)

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
            plt.savefig('./figures/{}/Ori_ISF_hdim_{}_rollingsize_{}_{}_pid={}.png'.format(config.dataset, config.n_estimators, 1, train_filename.stem, config.pid), dpi=600)
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
            plt.savefig('./figures/{}/VisInp_ISF_hdim_{}_rollingsize_{}_{}_pid={}.png'.format(config.dataset, config.n_estimators, 1, train_filename.stem, config.pid), dpi=600)
            plt.close()

            markercolors = ['blue' if i == 1 else 'red' for i in isf_output.y_hat["SD"]]
            markersize = [4 if i == 1 else 25 for i in isf_output.y_hat["SD"]]
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
            plt.savefig('./figures/{}/VisOut_ISF_hdim_{}_rollingsize_{}_SD_{}_pid={}.png'.format(config.dataset, config.n_estimators, 1, train_filename.stem, config.pid), dpi=600)
            plt.close()

            markercolors = ['blue' if i == 1 else 'red' for i in isf_output.y_hat["MAD"]]
            markersize = [4 if i == 1 else 25 for i in isf_output.y_hat["MAD"]]
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
            plt.savefig('./figures/{}/VisOut_ISF_hdim_{}_rollingsize_{}_MAD_{}_pid={}.png'.format(config.dataset, config.n_estimators, 1, train_filename.stem, config.pid), dpi=600)
            plt.close()

            markercolors = ['blue' if i == 1 else 'red' for i in isf_output.y_hat["IQR"]]
            markersize = [4 if i == 1 else 25 for i in isf_output.y_hat["IQR"]]
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
            plt.savefig('./figures/{}/VisOut_ISF_hdim_{}_rollingsize_{}_IQR_{}_pid={}.png'.format(config.dataset, config.n_estimators, 1, train_filename.stem, config.pid), dpi=600)
            plt.close()
        else:
            file_logger.info('cannot plot image with x_dim > 1')

    if config.use_spot:
        pass
    else:
        pos_label = -1
        TN, FP, FN, TP, precision, recall, f1 = {}, {}, {}, {}, {}, {}, {}
        for threshold_method in isf_output.y_hat:
            cm = confusion_matrix(y_true=abnormal_label, y_pred=isf_output.y_hat[threshold_method], labels=[1, -1])
            TN[threshold_method] = cm[0][0]
            FP[threshold_method] = cm[0][1]
            FN[threshold_method] = cm[1][0]
            TP[threshold_method] = cm[1][1]
            precision[threshold_method] = precision_score(y_true=abnormal_label,
                                                          y_pred=isf_output.y_hat[threshold_method],
                                                          pos_label=pos_label)
            recall[threshold_method] = recall_score(y_true=abnormal_label, y_pred=isf_output.y_hat[threshold_method],
                                                    pos_label=pos_label)
            f1[threshold_method] = f1_score(y_true=abnormal_label, y_pred=isf_output.y_hat[threshold_method],
                                            pos_label=pos_label)

        fpr, tpr, _ = roc_curve(y_true=abnormal_label, y_score=-isf_output.decision_function, pos_label=pos_label)
        roc_auc = auc(fpr, tpr)
        pre, re, _ = precision_recall_curve(y_true=abnormal_label, probas_pred=-isf_output.decision_function,
                                            pos_label=pos_label)
        pr_auc = auc(re, pre)
        metrics_result = MetricsResult(TN=TN, FP=FP, FN=FN, TP=TP, precision=precision, recall=recall, fbeta=f1,
                                       pr_auc=pr_auc, roc_auc=roc_auc)
        return metrics_result

if __name__ == '__main__':

    # %%
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, default=0)
    parser.add_argument('--x_dim', type=int, default=1)
    parser.add_argument('--ratio', type=float, default=0.05)
    # The number of base estimators in the ensemble.
    parser.add_argument('--n_estimators', type=int, default=128)
    # max_samples“auto”, int or float, default=”auto”
    # The number of samples to draw from X to train each base estimator.
    # If int, then draw max_samples samples.
    # If float, then draw max_samples * X.shape[0] samples.
    # If “auto”, then max_samples=min(256, n_samples).
    parser.add_argument('--max_samples', type=str, default='auto')
    # The amount of contamination of the data set, i.e. the proportion of outliers in the data set.
    parser.add_argument('--contamination', type=float, default=0.1)
    # If True, individual trees are fit on random subsets of the training data sampled with replacement.
    # If False, sampling without replacement is performed.
    parser.add_argument('--bootstrap', type=str2bool, default=False)
    # The number of features to draw from X to train each base estimator.
    parser.add_argument('--max_features', type=float, default=1.0)
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

    #for registered_dataset in dataset2path:
    #for registered_dataset in ["nyc_taxi"]:
    for registered_dataset in ["MSL", "SMAP", "SMD", "NAB", "AIOps", "Credit", "ECG", "nyc_taxi", "SWAT", "Yahoo"]:

        # the dim in args is useless, which should be deleted in the future version.
        if "noise" in registered_dataset:
            args.dataset = registered_dataset + "_{:.2f}".format(args.ratio)
        else:
            args.dataset = registered_dataset

        if args.load_config:
            config = ISFConfig(dataset=None, x_dim=None, n_estimators=None, max_samples=None, contamination=None,
                               bootstrap=None, max_features=None, save_output=None, save_figure=None, save_model=None,
                               load_model=None, use_spot=None, save_config=None, load_config=None, server_run=None,
                               robustness=None, pid=None, save_results=args.save_results)
            try:
                config.import_config('./config/{}/Config_ISF_hdim_{}_rollingsize_{}_pid={}.json'.format(config.dataset, config.n_estimators, 1, config.pid))
            except:
                print('There is no config.')
        else:
            config = ISFConfig(dataset=args.dataset, x_dim=args.x_dim, n_estimators=args.n_estimators,
                               max_samples=args.max_samples, contamination=args.contamination, bootstrap=args.bootstrap,
                               max_features=args.max_features, save_output=args.save_output, save_figure=args.save_figure,
                               save_model=args.save_model, load_model=args.load_model, use_spot=args.use_spot,
                               save_config=args.save_config, load_config=args.load_config, server_run=args.server_run,
                               robustness=args.robustness, pid=args.pid, save_results=args.save_results)
        if args.save_config:
            if not os.path.exists('./config/{}/'.format(config.dataset)):
                os.makedirs('./config/{}/'.format(config.dataset))
            config.export_config('./config/{}/Config_ISF_hdim_{}_rollingsize_{}_pid={}.json'.format(config.dataset, config.n_estimators, 1, config.pid))
        # %%
        if config.dataset not in dataset2path:
            raise ValueError("dataset {} is not registered.".format(config.dataset))
        else:
            train_path = dataset2path[config.dataset]["train"]
            test_path = dataset2path[config.dataset]["test"]
            label_path = dataset2path[config.dataset]["test_label"]
        # %%
        train_logger, file_logger, meta_logger = create_logger(dataset=args.dataset,
                                                               train_logger_name='isf_train_logger',
                                                               file_logger_name='isf_file_logger',
                                                               meta_logger_name='isf_meta_logger',
                                                               model_name='ISF',
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
            test_file = test_path / train_file.name
            label_file = label_path / train_file.name
            file_logger.info('============================')
            file_logger.info(train_file)

            metrics_result = RunModel(train_filename=train_file, test_filename=test_file, label_filename=label_file, config=config, ratio=args.ratio)
            result_dataframe = make_result_dataframe(metrics_result)

            if config.save_results == True:
                if not os.path.exists('./results/{}/'.format(config.dataset)):
                    os.makedirs('./results/{}/'.format(config.dataset))
                result_dataframe.to_csv('./results/{}/Results_isf_hdim_{}_rollingsize_{}_{}_pid={}.csv'.format(config.dataset, config.n_estimators, 1, train_file.stem, config.pid),
                                        index=False)



