import argparse
import os
import matplotlib as mpl
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, roc_curve, auc, confusion_matrix
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams["font.size"] = 16
import numpy as np
from utils.config import LOFConfig
from utils.logger import create_logger
from utils.outputs import LOFOutput
from utils.utils import str2bool
from sklearn.neighbors import LocalOutlierFactor
from utils.metrics import MetricsResult

from utils.data_provider import dataset2path, read_dataset
from utils.metrics import SD_autothreshold, MAD_autothreshold, IQR_autothreshold, get_labels_by_threshold
from utils.utils import make_result_dataframe
from sklearn.metrics import f1_score


class LOF(object):
    def __init__(self, file_name, config):
        self.dataset = config.dataset
        self.file_name = file_name

        self.x_dim = config.x_dim

        self.n_neighbors = config.n_neighbors
        self.algorithm = config.algorithm
        self.leaf_size = config.leaf_size
        self.metric = config.metric
        self.p = config.p
        self.contamination = config.contamination

        self.pid = config.pid

        self.model = LocalOutlierFactor(n_neighbors=self.n_neighbors, algorithm=self.algorithm,
                                        leaf_size=self.leaf_size, metric=self.metric, p=self.p,
                                        contamination=self.contamination)

    def fit(self, train_input, train_label, test_input, test_label):
        y_pred = self.model.fit_predict(train_input)
        negative_factor = self.model.negative_outlier_factor_

        lof_output = LOFOutput(y_hat=y_pred, negative_factor=negative_factor)

        return lof_output


def RunModel(train_filename, test_filename, label_filename, config, ratio):
    negative_sample = True if "noise" in config.dataset else False
    train_data, abnormal_data, abnormal_label = read_dataset(train_filename, test_filename, label_filename, normalize=True, file_logger=file_logger, negative_sample=negative_sample, ratio=ratio)

    original_x_dim = abnormal_data.shape[1]

    config.x_dim = abnormal_data.shape[1]

    model = LOF(train_filename, config)
    lof_output = model.fit(train_input=abnormal_data, train_label=abnormal_label, test_input=abnormal_data, test_label=abnormal_label)

    SD_Tmin, SD_Tmax = SD_autothreshold(-lof_output.negative_factor)
    SD_y_hat = get_labels_by_threshold(-lof_output.negative_factor, Tmax=SD_Tmax, use_max=True, use_min=False)
    MAD_Tmin, MAD_Tmax = MAD_autothreshold(-lof_output.negative_factor)
    MAD_y_hat = get_labels_by_threshold(-lof_output.negative_factor, Tmax=MAD_Tmax, use_max=True, use_min=False)
    IQR_Tmin, IQR_Tmax = IQR_autothreshold(-lof_output.negative_factor)
    IQR_y_hat = get_labels_by_threshold(-lof_output.negative_factor, Tmax=IQR_Tmax, use_max=True, use_min=False)
    lof_output.y_hat = {}
    lof_output.y_hat["SD"] = SD_y_hat
    lof_output.y_hat["MAD"] = MAD_y_hat
    lof_output.y_hat["IQR"] = IQR_y_hat

    if config.save_output == True:
        if not os.path.exists('./outputs/NPY/{}/'.format(config.dataset)):
            os.makedirs('./outputs/NPY/{}/'.format(config.dataset))
        np.save('./outputs/NPY/{}/Score_LOF_hdim_{}_rollingsize_{}_{}_pid={}.npy'.format(config.dataset, config.n_neighbors, 1, train_filename.stem, config.pid), lof_output.negative_factor)
        np.save('./outputs/NPY/{}/Pred_LOF_hdim_{}_rollingsize_{}_{}_pid={}.npy'.format(config.dataset, config.n_neighbors, 1, train_filename.stem, config.pid), lof_output.y_hat)

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
            plt.savefig('./figures/{}/Ori_LOF_hdim_{}_rollingsize_{}_{}_pid={}.png'.format(config.dataset, config.n_neighbors, 1, train_filename.stem, config.pid), dpi=300)
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
            plt.savefig('./figures/{}/VisInp_LOF_hdim_{}_rollingsize_{}_{}_pid={}.png'.format(config.dataset, config.n_neighbors, 1, train_filename.stem, config.pid), dpi=300)
            plt.close()

            markercolors = ['blue' if i == 1 else 'red' for i in lof_output.y_hat["SD"]]
            markersize = [4 if i == 1 else 25 for i in lof_output.y_hat["SD"]]
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
            plt.savefig('./figures/{}/VisOut_LOF_hdim_{}_rollingsize_{}_SD_{}_pid={}.png'.format(config.dataset, config.n_neighbors, 1, train_filename.stem, config.pid), dpi=300)
            plt.close()

            markercolors = ['blue' if i == 1 else 'red' for i in lof_output.y_hat["MAD"]]
            markersize = [4 if i == 1 else 25 for i in lof_output.y_hat["MAD"]]
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
            plt.savefig('./figures/{}/VisOut_LOF_hdim_{}_rollingsize_{}_MAD_{}_pid={}.png'.format(config.dataset, config.n_neighbors, 1, train_filename.stem, config.pid), dpi=300)
            plt.close()

            markercolors = ['blue' if i == 1 else 'red' for i in lof_output.y_hat["IQR"]]
            markersize = [4 if i == 1 else 25 for i in lof_output.y_hat["IQR"]]
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
            plt.savefig('./figures/{}/VisOut_LOF_hdim_{}_rollingsize_{}_IQR_{}_pid={}.png'.format(config.dataset, config.n_neighbors, 1, train_filename.stem, config.pid), dpi=300)
            plt.close()
        else:
            file_logger.info('cannot plot image with x_dim > 1')

    if config.use_spot:
        pass
    else:
        pos_label = -1
        TN, FP, FN, TP, precision, recall, f1 = {}, {}, {}, {}, {}, {}, {}
        for threshold_method in lof_output.y_hat:
            cm = confusion_matrix(y_true=abnormal_label, y_pred=lof_output.y_hat[threshold_method], labels=[1, -1])
            TN[threshold_method] = cm[0][0]
            FP[threshold_method] = cm[0][1]
            FN[threshold_method] = cm[1][0]
            TP[threshold_method] = cm[1][1]
            precision[threshold_method] = precision_score(y_true=abnormal_label, y_pred=lof_output.y_hat[threshold_method], pos_label=pos_label)
            recall[threshold_method] = recall_score(y_true=abnormal_label, y_pred=lof_output.y_hat[threshold_method], pos_label=pos_label)
            f1[threshold_method] = f1_score(y_true=abnormal_label, y_pred=lof_output.y_hat[threshold_method], pos_label=pos_label)

        fpr, tpr, _ = roc_curve(y_true=abnormal_label, y_score=-lof_output.negative_factor, pos_label=pos_label)
        roc_auc = auc(fpr, tpr)
        pre, re, _ = precision_recall_curve(y_true=abnormal_label, probas_pred=-lof_output.negative_factor,
                                            pos_label=pos_label)
        pr_auc = auc(re, pre)
        metrics_result = MetricsResult(TN=TN, FP=FP, FN=FN, TP=TP, precision=precision, recall=recall, fbeta=f1, pr_auc=pr_auc, roc_auc=roc_auc)
        return metrics_result

if __name__ == '__main__':


    # %%
    parser = argparse.ArgumentParser()
    # dataset can be one of in ["AIOps", "Credit", "ECG", "MSL", "SMAP", "SMD", "NAB", "nyc_taxi", "SWAT", "Yahoo"]
    parser.add_argument('--dataset', type=str, default="AIOps")
    parser.add_argument('--x_dim', type=int, default=1)
    parser.add_argument('--n_neighbors', type=int, default=64)
    parser.add_argument('--ratio', type=float, default=0.05)
    # what is the parameter of algorithm ?
    # Algorithm used to compute the nearest neighbors:
    # ‘ball_tree’ will use BallTree
    # ‘kd_tree’ will use KDTree
    # ‘brute’ will use a brute-force search.
    # ‘auto’ will attempt to decide the most appropriate algorithm based on the values passed to fit method.
    parser.add_argument('--algorithm', type=str, default='auto')
    # what is the parameter of leaf_size ?
    # Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query,
    # as well as the memory required to store the tree. The optimal value depends on the nature of the problem.
    parser.add_argument('--leaf_size', type=int, default=50)
    # what is the parameter of metric ?
    # metric used for the distance computation.
    parser.add_argument('--metric', type=str, default='euclidean')
    # what is p ?
    # Parameter for the Minkowski metric from sklearn.metrics.pairwise.pairwise_distances.
    # When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2.
    # For arbitrary p, minkowski_distance (l_p) is used.
    parser.add_argument('--p', type=int, default=2)
    # what is the parameter of contamination ?
    # The amount of contamination of the data set, i.e. the proportion of outliers in the data set.
    parser.add_argument('--contamination', type=float, default=0.1)
    parser.add_argument('--save_output', type=str2bool, default=True)
    parser.add_argument('--save_figure', type=str2bool, default=True)
    parser.add_argument('--save_model', type=str2bool, default=True)  # save model
    parser.add_argument('--load_model', type=str2bool, default=False)  # load model
    parser.add_argument('--use_spot', type=str2bool, default=False)
    parser.add_argument('--save_config', type=str2bool, default=True)
    parser.add_argument('--save_results', type=str2bool, default=True)

    parser.add_argument('--load_config', type=str2bool, default=False)
    # what is the parameter of server_run ?
    parser.add_argument('--server_run', type=str2bool, default=False)
    parser.add_argument('--robustness', type=str2bool, default=False)
    parser.add_argument('--pid', type=int, default=0)
    args = parser.parse_args()

    #for registered_dataset in dataset2path:
    #for registered_dataset in ["Credit", "nyc_taxi"]:
    #for registered_dataset in ["NAB_noise", "MSL_noise"]:
    #for registered_dataset in ["Credit"]:
    for registered_dataset in ["MSL", "SMAP", "SMD", "NAB", "AIOps", "Credit", "ECG", "nyc_taxi", "SWAT", "Yahoo"]:

        # the dim in args is useless, which should be deleted in the future version.
        if "noise" in registered_dataset:
            args.dataset = registered_dataset + "_{:.2f}".format(args.ratio)
        else:
            args.dataset = registered_dataset

        if args.load_config:
            config = LOFConfig(dataset=None, x_dim=None, n_neighbors=None, algorithm=None, leaf_size=None, metric=None,
                               p=None, contamination=None, save_output=None, save_figure=None, save_model=None,
                               load_model=None, use_spot=None, save_config=None, load_config=None,
                               server_run=None, robustness=None, pid=None, save_results=None)
            try:
                config.import_config('./config/{}/Config_LOF_hdim_{}_rollingsize_{}_pid={}.json'.format(config.dataset, config.n_neighbors, 1, config.pid))
            except:
                print('There is no config.')
        else:
            config = LOFConfig(dataset=args.dataset, x_dim=args.x_dim, n_neighbors=args.n_neighbors,
                               algorithm=args.algorithm, leaf_size=args.leaf_size, metric=args.metric, p=args.p,
                               contamination=args.contamination, save_output=args.save_output, save_figure=args.save_figure,
                               save_model=args.save_model, load_model=args.load_model, use_spot=args.use_spot,
                               save_config=args.save_config, load_config=args.load_config, server_run=args.server_run,
                               robustness=args.robustness, pid=args.pid, save_results=args.save_results)
        if args.save_config:
            if not os.path.exists('./config/{}/'.format(config.dataset)):
                os.makedirs('./config/{}/'.format(config.dataset))
            config.export_config('./config/{}/Config_LOF_hdim_{}_rollingsize_{}_pid={}.json'.format(config.dataset, config.n_neighbors, 1, config.pid))
        # %%
        if config.dataset not in dataset2path:
            raise ValueError("dataset {} is not registered.".format(config.dataset))
        else:
            train_path = dataset2path[config.dataset]["train"]
            test_path = dataset2path[config.dataset]["test"]
            label_path = dataset2path[config.dataset]["test_label"]

        train_logger, file_logger, meta_logger = create_logger(dataset=args.dataset,
                                                               train_logger_name='lof_train_logger',
                                                               file_logger_name='lof_file_logger',
                                                               meta_logger_name='lof_meta_logger',
                                                               model_name='LOF',
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
                result_dataframe.to_csv('./results/{}/Results_LOF_hdim_{}_rollingsize_{}_{}_pid={}.csv'.format(config.dataset, config.n_neighbors, 1, train_file.stem, config.pid),
                                        index=False)
