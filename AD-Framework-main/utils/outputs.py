class VRAEOutput(object):
    def __init__(self, best_TN, best_FP, best_FN, best_TP, best_precision, best_recall, best_fbeta,  best_pr_auc,
                 best_roc_auc, best_cks, zs=None, z_infer_means=None, z_infer_stds=None, decs=None, dec_means=None,
                 dec_stds=None, kld_loss=None, nll_loss=None, min_valid_loss=None):
        self.zs = zs
        self.z_infer_means = z_infer_means
        self.z_infer_stds = z_infer_stds
        self.decs = decs
        self.dec_means = dec_means
        self.dec_stds = dec_stds
        self.kld_loss = kld_loss
        self.nll_loss = nll_loss
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


class VAEOutput(object):
    def __init__(self, best_TN, best_FP, best_FN, best_TP, best_precision, best_recall, best_fbeta,  best_pr_auc,
                 best_roc_auc, best_cks, zs=None, z_infer_means=None, z_infer_stds=None, decs=None, dec_means=None,
                 dec_stds=None, kld_loss=None, nll_loss=None, min_valid_loss=None, training_time=None, testing_time=None,
                 memory_usage_nvidia=None):
        self.zs = zs
        self.z_infer_means = z_infer_means
        self.z_infer_stds = z_infer_stds
        self.decs = decs
        self.dec_means = dec_means
        self.dec_stds = dec_stds
        self.kld_loss = kld_loss
        self.nll_loss = nll_loss
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
        self.memory_usage_nvidia = memory_usage_nvidia


class BEATGANOutput(object):
    def __init__(self, dec_means, best_TN, best_FP, best_FN, best_TP, best_precision, best_recall, best_fbeta,
                 best_pr_auc, best_roc_auc, best_cks, min_valid_loss, training_time=None, testing_time=None,
                 memory_usage_nvidia=None):
        self.dec_means = dec_means
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
        self.memory_usage_nvidia = memory_usage_nvidia


class LOFOutput(object):
    def __init__(self, y_hat=None, negative_factor=None, training_time=None, testing_time=None):
        self.y_hat = y_hat
        self.negative_factor = negative_factor
        self.training_time = training_time
        self.testing_time = testing_time


class ISFOutput(object):
    def __init__(self, y_hat=None, decision_function=None, scores=None, training_time=None, testing_time=None):
        self.y_hat = y_hat
        self.decision_function = decision_function
        self.scores = scores
        self.training_time = training_time
        self.testing_time = testing_time


class OCSVMOutput(object):
    def __init__(self, y_hat=None, decision_function=None, scores=None, training_time=None, testing_time=None):
        self.y_hat = y_hat
        self.decision_function = decision_function
        self.scores = scores
        self.training_time = training_time
        self.testing_time = testing_time


class AEOutput(object):
    def __init__(self, dec_means, best_TN, best_FP, best_FN, best_TP, best_precision, best_recall, best_fbeta,
                 best_pr_auc, best_roc_auc, best_cks, min_valid_loss, training_time, testing_time, memory_usage_nvidia):
        self.dec_means = dec_means
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
        self.memory_usage_nvidia = memory_usage_nvidia


class RAEOutput(object):
    def __init__(self, dec_means, best_TN, best_FP, best_FN, best_TP, best_precision, best_recall, best_fbeta,
                 best_pr_auc, best_roc_auc, best_cks, min_valid_loss, training_time, testing_time, memory_usage_nvidia):
        self.dec_means = dec_means
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
        self.memory_usage_nvidia = memory_usage_nvidia


class LSTMNDTOutput(object):
    def __init__(self, dec_means, best_TN, best_FP, best_FN, best_TP, best_precision, best_recall, best_fbeta,
                 best_pr_auc, best_roc_auc, best_cks, min_valid_loss, training_time, testing_time, memory_usage_nvidia):
        self.dec_means = dec_means
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
        self.memory_usage_nvidia = memory_usage_nvidia


class HIFIOutput(object):
    def __init__(self, dec_means, best_TN, best_FP, best_FN, best_TP, best_precision, best_recall, best_fbeta,
                 best_pr_auc, best_roc_auc, best_cks, min_valid_loss, training_time=None, testing_time=None, memory_usage_nvidia=None):
        self.dec_means = dec_means
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
        self.memory_usage_nvidia = memory_usage_nvidia


class CAEOutput(object):
    def __init__(self, dec_means, best_TN, best_FP, best_FN, best_TP, best_precision, best_recall, best_fbeta,
                 best_pr_auc, best_roc_auc, best_cks, min_valid_loss, training_time, testing_time, memory_usage_nvidia):
        self.dec_means = dec_means
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
        self.memory_usage_nvidia = memory_usage_nvidia

