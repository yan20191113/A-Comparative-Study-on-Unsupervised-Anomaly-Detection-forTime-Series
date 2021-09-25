import json


class AEConfig(object):
    def __init__(self, dataset, x_dim, h_dim, ensemble_space, preprocessing, use_overlapping, rolling_size, epochs,
                 milestone_epochs, lr, gamma, batch_size, weight_decay, early_stopping, loss_function, display_epoch,
                 save_output, save_figure, save_model, load_model, continue_training, dropout, use_spot,
                 use_last_point, save_config, load_config, server_run, robustness, pid, save_results):
        self.dataset = dataset
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.ensemble_space = ensemble_space
        self.preprocessing = preprocessing
        self.use_overlapping = use_overlapping
        self.rolling_size = rolling_size
        self.epochs = epochs
        self.milestone_epochs = milestone_epochs
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.early_stopping = early_stopping
        self.loss_function = loss_function
        self.display_epoch = display_epoch
        self.save_output = save_output
        self.save_figure = save_figure
        self.save_model = save_model
        self.load_model = load_model
        self.continue_training = continue_training
        self.dropout = dropout
        self.use_spot = use_spot
        self.use_last_point = use_last_point
        self.save_config = save_config
        self.load_config = load_config
        self.server_run = server_run
        self.robustness = robustness
        self.pid = pid
        self.save_results = save_results

    def import_config(self, import_json):
        """Load settings dict from import_json (path/filename.json) JSON-file."""

        with open(import_json, 'r') as fp:
            settings = json.load(fp)

        for key, value in settings.items():
            setattr(self, key, value)

    def export_config(self, export_json):
        """Save settings dict to export_json (path/filename.json) JSON-file."""

        with open(export_json, 'w+') as fp:
            json.dump(vars(self), fp)

    def to_string(self):
        string = ''
        for key, value in vars(self).items():
            string = string + key + ' = ' + str(value) + ', '
        return string


class RAEEnsembleConfig:
    def __init__(self, dataset, x_dim, h_dim, preprocessing, use_overlapping, rolling_size, epochs, milestone_epochs,
                 lr, gamma, batch_size, weight_decay, early_stopping, loss_function, rnn_layers, use_clip_norm, gradient_clip_norm,
                 use_bidirection, force_teaching, force_teaching_threshold, display_epoch, save_output, save_figure, save_model, load_model,
                 continue_training, dropout, use_spot, use_last_point, save_config, load_config, server_run, robustness, adjusted_points,
                 use_best_f1, pid, ensemble_members, save_results):

        self.dataset = dataset
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.preprocessing = preprocessing
        self.use_overlapping = use_overlapping
        self.rolling_size = rolling_size
        self.epochs = epochs
        self.milestone_epochs = milestone_epochs
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.early_stopping = early_stopping
        self.loss_function = loss_function
        self.use_clip_norm = use_clip_norm
        self.gradient_clip_norm = gradient_clip_norm
        self.rnn_layers = rnn_layers
        self.use_bidirection = use_bidirection
        self.force_teaching = force_teaching
        self.force_teaching_threshold = force_teaching_threshold
        self.display_epoch = display_epoch
        self.save_output = save_output
        self.save_figure = save_figure
        self.save_model = save_model
        self.load_model = load_model
        self.continue_training = continue_training
        self.dropout = dropout
        self.use_spot = use_spot
        self.use_last_point = use_last_point
        self.save_config = save_config
        self.load_config = load_config
        self.server_run = server_run
        self.robustness = robustness
        self.pid = pid
        self.adjusted_points = adjusted_points
        self.use_best_f1 = use_best_f1
        self.ensemble_members = ensemble_members
        self.save_results = save_results

    def import_config(self, import_json):
        """Load settings dict from import_json (path/filename.json) JSON-file."""

        with open(import_json, 'r') as fp:
            settings = json.load(fp)

        for key, value in settings.items():
            setattr(self, key, value)

    def export_config(self, export_json):
        """Save settings dict to export_json (path/filename.json) JSON-file."""

        with open(export_json, 'w+') as fp:
            json.dump(vars(self), fp)

    def to_string(self):
        string = ''
        for key, value in vars(self).items():
            string = string + key + ' = ' + str(value) + ', '
        return string

class RAEConfig(object):
    def __init__(self, dataset, x_dim, h_dim, preprocessing, use_overlapping, rolling_size, epochs, milestone_epochs,
                 lr, gamma, batch_size, weight_decay, early_stopping, loss_function, use_clip_norm, gradient_clip_norm,
                 rnn_layers, use_bidirection, force_teaching, force_teaching_threshold, display_epoch, save_output,
                 save_figure, save_model, load_model, continue_training, dropout, use_spot, use_last_point, save_config,
                 load_config, server_run, robustness, pid, save_results):

        self.dataset = dataset
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.preprocessing = preprocessing
        self.use_overlapping = use_overlapping
        self.rolling_size = rolling_size
        self.epochs = epochs
        self.milestone_epochs = milestone_epochs
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.early_stopping = early_stopping
        self.loss_function = loss_function
        self.use_clip_norm = use_clip_norm
        self.gradient_clip_norm = gradient_clip_norm
        self.rnn_layers = rnn_layers
        self.use_bidirection = use_bidirection
        self.force_teaching = force_teaching
        self.force_teaching_threshold = force_teaching_threshold
        self.display_epoch = display_epoch
        self.save_output = save_output
        self.save_figure = save_figure
        self.save_model = save_model
        self.load_model = load_model
        self.continue_training = continue_training
        self.dropout = dropout
        self.use_spot = use_spot
        self.use_last_point = use_last_point
        self.save_config = save_config
        self.load_config = load_config
        self.server_run = server_run
        self.robustness = robustness
        self.pid = pid
        self.save_results = save_results

    def import_config(self, import_json):
        """Load settings dict from import_json (path/filename.json) JSON-file."""

        with open(import_json, 'r') as fp:
            settings = json.load(fp)

        for key, value in settings.items():
            setattr(self, key, value)

    def export_config(self, export_json):
        """Save settings dict to export_json (path/filename.json) JSON-file."""

        with open(export_json, 'w+') as fp:
            json.dump(vars(self), fp)

    def to_string(self):
        string = ''
        for key, value in vars(self).items():
            string = string + key + ' = ' + str(value) + ', '
        return string


class RNNVAEConfig(object):
    def __init__(self, dataset, x_dim, h_dim, z_dim, preprocessing, use_overlapping, rolling_size, epochs,
                 milestone_epochs, lr, gamma, batch_size, weight_decay, early_stopping, loss_function, use_clip_norm,
                 gradient_clip_norm, rnn_layers, lmbda, use_bidirection, force_teaching, force_teaching_threshold,
                 display_epoch, save_output, save_figure, save_model, load_model, continue_training, dropout, use_spot,
                 use_last_point, save_config, load_config, server_run, robustness, pid):

        self.dataset = dataset
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.preprocessing = preprocessing
        self.use_overlapping = use_overlapping
        self.rolling_size = rolling_size
        self.epochs = epochs
        self.milestone_epochs = milestone_epochs
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.early_stopping = early_stopping
        self.loss_function = loss_function
        self.use_clip_norm = use_clip_norm
        self.gradient_clip_norm = gradient_clip_norm
        self.rnn_layers = rnn_layers
        self.lmbda = lmbda
        self.use_bidirection = use_bidirection
        self.force_teaching = force_teaching
        self.force_teaching_threshold = force_teaching_threshold
        self.display_epoch = display_epoch
        self.save_output = save_output
        self.save_figure = save_figure
        self.save_model = save_model
        self.load_model = load_model
        self.continue_training = continue_training
        self.dropout = dropout
        self.use_spot = use_spot
        self.use_last_point = use_last_point
        self.save_config = save_config
        self.load_config = load_config
        self.server_run = server_run
        self.robustness = robustness
        self.pid = pid

    def import_config(self, import_json):
        """Load settings dict from import_json (path/filename.json) JSON-file."""

        with open(import_json, 'r') as fp:
            settings = json.load(fp)

        for key, value in settings.items():
            setattr(self, key, value)

    def export_config(self, export_json):
        """Save settings dict to export_json (path/filename.json) JSON-file."""

        with open(export_json, 'w+') as fp:
            json.dump(vars(self), fp)

    def to_string(self):
        string = ''
        for key, value in vars(self).items():
            string = string + key + ' = ' + str(value) + ', '
        return string


class CAEConfig(object):
    def __init__(self, dataset, x_dim, h_dim, preprocessing, use_overlapping, rolling_size, epochs, milestone_epochs,
                 lr, gamma, batch_size, weight_decay, early_stopping, loss_function, display_epoch, save_output,
                 save_figure, save_model, load_model, continue_training, dropout, use_spot, use_last_point, save_config,
                 load_config, server_run, robustness, pid, save_results):

        self.dataset = dataset
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.preprocessing = preprocessing
        self.use_overlapping = use_overlapping
        self.rolling_size = rolling_size
        self.epochs = epochs
        self.milestone_epochs = milestone_epochs
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.early_stopping = early_stopping
        self.loss_function = loss_function
        self.display_epoch = display_epoch
        self.save_output = save_output
        self.save_figure = save_figure
        self.save_model = save_model
        self.load_model = load_model
        self.continue_training = continue_training
        self.dropout = dropout
        self.use_spot = use_spot
        self.use_last_point = use_last_point
        self.save_config = save_config
        self.load_config = load_config
        self.server_run = server_run
        self.robustness = robustness
        self.pid = pid
        self.save_results = save_results

    def import_config(self, import_json):
        """Load settings dict from import_json (path/filename.json) JSON-file."""

        with open(import_json, 'r') as fp:
            settings = json.load(fp)

        for key, value in settings.items():
            setattr(self, key, value)

    def export_config(self, export_json):
        """Save settings dict to export_json (path/filename.json) JSON-file."""

        with open(export_json, 'w+') as fp:
            json.dump(vars(self), fp)

    def to_string(self):
        string = ''
        for key, value in vars(self).items():
            string = string + key + ' = ' + str(value) + ', '
        return string


class CBHConfig(object):
    def __init__(self, dataset, x_dim, h_dim, k, preprocessing, use_overlapping, rolling_size, epochs,
                 milestone_epochs, lr, gamma, batch_size, weight_decay, early_stopping, loss_function, display_epoch,
                 save_output, save_figure, save_model, load_model, continue_training, dropout, use_spot, use_last_point,
                 save_config, load_config, server_run, robustness, pid):

        self.dataset = dataset
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.k = k
        self.preprocessing = preprocessing
        self.use_overlapping = use_overlapping
        self.rolling_size = rolling_size
        self.epochs = epochs
        self.milestone_epochs = milestone_epochs
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.early_stopping = early_stopping
        self.loss_function = loss_function
        self.display_epoch = display_epoch
        self.save_output = save_output
        self.save_figure = save_figure
        self.save_model = save_model
        self.load_model = load_model
        self.continue_training = continue_training
        self.dropout = dropout
        self.use_spot = use_spot
        self.use_last_point = use_last_point
        self.save_config = save_config
        self.load_config = load_config
        self.server_run = server_run
        self.robustness = robustness
        self.pid = pid

    def import_config(self, import_json):
        """Load settings dict from import_json (path/filename.json) JSON-file."""

        with open(import_json, 'r') as fp:
            settings = json.load(fp)

        for key, value in settings.items():
            setattr(self, key, value)

    def export_config(self, export_json):
        """Save settings dict to export_json (path/filename.json) JSON-file."""

        with open(export_json, 'w+') as fp:
            json.dump(vars(self), fp)

    def to_string(self):
        string = ''
        for key, value in vars(self).items():
            string = string + key + ' = ' + str(value) + ', '
        return string



class OCConfig(object):
    def __init__(self, dataset, x_dim, embedding_dim, w_dim, V_dim, cell_type, preprocessing, encoding_type, epochs,
                 ae_epochs, milestone_epochs, ae_milestone_epochs, lr, ae_lr, gamma, ae_gamma, batch_size,
                 ae_batch_size, weight_decay, ae_weight_decay, early_stopping, ae_early_stopping, loss_function,
                 use_overlapping, rolling_size, display_epoch, save_output, save_figure, save_model, ae_save_model,
                 load_model, ae_load_model, continue_training, dropout, ae_dropout, nu, use_spot, save_config,
                 load_config, server_run, robustness, pid):

        self.dataset = dataset
        self.x_dim = x_dim
        self.embedding_dim = embedding_dim
        self.w_dim = w_dim
        self.V_dim = V_dim
        self.cell_type = cell_type
        self.preprocessing = preprocessing
        self.encoding_type = encoding_type
        self.epochs = epochs
        self.ae_epochs = ae_epochs
        self.milestone_epochs = milestone_epochs
        self.ae_milestone_epochs = ae_milestone_epochs
        self.lr = lr
        self.ae_lr = ae_lr
        self.gamma = gamma
        self.ae_gamma = ae_gamma
        self.batch_size = batch_size
        self.ae_batch_size = ae_batch_size
        self.weight_decay = weight_decay
        self.ae_weight_decay = ae_weight_decay
        self.early_stopping = early_stopping
        self.ae_early_stopping = ae_early_stopping
        self.loss_function = loss_function
        self.use_overlapping = use_overlapping
        self.rolling_size = rolling_size
        self.display_epoch = display_epoch
        self.save_output = save_output
        self.save_figure = save_figure
        self.save_model = save_model
        self.ae_save_model = ae_save_model
        self.load_model = load_model
        self.ae_load_model = ae_load_model
        self.continue_training = continue_training
        self.dropout = dropout
        self.ae_dropout = ae_dropout
        self.nu = nu
        self.use_spot = use_spot
        self.save_config = save_config
        self.load_config = load_config
        self.server_run = server_run
        self.robustness = robustness
        self.pid = pid

    def import_config(self, import_json):
        """Load settings dict from import_json (path/filename.json) JSON-file."""

        with open(import_json, 'r') as fp:
            settings = json.load(fp)

        for key, value in settings.items():
            setattr(self, key, value)

    def export_config(self, export_json):
        """Save settings dict to export_json (path/filename.json) JSON-file."""

        with open(export_json, 'w+') as fp:
            json.dump(vars(self), fp)

    def to_string(self):
        string = ''
        for key, value in vars(self).items():
            string = string + key + ' = ' + str(value) + ', '
        return string


class VRAEConfig(object):
    def __init__(self, dataset, x_dim, h_dim, z_dim, preprocessing, use_overlapping, rolling_size, epochs,
                 milestone_epochs, lr, gamma, batch_size, weight_decay, early_stopping, loss_function, lmbda,
                 use_clip_norm, gradient_clip_norm, rnn_layers, use_PNF, PNF_layers, use_bidirection, use_seq2seq,
                 force_teaching, force_teaching_threshold, flexible_h, alpha, beta, display_epoch, save_output,
                 save_figure, save_model, load_model, continue_training, dropout, use_spot, use_last_point,
                 save_config, load_config, server_run, robustness, pid, save_results):

        self.dataset = dataset
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.preprocessing = preprocessing
        self.use_overlapping = use_overlapping
        self.rolling_size = rolling_size
        self.epochs = epochs
        self.milestone_epochs = milestone_epochs
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.early_stopping = early_stopping
        self.loss_function = loss_function
        self.lmbda = lmbda
        self.use_clip_norm = use_clip_norm
        self.gradient_clip_norm = gradient_clip_norm
        self.rnn_layers = rnn_layers
        self.use_PNF = use_PNF
        self.PNF_layers = PNF_layers
        self.use_bidirection = use_bidirection
        self.use_seq2seq = use_seq2seq
        self.force_teaching = force_teaching
        self.force_teaching_threshold = force_teaching_threshold
        self.flexible_h = flexible_h
        self.alpha = alpha
        self.beta = beta
        self.display_epoch = display_epoch
        self.save_output = save_output
        self.save_figure = save_figure
        self.save_model = save_model
        self.save_results = save_results
        self.load_model = load_model
        self.continue_training = continue_training
        self.dropout = dropout
        self.use_spot = use_spot
        self.use_last_point = use_last_point
        self.save_config = save_config
        self.load_config = load_config
        self.server_run = server_run
        self.robustness = robustness
        self.pid = pid

    def import_config(self, import_json):
        """Load settings dict from import_json (path/filename.json) JSON-file."""

        with open(import_json, 'r') as fp:
            settings = json.load(fp)

        for key, value in settings.items():
            setattr(self, key, value)

    def export_config(self, export_json):
        """Save settings dict to export_json (path/filename.json) JSON-file."""

        with open(export_json, 'w+') as fp:
            json.dump(vars(self), fp)

    def to_string(self):
        string = ''
        for key, value in vars(self).items():
            string = string + key + ' = ' + str(value) + ', '
        return string


class VQRAEConfig(object):
    def __init__(self, dataset, x_dim, h_dim, z_dim, adversarial_training, preprocessing, use_overlapping, rolling_size,
                 epochs, milestone_epochs, lr, gamma, batch_size, weight_decay, early_stopping, loss_function, lmbda,
                 use_clip_norm, gradient_clip_norm, rnn_layers, use_PNF, PNF_layers, use_bidirection, use_seq2seq,
                 force_teaching, force_teaching_threshold, flexible_h, alpha, beta, display_epoch, save_output,
                 save_figure, save_model, load_model, continue_training, dropout, use_spot, use_last_point,
                 save_config, load_config, server_run, robustness, pid):

        self.dataset = dataset
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.adversarial_training = adversarial_training
        self.preprocessing = preprocessing
        self.use_overlapping = use_overlapping
        self.rolling_size = rolling_size
        self.epochs = epochs
        self.milestone_epochs = milestone_epochs
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.early_stopping = early_stopping
        self.loss_function = loss_function
        self.lmbda = lmbda
        self.use_clip_norm = use_clip_norm
        self.gradient_clip_norm = gradient_clip_norm
        self.rnn_layers = rnn_layers
        self.use_PNF = use_PNF
        self.PNF_layers = PNF_layers
        self.use_bidirection = use_bidirection
        self.use_seq2seq = use_seq2seq
        self.force_teaching = force_teaching
        self.force_teaching_threshold = force_teaching_threshold
        self.flexible_h = flexible_h
        self.alpha = alpha
        self.beta = beta
        self.display_epoch = display_epoch
        self.save_output = save_output
        self.save_figure = save_figure
        self.save_model = save_model
        self.load_model = load_model
        self.continue_training = continue_training
        self.dropout = dropout
        self.use_spot = use_spot
        self.use_last_point = use_last_point
        self.save_config = save_config
        self.load_config = load_config
        self.server_run = server_run
        self.robustness = robustness
        self.pid = pid

    def import_config(self, import_json):
        """Load settings dict from import_json (path/filename.json) JSON-file."""

        with open(import_json, 'r') as fp:
            settings = json.load(fp)

        for key, value in settings.items():
            setattr(self, key, value)

    def export_config(self, export_json):
        """Save settings dict to export_json (path/filename.json) JSON-file."""

        with open(export_json, 'w+') as fp:
            json.dump(vars(self), fp)

    def to_string(self):
        string = ''
        for key, value in vars(self).items():
            string = string + key + ' = ' + str(value) + ', '
        return string


class DAGMMConfig(object):
    def __init__(self, dataset, x_dim, h_dim, z_dim, n_gmm, preprocessing, use_overlapping, rolling_size, epochs,
                 milestone_epochs, lr, gamma, batch_size, weight_decay, early_stopping, use_clip_norm,
                 gradient_clip_norm, lambda_energy, lambda_cov, display_epoch, save_output, save_figure, save_model,
                 load_model, continue_training, dropout, use_spot, use_last_point, save_config, load_config,
                 server_run, robustness, pid):

        self.dataset = dataset
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_gmm = n_gmm
        self.preprocessing = preprocessing
        self.use_overlapping = use_overlapping
        self.rolling_size = rolling_size
        self.epochs = epochs
        self.milestone_epochs = milestone_epochs
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.early_stopping = early_stopping
        self.use_clip_norm = use_clip_norm
        self.gradient_clip_norm = gradient_clip_norm
        self.lambda_energy = lambda_energy
        self.lambda_cov = lambda_cov
        self.display_epoch = display_epoch
        self.save_output = save_output
        self.save_figure = save_figure
        self.save_model = save_model
        self.load_model = load_model
        self.continue_training = continue_training
        self.dropout = dropout
        self.use_spot = use_spot
        self.use_last_point = use_last_point
        self.save_config = save_config
        self.load_config = load_config
        self.server_run = server_run
        self.robustness = robustness
        self.pid = pid

    def import_config(self, import_json):
        """Load settings dict from import_json (path/filename.json) JSON-file."""

        with open(import_json, 'r') as fp:
            settings = json.load(fp)

        for key, value in settings.items():
            setattr(self, key, value)

    def export_config(self, export_json):
        """Save settings dict to export_json (path/filename.json) JSON-file."""

        with open(export_json, 'w+') as fp:
            json.dump(vars(self), fp)

    def to_string(self):
        string = ''
        for key, value in vars(self).items():
            string = string + key + ' = ' + str(value) + ', '
        return string


class BEATGANConfig(object):
    def __init__(self, dataset, model_type, x_dim, h_dim, z_dim, preprocessing, use_overlapping, rolling_size, epochs,
                 milestone_epochs, lr, gamma, batch_size, weight_decay, early_stopping, use_clip_norm,
                 gradient_clip_norm, lmbda, display_epoch, save_output, save_figure, save_model, load_model,
                 continue_training, dropout, use_spot, use_last_point, save_config, load_config, server_run, robustness,
                 pid, save_results):

        self.dataset = dataset
        self.model_type = model_type
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.preprocessing = preprocessing
        self.use_overlapping = use_overlapping
        self.rolling_size = rolling_size
        self.epochs = epochs
        self.milestone_epochs = milestone_epochs
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.early_stopping = early_stopping
        self.use_clip_norm = use_clip_norm
        self.gradient_clip_norm = gradient_clip_norm
        self.lmbda = lmbda
        self.display_epoch = display_epoch
        self.save_output = save_output
        self.save_figure = save_figure
        self.save_model = save_model
        self.save_results = save_results
        self.load_model = load_model
        self.continue_training = continue_training
        self.dropout = dropout
        self.use_spot = use_spot
        self.use_last_point = use_last_point
        self.save_config = save_config
        self.load_config = load_config
        self.server_run = server_run
        self.robustness = robustness
        self.pid = pid

    def import_config(self, import_json):
        """Load settings dict from import_json (path/filename.json) JSON-file."""

        with open(import_json, 'r') as fp:
            settings = json.load(fp)

        for key, value in settings.items():
            setattr(self, key, value)

    def export_config(self, export_json):
        """Save settings dict to export_json (path/filename.json) JSON-file."""

        with open(export_json, 'w+') as fp:
            json.dump(vars(self), fp)

    def to_string(self):
        string = ''
        for key, value in vars(self).items():
            string = string + key + ' = ' + str(value) + ', '
        return string


class ISFConfig(object):
    def __init__(self, dataset, x_dim, n_estimators, max_samples, contamination, max_features, bootstrap, save_output,
                 save_figure, save_model, load_model, use_spot, save_config, load_config, server_run,
                 robustness, pid, save_results):

        self.dataset = dataset
        self.x_dim = x_dim
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.bootstrap = bootstrap
        self.max_features = max_features
        self.save_output = save_output
        self.save_results = save_results
        self.save_figure = save_figure
        self.save_model = save_model
        self.load_model = load_model
        self.use_spot = use_spot
        self.save_config = save_config
        self.load_config = load_config
        self.server_run = server_run
        self.robustness = robustness
        self.pid = pid

    def import_config(self, import_json):
        """Load settings dict from import_json (path/filename.json) JSON-file."""

        with open(import_json, 'r') as fp:
            settings = json.load(fp)

        for key, value in settings.items():
            setattr(self, key, value)

    def export_config(self, export_json):
        """Save settings dict to export_json (path/filename.json) JSON-file."""

        with open(export_json, 'w+') as fp:
            json.dump(vars(self), fp)

    def to_string(self):
        string = ''
        for key, value in vars(self).items():
            string = string + key + ' = ' + str(value) + ', '
        return string


class LOFConfig(object):
    def __init__(self, dataset, x_dim, n_neighbors, algorithm, leaf_size, metric, p, contamination, save_output, save_results,
                 save_figure, save_model, load_model, use_spot, save_config, load_config, server_run, robustness, pid):

        self.dataset = dataset
        self.x_dim = x_dim
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.contamination = contamination
        self.save_output = save_output
        self.save_results = save_results
        self.save_figure = save_figure
        self.save_model = save_model
        self.load_model = load_model
        self.use_spot = use_spot
        self.save_config = save_config
        self.load_config = load_config
        self.server_run = server_run
        self.robustness = robustness
        self.pid = pid

    def import_config(self, import_json):
        """Load settings dict from import_json (path/filename.json) JSON-file."""

        with open(import_json, 'r') as fp:
            settings = json.load(fp)

        for key, value in settings.items():
            setattr(self, key, value)

    def export_config(self, export_json):
        """Save settings dict to export_json (path/filename.json) JSON-file."""

        with open(export_json, 'w+') as fp:
            json.dump(vars(self), fp)

    def to_string(self):
        string = ''
        for key, value in vars(self).items():
            string = string + key + ' = ' + str(value) + ', '
        return string


class OCSVMConfig(object):
    def __init__(self, dataset, x_dim, kernel, degree, gamma, coef0, tol, nu, save_output, save_figure,
                 save_model, load_model, use_spot, save_config, load_config, server_run, robustness, pid, save_results):

        self.dataset = dataset
        self.x_dim = x_dim
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.nu = nu
        self.save_output = save_output
        self.save_figure = save_figure
        self.save_model = save_model
        self.save_results = save_results
        self.load_model = load_model
        self.use_spot = use_spot
        self.save_config = save_config
        self.load_config = load_config
        self.server_run = server_run
        self.robustness = robustness
        self.pid = pid

    def import_config(self, import_json):
        """Load settings dict from import_json (path/filename.json) JSON-file."""

        with open(import_json, 'r') as fp:
            settings = json.load(fp)

        for key, value in settings.items():
            setattr(self, key, value)

    def export_config(self, export_json):
        """Save settings dict to export_json (path/filename.json) JSON-file."""

        with open(export_json, 'w+') as fp:
            json.dump(vars(self), fp)

    def to_string(self):
        string = ''
        for key, value in vars(self).items():
            string = string + key + ' = ' + str(value) + ', '
        return string


class MPConfig(object):
    def __init__(self, dataset, x_dim, pattern_size, save_output, save_figure, use_spot, use_last_point, save_config,
                 load_config, server_run, robustness, pid, save_results):

        self.dataset = dataset
        self.x_dim = x_dim
        self.pattern_size = pattern_size
        self.save_output = save_output
        self.save_results = save_results
        self.save_figure = save_figure
        self.use_spot = use_spot
        self.use_last_point = use_last_point
        self.save_config = save_config
        self.load_config = load_config
        self.server_run = server_run
        self.robustness = robustness
        self.pid = pid

    def import_config(self, import_json):
        """Load settings dict from import_json (path/filename.json) JSON-file."""

        with open(import_json, 'r') as fp:
            settings = json.load(fp)

        for key, value in settings.items():
            setattr(self, key, value)

    def export_config(self, export_json):
        """Save settings dict to export_json (path/filename.json) JSON-file."""

        with open(export_json, 'w+') as fp:
            json.dump(vars(self), fp)

    def to_string(self):
        string = ''
        for key, value in vars(self).items():
            string = string + key + ' = ' + str(value) + ', '
        return string


class ARMAConfig(object):
    def __init__(self, dataset, x_dim, train_window, AR, MA, display, save_output, save_figure, use_spot,
                 use_last_point, save_config, load_config, server_run, robustness, pid):

        self.dataset = dataset
        self.x_dim = x_dim
        self.train_window = train_window
        self.AR = AR
        self.MA = MA
        self.display = display
        self.save_output = save_output
        self.save_figure = save_figure
        self.use_spot = use_spot
        self.use_last_point = use_last_point
        self.save_config = save_config
        self.load_config = load_config
        self.server_run = server_run
        self.robustness = robustness
        self.pid = pid

    def import_config(self, import_json):
        """Load settings dict from import_json (path/filename.json) JSON-file."""

        with open(import_json, 'r') as fp:
            settings = json.load(fp)

        for key, value in settings.items():
            setattr(self, key, value)

    def export_config(self, export_json):
        """Save settings dict to export_json (path/filename.json) JSON-file."""

        with open(export_json, 'w+') as fp:
            json.dump(vars(self), fp)

    def to_string(self):
        string = ''
        for key, value in vars(self).items():
            string = string + key + ' = ' + str(value) + ', '
        return string


class MSCREDConfig(object):
    def __init__(self, dataset, x_dim, h_dim, step_max, epochs, milestone_epochs, lr, gamma, batch_size, weight_decay,
                 early_stopping, display_epoch, save_output, save_figure, save_model, load_model, continue_training,
                 dropout, use_spot, use_last_point, save_config, load_config, server_run, robustness, pid):

        self.dataset = dataset
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.step_max = step_max
        self.epochs = epochs
        self.milestone_epochs = milestone_epochs
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.early_stopping = early_stopping
        self.display_epoch = display_epoch
        self.save_output = save_output
        self.save_figure = save_figure
        self.save_model = save_model
        self.load_model = load_model
        self.continue_training = continue_training
        self.dropout = dropout
        self.use_spot = use_spot
        self.use_last_point = use_last_point
        self.save_config = save_config
        self.load_config = load_config
        self.server_run = server_run
        self.robustness = robustness
        self.pid = pid

    def import_config(self, import_json):
        """Load settings dict from import_json (path/filename.json) JSON-file."""

        with open(import_json, 'r') as fp:
            settings = json.load(fp)

        for key, value in settings.items():
            setattr(self, key, value)

    def export_config(self, export_json):
        """Save settings dict to export_json (path/filename.json) JSON-file."""

        with open(export_json, 'w+') as fp:
            json.dump(vars(self), fp)

    def to_string(self):
        string = ''
        for key, value in vars(self).items():
            string = string + key + ' = ' + str(value) + ', '
        return string


class RDAEConfig(object):
    def __init__(self, dataset, window_length, stride, in_out_features, hidden_features_enc, hidden_features_btneck_L,
                 hidden_features_dec_L, hidden_channels_enc, hidden_channels_btneck_L, hidden_channels_dec_L,
                 sequence_length, variant, save_figure, save_output, save_model, load_model, use_spot, use_last_point,
                 save_config, load_config, shrink, lmbda, m_epochs, m_lr, m_batch_size, m_display_epoch, e_epochs, e_lr,
                 e_batch_size, e_display_epoch, ae_epochs, ae_lr, ae_batch_size, ae_display_epoch, d_epochs, d_lr,
                 d_batch_size, d_display_epoch, server_run, robustness, pid):

        self.dataset = dataset
        self.window_length = window_length
        self.stride = stride
        self.in_out_features = in_out_features
        self.hidden_features_enc = hidden_features_enc
        self.hidden_features_btneck_L = hidden_features_btneck_L
        self.hidden_features_dec_L = hidden_features_dec_L
        self.hidden_channels_enc = hidden_channels_enc
        self.hidden_channels_btneck_L = hidden_channels_btneck_L
        self.hidden_channels_dec_L = hidden_channels_dec_L
        self.sequence_length = sequence_length
        self.variant = variant
        self.save_figure = save_figure
        self.save_output = save_output
        self.save_model = save_model
        self.load_model = load_model
        self.use_spot = use_spot
        self.use_last_point = use_last_point
        self.save_config = save_config
        self.load_config = load_config
        self.shrink = shrink
        self.lmbda = lmbda
        self.m_epochs = m_epochs
        self.m_lr = m_lr
        self.m_batch_size = m_batch_size
        self.m_display_epoch = m_display_epoch
        self.e_epochs = e_epochs
        self.e_lr = e_lr
        self.e_batch_size = e_batch_size
        self.e_display_epoch = e_display_epoch
        self.ae_epochs = ae_epochs
        self.ae_lr = ae_lr
        self.ae_batch_size = ae_batch_size
        self.ae_display_epoch = ae_display_epoch
        self.d_epochs = d_epochs
        self.d_lr = d_lr
        self.d_batch_size = d_batch_size
        self.d_display_epoch = d_display_epoch
        self.server_run = server_run
        self.robustness = robustness
        self.pid = pid

    def import_config(self, import_json):
        """Load settings dict from import_json (path/filename.json) JSON-file."""

        with open(import_json, 'r') as fp:
            settings = json.load(fp)

        for key, value in settings.items():
            setattr(self, key, value)

    def export_config(self, export_json):
        """Save settings dict to export_json (path/filename.json) JSON-file."""

        with open(export_json, 'w+') as fp:
            json.dump(vars(self), fp)

    def to_string(self):
        string = ''
        for key, value in vars(self).items():
            string = string + key + ' = ' + str(value) + ', '
        return string


class HIFIConfig(object):
    def __init__(self, dataset, x_dim, h_dim, d_inner, n_layers, n_head, d_k, d_v, gcn_layers, gcn_alpha, k,
                 preprocessing, use_overlapping, rolling_size, epochs, milestone_epochs, lr, gamma,
                 batch_size, weight_decay, early_stopping, loss_function, use_clip_norm, gradient_clip_norm,
                 display_epoch, save_output, save_figure, save_model, load_model, continue_training, dropout,
                 use_spot, use_last_point, save_config, load_config, server_run, robustness, pid, save_results,
                 kl_start, kl_warmup):

        self.dataset = dataset
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.d_inner = d_inner
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.gcn_layers = gcn_layers
        self.gcn_alpha = gcn_alpha
        self.k = k
        self.preprocessing = preprocessing
        self.use_overlapping = use_overlapping
        self.rolling_size = rolling_size
        self.epochs = epochs
        self.milestone_epochs = milestone_epochs
        self.lr = lr
        self.kl_start = kl_start
        self.kl_warmup = kl_warmup
        self.gamma = gamma
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.early_stopping = early_stopping
        self.loss_function = loss_function
        self.use_clip_norm = use_clip_norm
        self.gradient_clip_norm = gradient_clip_norm
        self.display_epoch = display_epoch
        self.save_output = save_output
        self.save_figure = save_figure
        self.save_model = save_model
        self.save_results = save_results
        self.load_model = load_model
        self.continue_training = continue_training
        self.dropout = dropout
        self.use_spot = use_spot
        self.use_last_point = use_last_point
        self.save_config = save_config
        self.load_config = load_config
        self.server_run = server_run
        self.robustness = robustness
        self.pid = pid

    def import_config(self, import_json):
        """Load settings dict from import_json (path/filename.json) JSON-file."""

        with open(import_json, 'r') as fp:
            settings = json.load(fp)

        for key, value in settings.items():
            setattr(self, key, value)

    def export_config(self, export_json):
        """Save settings dict to export_json (path/filename.json) JSON-file."""

        with open(export_json, 'w+') as fp:
            json.dump(vars(self), fp)

    def to_string(self):
        string = ''
        for key, value in vars(self).items():
            string = string + key + ' = ' + str(value) + ', '
        return string
