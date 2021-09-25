import logging
import os

# The default level of logger is logging.INFO
def setup_logger(name, log_file, level=logging.INFO):
    """Function setup as many loggers as you want"""
    log_format = (
        '%(asctime)s - '
        '%(name)s - '
        '%(funcName)s - '
        '%(levelname)s - '
        '%(message)s'
    )
    logger = logging.getLogger(name)
    # Before add handler, we should clear the previous handler to avoid duplicate printing.
    logger.handlers.clear()
    logger.setLevel(level)
    formatter = logging.Formatter(log_format)
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    # copy the handler to the global variables
    g_filehandler = fh
    g_streamhandle = sh
    # copy the logger to the global variables
    g_logger = logger
    return logger


def create_logger(dataset, train_logger_name, file_logger_name, meta_logger_name, model_name, pid, h_dim=None, rolling_size=None):
    if not os.path.exists('./logs/{}/'.format(dataset)):
        os.makedirs('./logs/{}/'.format(dataset))

    # first file logger
    file_logger = setup_logger(file_logger_name + model_name + str(pid),
                               './logs/{}/file_hdim_{}_rollingsize_{}_{}_{}.log'.format(dataset, h_dim, rolling_size, model_name, pid))
    # second file logger
    meta_logger = setup_logger(meta_logger_name + model_name + str(pid),
                               './logs/{}/meta_hdim_{}_rollingsize_{}_{}_{}.log'.format(dataset, h_dim, rolling_size, model_name, pid))
    # third file logger
    train_logger = setup_logger(train_logger_name + model_name + str(pid),
                                './logs/{}/train_hdim_{}_rollingsize_{}_{}_{}.log'.format(dataset, h_dim,
                                                                                                   rolling_size,
                                                                                                   model_name, pid))

    return train_logger, file_logger, meta_logger
