# Copyright (c) Mike.
import paddle
import os
import numpy as np
import random
import logging
import time
import math
import sys
from paddle.optimizer.lr import LRScheduler


def label_to_onehot(label, total_cls):
    """ one-hot encodes a tensor 
    
    Args:
        label: [B,1]
        total_cls: xxx
    Return:
        paddle.Tensor(float32)
    """
    label_onehot = np.eye(total_cls)[label.reshape([-1]),]
    label_onehot = paddle.to_tensor(label_onehot).astype('float32')

    return label_onehot


def get_seed(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def get_work_dir():

    work_dir = os.path.join(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    print('----- 📂 output directory:', os.path.abspath(work_dir))

    return work_dir


def get_logger(work_dir=None):
    """set logging file and format
    Args:
        workdir: output directory
    Return:
        logger: python logger
    """
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt="[%m-%d %H:%M:%S]")
    # different name is needed when creating multiple logger in one process
    logger = logging.getLogger()

    if not work_dir == None:
        fh = logging.FileHandler(os.path.join(work_dir, 'log.txt'))
    else:
        fh = logging.FileHandler('log.txt')

    fh.setFormatter(logging.Formatter(log_format))
    logger.addHandler(fh)

    return logger

class AverageMeter():
    """ Meter for monitoring losses"""
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.reset()

    def reset(self):
        """reset all values to zeros"""
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        """update avg by val and n, where val is the avg of n values"""
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
