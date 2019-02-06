import numpy as np
from json import dump
from os import makedirs
from os.path import join, exists
from uuid import uuid4
from google.protobuf.json_format import MessageToDict
from datetime import datetime as dt


def gen_model_name():
    now = dt.now().strftime('%Y-%m-%d_%H-%m')
    return '{}_{:06X}'.format(now, uuid4().int >> 104)


def get_logs_dir(options, model_name=None):
    if not exists(options.storage.logs):
        makedirs(options.storage.logs)

    model_name = model_name or gen_model_name()
    logs_dir = join(options.storage.logs, model_name)
    if not exists(logs_dir):
        makedirs(logs_dir)
    options_dict = MessageToDict(
        message=options, including_default_value_fields=True, preserving_proto_field_name=True)

    with open(join(logs_dir, 'options.json'), 'w') as f:
        dump(options_dict, f, sort_keys=True, indent=2)

    return logs_dir


class StatsRecorder:
    """
    StatsRecords is usefull when computing mean and standard deviation 
    in a huge amount of data.

    source: http://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
    """

    def __init__(self, data=None):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if data is not None:
            data = np.atleast_2d(data)
            self.mean = data.mean(axis=0)
            self.std = data.std(axis=0)
            self.nobservations = data.shape[0]
            self.ndimensions = data.shape[1]
        else:
            self.nobservations = 0

    def update(self, data):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if self.nobservations == 0:
            self.__init__(data)
        else:
            data = np.atleast_2d(data)
            if data.shape[1] != self.ndimensions:
                raise ValueError("Data dims don't match prev observations.")

            newmean = data.mean(axis=0)
            newstd = data.std(axis=0)

            m = self.nobservations * 1.0
            n = data.shape[0]

            tmp = self.mean

            self.mean = m / (m + n) * tmp + n / (m + n) * newmean
            self.std  = m/(m+n)*self.std**2 + n/(m+n)*newstd**2 +\
                        m*n/(m+n)**2 * (tmp - newmean)**2
            self.std = np.sqrt(self.std)

            self.nobservations += n
