import os, inspect, glob

import numpy as np

class DataSet(object):

    def __init__(self):

        self.data_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/../dataset"
        self.list_input = glob.glob(os.path.join(self.data_path, "low_tr", "*.npy"))
        self.list_input.sort()
        self.list_ground = glob.glob(os.path.join(self.data_path, "high_tr", "*.npy"))
        self.list_ground.sort()

        self.amount = len(self.list_input)
        self.data_idx = 0

    def next_batch(self, idx=-1):

        if(idx == -1):
            np_input = np.load(self.list_input[self.data_idx])
            np_input = np_input.reshape((1, np_input.shape[0], np_input.shape[1], -1))
            np_ground = np.load(self.list_ground[self.data_idx])
            np_ground = np_ground.reshape((1, np_ground.shape[0], np_ground.shape[1], -1))

            input, ground = np_input, np_ground

            self.data_idx = (self.data_idx + 1) % self.amount
        else:
            np_input = np.load(self.list_input[idx])
            np_input = np_input.reshape((1, np_input.shape[0], np_input.shape[1], -1))
            np_ground = np.load(self.list_ground[idx])
            np_ground = np_ground.reshape((1, np_ground.shape[0], np_ground.shape[1], -1))

            input, ground = np_input, np_ground

        return input, ground
