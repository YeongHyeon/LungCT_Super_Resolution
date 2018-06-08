import os, inspect, glob

import numpy as np

class DataSet(object):

    def __init__(self):

        self.data_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/../dataset"
        self.input_tr = glob.glob(os.path.join(self.data_path, "low_tr", "*.npy"))
        self.input_tr.sort()
        self.ground_tr = glob.glob(os.path.join(self.data_path, "high_tr", "*.npy"))
        self.ground_tr.sort()

        self.input_te = glob.glob(os.path.join(self.data_path, "low_te", "*.npy"))
        self.input_te.sort()
        self.ground_te = glob.glob(os.path.join(self.data_path, "high_te", "*.npy"))
        self.ground_te.sort()

        self.amount_tr = len(self.input_tr)
        self.amount_te = len(self.ground_te)
        self.idx_tr = 0

    def next_batch(self, batch_size=1, idx=-1):

        input = np.zeros((0, 1, 1, 1))
        ground = np.zeros((0, 1, 1, 1))
        if(idx == -1):
            while(True):
                np_input = np.load(self.input_tr[self.idx_tr])
                np_input = np_input.reshape((1, np_input.shape[0], np_input.shape[1], -1))
                np_ground = np.load(self.ground_tr[self.idx_tr])
                np_ground = np_ground.reshape((1, np_ground.shape[0], np_ground.shape[1], -1))

                if(input.shape[0] == 0):
                    input = np.zeros((0, np_input.shape[1], np_input.shape[2], np_input.shape[3]))
                    ground = np.zeros((0, np_ground.shape[1], np_ground.shape[2], np_ground.shape[3]))

                input = np.append(input, np_input, axis=0)
                ground = np.append(ground, np_ground, axis=0)

                if(input.shape[0] >= batch_size):
                    break
                else:
                    self.idx_tr = (self.idx_tr + 1) % self.amount_tr

            input, ground = np_input, np_ground

        else:
            np_input = np.load(self.input_te[idx])
            np_input = np_input.reshape((1, np_input.shape[0], np_input.shape[1], -1))
            np_ground = np.load(self.ground_te[idx])
            np_ground = np_ground.reshape((1, np_ground.shape[0], np_ground.shape[1], -1))

            input, ground = np_input, np_ground

        return input, ground
