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

    def next_batch(self, batch_size=1, idx=-1):

        input = np.zeros((0, 1, 1, 1))
        ground = np.zeros((0, 1, 1, 1))
        if(idx == -1):
            while(True):
                np_input = np.load(self.list_input[self.data_idx])
                np_input = np_input.reshape((1, np_input.shape[0], np_input.shape[1], -1))
                np_ground = np.load(self.list_ground[self.data_idx])
                np_ground = np_ground.reshape((1, np_ground.shape[0], np_ground.shape[1], -1))

                if(input.shape[0] == 0):
                    input = np.zeros((0, np_input.shape[1], np_input.shape[2], np_input.shape[3]))
                    ground = np.zeros((0, np_ground.shape[1], np_ground.shape[2], np_ground.shape[3]))

                input = np.append(input, np_input, axis=0)
                ground = np.append(ground, np_ground, axis=0)

                if(input.shape[0] >= batch_size):
                    break
                else:
                    self.data_idx = (self.data_idx + 1) % self.amount

            input, ground = np_input, np_ground

        else:
            np_input = np.load(self.list_input[idx])
            np_input = np_input.reshape((1, np_input.shape[0], np_input.shape[1], -1))
            np_ground = np.load(self.list_ground[idx])
            np_ground = np_ground.reshape((1, np_ground.shape[0], np_ground.shape[1], -1))

            input, ground = np_input, np_ground

        return input, ground
