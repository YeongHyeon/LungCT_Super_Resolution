import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np

class SRNET(object):

    def __init__(self):

        print("\n** Initialize Super-Resolution Network")

        self.inputs = tf.placeholder(tf.float32, [None, None, None, None])
        self.outputs = tf.placeholder(tf.float32, [None, None, None, None])

        self.channel = 1
        self.n1 = 32
        self.n2 = 16
        self.n3 = 16
        self.n4 = 16

        self.f1 = 3
        self.f2 = 3
        self.f3 = 3
        self.f4 = 1
        self.f5 = 3

        self.weights = {
            'w1': tf.Variable(tf.random_normal([self.f1, self.f1, self.channel, self.n1], stddev=0.001)),
            'w2': tf.Variable(tf.random_normal([self.f2, self.f2, self.n1, self.n2], stddev=0.001)),
            'w3': tf.Variable(tf.random_normal([self.f3, self.f3, self.n2, self.n3], stddev=0.001)),
            'res': tf.Variable(tf.random_normal([1, 1, self.channel, self.n3], stddev=0.001)),
            'w4': tf.Variable(tf.random_normal([self.f4, self.f4, self.n3, self.n4], stddev=0.001)),
            'w5': tf.Variable(tf.random_normal([self.f5, self.f5, self.n4, self.channel], stddev=0.001)),
        }

        self.biases = {
            'b1': tf.Variable(tf.zeros([self.n1])),
            'b2': tf.Variable(tf.zeros([self.n2])),
            'b3': tf.Variable(tf.zeros([self.n3])),
            'res': tf.Variable(tf.zeros([self.n3])),
            'b4': tf.Variable(tf.zeros([self.n4])),
            'b5': tf.Variable(tf.zeros([self.channel])),
        }

        self.conv1 = tf.nn.relu(tf.add(tf.nn.conv2d(self.inputs, self.weights['w1'], strides=[1, 1, 1, 1], padding='SAME'), self.biases['b1']))
        self.conv2 = tf.nn.relu(tf.add(tf.nn.conv2d(self.conv1, self.weights['w2'], strides=[1, 1, 1, 1], padding='SAME'), self.biases['b2']))
        self.conv3 = tf.nn.relu(tf.add(tf.nn.conv2d(self.conv2, self.weights['w3'], strides=[1, 1, 1, 1], padding='SAME'), self.biases['b3']))
        self.res = tf.nn.relu(tf.add(tf.nn.conv2d(self.inputs, self.weights['res'], strides=[1, 1, 1, 1], padding='SAME'), self.biases['res']))
        self.conv4 = tf.nn.relu(tf.add(tf.nn.conv2d(self.conv3+self.res, self.weights['w4'], strides=[1, 1, 1, 1], padding='SAME'), self.biases['b4']))
        self.recon_tmp = tf.add(tf.nn.conv2d(self.conv4, self.weights['w5'], strides=[1, 1, 1, 1], padding='SAME'), self.biases['b5'])

        self.recon = tf.clip_by_value(self.recon_tmp, clip_value_min=0.0, clip_value_max=1.0)

        self.loss = tf.sqrt(tf.reduce_sum(tf.square(self.recon - self.outputs)))
        self.psnr = tf.log(1 / tf.sqrt(tf.reduce_mean(tf.square(self.recon - self.outputs)))) / tf.log(10.0) * 20

        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss=self.loss)

        tf.summary.histogram('w1', self.weights['w1'])
        tf.summary.histogram('w2', self.weights['w2'])
        tf.summary.histogram('w3', self.weights['w3'])
        tf.summary.histogram('res', self.weights['res'])
        tf.summary.histogram('w4', self.weights['w4'])
        tf.summary.histogram('w5', self.weights['w5'])
        tf.summary.histogram('b1', self.biases['b1'])
        tf.summary.histogram('b2', self.biases['b2'])
        tf.summary.histogram('b3', self.biases['b3'])
        tf.summary.histogram('res', self.biases['res'])
        tf.summary.histogram('b4', self.biases['b4'])
        tf.summary.histogram('b5', self.biases['b5'])

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('psnr', self.psnr)

        self.summaries = tf.summary.merge_all()
