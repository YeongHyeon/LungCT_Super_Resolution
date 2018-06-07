import argparse

import tensorflow as tf

import source.neuralnet as nn
import source.datamanager as dman
import source.tf_process as tfp
import source.stamper as stamper
stamper.print_stamp()

def main():

    srnet = nn.SRNET()

    dataset = dman.DataSet()

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    tfp.training(sess=sess, neuralnet=srnet, saver=saver, dataset=dataset, iteration=int(FLAGS.iter))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--iter', type=int, default=8*1e8, help='-')

    FLAGS, unparsed = parser.parse_known_args()

    main()
