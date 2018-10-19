import tensorflow as tf
# import tensorlayer as tl
# from tensorlayer.layers import *
from net import InputLayer, Conv2d, BatchNormLayer, DepthwiseConv2d
import numpy as np
import time
import cv2


# input_pb = tf.placeholder(tf.float32, [None, 416, 416, 3])
def infenence(input_pb):
    is_train = True
    n = 9 * (5 + 1)
    net = InputLayer(input_pb, name='input')
    net = Conv2d(net, n_filter=32, filter_size=(3, 3), strides=(2, 2), b_init=None, name='cin')
    net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bnin')

    net = DepthwiseConv2d(net, shape=(3, 3), strides=(1, 1), b_init=None, name='cdw1')
    net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn11')
    net = Conv2d(net, n_filter=64, filter_size=(1, 1), strides=(1, 1), b_init=None, name='c1')
    net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn12')

    net = DepthwiseConv2d(net, shape=(3, 3), strides=(2, 2), b_init=None, name='cdw2')
    net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn13')
    net = Conv2d(net, n_filter=128, filter_size=(1, 1), strides=(1, 1), b_init=None, name='c2')
    net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn14')

    net = DepthwiseConv2d(net, shape=(3, 3), strides=(1, 1), b_init=None, name='cdw3')
    net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn15')
    net = Conv2d(net, n_filter=128, filter_size=(1, 1), strides=(1, 1), b_init=None, name='c3')
    net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn16')

    net = DepthwiseConv2d(net, shape=(3, 3), strides=(2, 2), b_init=None, name='cdw4')
    net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn17')
    net = Conv2d(net, n_filter=256, filter_size=(1, 1), strides=(1, 1), b_init=None, name='c4')
    net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn18')

    net = DepthwiseConv2d(net, shape=(3, 3), strides=(1, 1), b_init=None, name='cdw5')
    net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn19')
    net = Conv2d(net, n_filter=256, filter_size=(1, 1), strides=(1, 1), b_init=None, name='c5')
    net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn110')

    net = DepthwiseConv2d(net, shape=(3, 3), strides=(2, 2), b_init=None, name='cdw6')
    net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn111')
    net = Conv2d(net, n_filter=512, filter_size=(1, 1), strides=(1, 1), b_init=None, name='c6')
    net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn112')

    net = DepthwiseConv2d(net, shape=(3, 3), strides=(1, 1), b_init=None, name='cdw7')
    net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn113')
    net = Conv2d(net, n_filter=512, filter_size=(1, 1), strides=(1, 1), b_init=None, name='c7')
    net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn114')
    net = DepthwiseConv2d(net, shape=(3, 3), strides=(1, 1), b_init=None, name='cdw8')
    net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn115')
    net = Conv2d(net, n_filter=512, filter_size=(1, 1), strides=(1, 1), b_init=None, name='c8')
    net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn116')
    net = DepthwiseConv2d(net, shape=(3, 3), strides=(1, 1), b_init=None, name='cdw9')
    net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn117')
    net = Conv2d(net, n_filter=512, filter_size=(1, 1), strides=(1, 1), b_init=None, name='c9')
    net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn118')
    net = DepthwiseConv2d(net, shape=(3, 3), strides=(1, 1), b_init=None, name='cdw10')
    net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn119')
    net = Conv2d(net, n_filter=512, filter_size=(1, 1), strides=(1, 1), b_init=None, name='c10')
    net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn120')
    net = DepthwiseConv2d(net, shape=(3, 3), strides=(1, 1), b_init=None, name='cdw11')
    net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn121')
    net = Conv2d(net, n_filter=512, filter_size=(1, 1), strides=(1, 1), b_init=None, name='c11')
    net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn122')
    #
    net = DepthwiseConv2d(net, shape=(3, 3), strides=(2, 2), b_init=None, name='cdw12')
    net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn123')
    net = Conv2d(net, n_filter=1024, filter_size=(1, 1), strides=(1, 1), b_init=None, name='c12')
    net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn124')
    #
    net = DepthwiseConv2d(net, shape=(3, 3), strides=(1, 1), b_init=None, name='cdw13')
    net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn125')
    net = Conv2d(net, n_filter=n, filter_size=(1, 1), strides=(1, 1), b_init=None, name='c13')
    net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn126')

    # net = UpSampling2dLayer(net, (2, 2), method=1, is_scale=True, name='upsample')
    return net.outputs


if __name__ == '__main__':
    input_pb = tf.placeholder(tf.float32, [None, 416, 416, 3])
    with tf.Session() as sess:
        net = infenence(input_pb)

        sess.run(tf.global_variables_initializer())
        for i in range(10):
            img = cv2.imread('4.jpg')
            img = cv2.resize(img, (416, 416))
            img = np.expand_dims(img, axis=0)

            time_1 = time.time()
            a = sess.run(net, feed_dict={input_pb: img})
            print(time.time() - time_1)
