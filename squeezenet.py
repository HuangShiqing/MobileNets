import tensorflow as tf
from tensorflow.contrib.layers import conv2d, avg_pool2d, max_pool2d
from tensorflow.contrib.layers import batch_norm, l2_regularizer
import cv2
import numpy as np
import time


def fire_module(inputs,
                squeeze_depth,
                expand_depth,
                reuse=None,
                scope=None):
    with tf.variable_scope(scope, 'fire', [inputs], reuse=reuse):
        # with arg_scope([conv2d, max_pool2d]):
        net = _squeeze(inputs, squeeze_depth)
        net = _expand(net, expand_depth)
        return net


def _squeeze(inputs, num_outputs):
    return conv2d(inputs, num_outputs, [1, 1], stride=1, scope='squeeze')


def _expand(inputs, num_outputs):
    with tf.variable_scope('expand'):
        e1x1 = conv2d(inputs, num_outputs, [1, 1], stride=1, scope='1x1')
        e3x3 = conv2d(inputs, num_outputs, [3, 3], scope='3x3')
    return tf.concat([e1x1, e3x3], axis=-1)


def _squeezenet(images, num_classes=1000):
    # net = conv2d(images, 96, [7, 7], stride=2, scope='conv1')
    net = conv2d(images, 96, [3, 3], stride=2, scope='conv1')
    net = max_pool2d(net, [3, 3], stride=2, scope='maxpool1')
    net = fire_module(net, 16, 64, scope='fire2')
    net = fire_module(net, 16, 64, scope='fire3')
    net = fire_module(net, 32, 128, scope='fire4')
    net = max_pool2d(net, [3, 3], stride=2, scope='maxpool4')
    net = fire_module(net, 32, 128, scope='fire5')
    net = fire_module(net, 48, 192, scope='fire6')
    net = fire_module(net, 48, 192, scope='fire7')
    net = fire_module(net, 64, 256, scope='fire8')
    net = max_pool2d(net, [3, 3], stride=2, scope='maxpool8')
    net = fire_module(net, 64, 256, scope='fire9')
    net = conv2d(net, num_classes, [1, 1], stride=1, scope='conv10')
    # net = avg_pool2d(net, [13, 13], stride=1, scope='avgpool10')
    # logits = tf.squeeze(net, [2], name='logits')
    return net


if __name__ == '__main__':
    input_pb = tf.placeholder(tf.float32, [None, 416, 416, 3])
    with tf.Session() as sess:
        net = _squeezenet(input_pb, 10)

        sess.run(tf.global_variables_initializer())
        for i in range(10):
            img = cv2.imread('4.jpg')
            img = cv2.resize(img, (416, 416))
            img = np.expand_dims(img, axis=0)

            time_1 = time.time()
            a = sess.run(net, feed_dict={input_pb: img})
            print(time.time() - time_1)
