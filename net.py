import tensorflow as tf
import numpy as np

import functools
import warnings
import inspect
import six
from abc import ABCMeta, abstractmethod
from tensorflow.python.training import moving_averages

# from tensorlayer import tl_logging as logging


@six.add_metaclass(ABCMeta)
class LayersConfig(object):
    tf_dtype = tf.float32  # TensorFlow DType
    set_keep = {}  # A dictionary for holding tf.placeholders

    @abstractmethod
    def __init__(self):
        pass


def get_collection_trainable(name=''):
    variables = []
    for p in tf.trainable_variables():
        # print(p.name.rpartition('/')[0], self.name)
        if p.name.rpartition('/')[0] == name:
            variables.append(p)
    return variables


def list_remove_repeat(x):
    """Remove the repeated items in a list, and return the processed list.
    You may need it to create merged layer like Concat, Elementwise and etc.

    Parameters
    ----------
    x : list
        Input

    Returns
    -------
    list
        A list that after removing it's repeated items

    Examples
    -------
    >>> l = [2, 3, 4, 2, 3]
    >>> l = list_remove_repeat(l)
    [2, 3, 4]

    """
    y = []
    for i in x:
        if i not in y:
            y.append(i)

    return y


def private_method(func):
    """decorator for making an instance method private"""

    def func_wrapper(*args, **kwargs):
        """decorator wrapper function"""
        outer_frame = inspect.stack()[1][0]
        if 'self' not in outer_frame.f_locals or outer_frame.f_locals['self'] is not args[0]:
            raise RuntimeError('%s.%s is a private method' % (args[0].__class__.__name__, func.__name__))

        return func(*args, **kwargs)

    return func_wrapper


def protected_method(func):
    """decorator for making an instance method private"""

    def func_wrapper(*args, **kwargs):
        """decorator wrapper function"""
        outer_frame = inspect.stack()[1][0]

        caller = inspect.getmro(outer_frame.f_locals['self'].__class__)[:-1]
        target = inspect.getmro(args[0].__class__)[:-1]

        share_subsclass = False

        for cls_ in target:
            if issubclass(caller[0], cls_) or caller[0] is cls_:
                share_subsclass = True
                break

        if ('self' not in outer_frame.f_locals or
            outer_frame.f_locals['self'] is not args[0]) and (not share_subsclass):
            raise RuntimeError('%s.%s is a protected method' % (args[0].__class__.__name__, func.__name__))

        return func(*args, **kwargs)

    return func_wrapper


def deprecated_alias(end_support_version, **aliases):
    def deco(f):

        @functools.wraps(f)
        def wrapper(*args, **kwargs):

            try:
                func_name = "{}.{}".format(args[0].__class__.__name__, f.__name__)
            except (NameError, IndexError):
                func_name = f.__name__

            rename_kwargs(kwargs, aliases, end_support_version, func_name)

            return f(*args, **kwargs)

        return wrapper

    return deco


def rename_kwargs(kwargs, aliases, end_support_version, func_name):
    for alias, new in aliases.items():

        if alias in kwargs:

            if new in kwargs:
                raise TypeError('{}() received both {} and {}'.format(func_name, alias, new))

            warnings.warn('{}() - {} is deprecated; use {}'.format(func_name, alias, new), DeprecationWarning)
            # logging.warning(
            #     "DeprecationWarning: {}(): "
            #     "`{}` argument is deprecated and will be removed in version {}, "
            #     "please change for `{}.`".format(func_name, alias, end_support_version, new)
            # )
            kwargs[new] = kwargs.pop(alias)


class Layer(object):
    """The basic :class:`Layer` class represents a single layer of a neural network.

    It should be subclassed when implementing new types of layers.
    Because each layer can keep track of the layer(s) feeding into it, a
    network's output :class:`Layer` instance can double as a handle to the full
    network.

    Parameters
    ----------
    prev_layer : :class:`Layer` or None
        Previous layer (optional), for adding all properties of previous layer(s) to this layer.
    act : activation function (None by default)
        The activation function of this layer.
    name : str or None
        A unique layer name.

    Methods
    ---------
    print_params(details=True, session=None)
        Print all parameters of this network.
    print_layers()
        Print all outputs of all layers of this network.
    count_params()
        Return the number of parameters of this network.

    Examples
    ---------

    - Define model

    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> x = tf.placeholder("float32", [None, 100])
    >>> n = tl.layers.InputLayer(x, name='in')
    >>> n = tl.layers.DenseLayer(n, 80, name='d1')
    >>> n = tl.layers.DenseLayer(n, 80, name='d2')

    - Get information

    >>> print(n)
    Last layer is: DenseLayer (d2) [None, 80]
    >>> n.print_layers()
    [TL]   layer   0: d1/Identity:0        (?, 80)            float32
    [TL]   layer   1: d2/Identity:0        (?, 80)            float32
    >>> n.print_params(False)
    [TL]   param   0: d1/W:0               (100, 80)          float32_ref
    [TL]   param   1: d1/b:0               (80,)              float32_ref
    [TL]   param   2: d2/W:0               (80, 80)           float32_ref
    [TL]   param   3: d2/b:0               (80,)              float32_ref
    [TL]   num of params: 14560
    >>> n.count_params()
    14560

    - Slicing the outputs

    >>> n2 = n[:, :30]
    >>> print(n2)
    Last layer is: Layer (d2) [None, 30]

    - Iterating the outputs

    >>> for l in n:
    >>>    print(l)
    Tensor("d1/Identity:0", shape=(?, 80), dtype=float32)
    Tensor("d2/Identity:0", shape=(?, 80), dtype=float32)

    """

    # Added to allow auto-completion

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(self, prev_layer, act=None, name=None, *args, **kwargs):

        self.inputs = None
        self.outputs = None

        self.all_layers = list()
        self.all_params = list()
        self.all_drop = dict()

        if name is None:
            raise ValueError('Layer must have a name.')

        for key in kwargs.keys():
            setattr(self, key, self._argument_dict_checkup(kwargs[key]))

        self.act = act if act not in [None, tf.identity] else None

        scope_name = tf.get_variable_scope().name

        self.name = scope_name + '/' + name if scope_name else name

        if isinstance(prev_layer, Layer):
            # 1. for normal layer have only 1 input i.e. DenseLayer
            # Hint : list(), dict() is pass by value (shallow), without them,
            # it is pass by reference.

            self.inputs = prev_layer.outputs

            self._add_layers(prev_layer.all_layers)
            self._add_params(prev_layer.all_params)
            self._add_dropout_layers(prev_layer.all_drop)

        elif isinstance(prev_layer, list):
            # 2. for layer have multiply inputs i.e. ConcatLayer

            self.inputs = [layer.outputs for layer in prev_layer]

            self._add_layers(sum([l.all_layers for l in prev_layer], []))
            self._add_params(sum([l.all_params for l in prev_layer], []))
            self._add_dropout_layers(sum([list(l.all_drop.items()) for l in prev_layer], []))

        elif isinstance(prev_layer, tf.Tensor) or isinstance(prev_layer, tf.Variable):  # placeholders
            if self.__class__.__name__ not in ['InputLayer', 'OneHotInputLayer', 'Word2vecEmbeddingInputlayer',
                                               'EmbeddingInputlayer', 'AverageEmbeddingInputlayer']:
                raise RuntimeError("Please use `tl.layers.InputLayer` to convert Tensor/Placeholder to a TL layer")

            self.inputs = prev_layer

        elif prev_layer is not None:
            # 4. tl.models
            self._add_layers(prev_layer.all_layers)
            self._add_params(prev_layer.all_params)
            self._add_dropout_layers(prev_layer.all_drop)

            if hasattr(prev_layer, "outputs"):
                self.inputs = prev_layer.outputs

    def print_params(self, details=True, session=None):
        """Print all info of parameters in the network"""
        for i, p in enumerate(self.all_params):
            if details:
                try:
                    val = p.eval(session=session)
                    # logging.info(
                    #     "  param {:3}: {:20} {:15}    {} (mean: {:<18}, median: {:<18}, std: {:<18})   ".format(
                    #         i, p.name, str(val.shape), p.dtype.name, val.mean(), np.median(val), val.std()
                    #     )
                    # )
                except Exception as e:
                    # logging.info(str(e))
                    raise Exception(
                        "Hint: print params details after tl.layers.initialize_global_variables(sess) "
                        "or use network.print_params(False)."
                    )
            else:
                pass
        #         logging.info("  param {:3}: {:20} {:15}    {}".format(i, p.name, str(p.get_shape()), p.dtype.name))
        # logging.info("  num of params: %d" % self.count_params())

    def print_layers(self):
        """Print all info of layers in the network"""

        for i, layer in enumerate(self.all_layers):
            # logging.info("  layer %d: %s" % (i, str(layer)))
            pass
            # logging.info(
            #     "  layer {:3}: {:20} {:15}    {}".format(i, layer.name, str(layer.get_shape()), layer.dtype.name)
            # )

    def count_params(self):
        """Return the number of parameters in the network"""
        n_params = 0
        for _i, p in enumerate(self.all_params):
            n = 1
            # for s in p.eval().shape:
            for s in p.get_shape():
                try:
                    s = int(s)
                except Exception:
                    s = 1
                if s:
                    n = n * s
            n_params = n_params + n
        return n_params

    def __str__(self):
        return "  Last layer is: %s (%s) %s" % (self.__class__.__name__, self.name, self.outputs.get_shape().as_list())

    def __getitem__(self, key):

        net_new = Layer(prev_layer=None, name=self.name)

        net_new.inputs = self.inputs
        net_new.outputs = self.outputs[key]

        net_new._add_layers(self.all_layers[:-1])
        net_new._add_layers(net_new.outputs)

        net_new._add_params(self.all_params)

        net_new._add_dropout_layers(self.all_drop)

        return net_new

    def __setitem__(self, key, item):
        raise TypeError("The Layer API does not allow to use the method: `__setitem__`")

    def __delitem__(self, key):
        raise TypeError("The Layer API does not allow to use the method: `__delitem__`")

    def __iter__(self):
        for x in self.all_layers:
            yield x

    def __len__(self):
        return len(self.all_layers)

    @protected_method
    def _add_layers(self, layers):
        if isinstance(layers, list):
            try:  # list of class Layer
                new_layers = [layer.outputs for layer in layers]
                self.all_layers.extend(list(new_layers))

            except AttributeError:  # list of tf.Tensor
                self.all_layers.extend(list(layers))

        else:
            self.all_layers.append(layers)

        self.all_layers = list_remove_repeat(self.all_layers)

    @protected_method
    def _add_params(self, params):

        if isinstance(params, list):
            self.all_params.extend(list(params))

        else:
            self.all_params.append(params)

        self.all_params = list_remove_repeat(self.all_params)

    @protected_method
    def _add_dropout_layers(self, drop_layers):
        if isinstance(drop_layers, dict) or isinstance(drop_layers, list):
            self.all_drop.update(dict(drop_layers))

        elif isinstance(drop_layers, tuple):
            self.all_drop.update(list(drop_layers))

        else:
            raise ValueError()

    @private_method
    def _apply_activation(self, logits, **kwargs):
        if not kwargs:
            kwargs = {}
        return self.act(logits, **kwargs) if self.act is not None else logits

    @private_method
    def _argument_dict_checkup(self, args):

        if not isinstance(args, dict) and args is not None:
            raise AssertionError(
                "One of the argument given to %s should be formatted as a dictionary" % self.__class__.__name__
            )

        return args if args is not None else {}


class InputLayer(Layer):
    """
    The :class:`InputLayer` class is the starting layer of a neural network.

    Parameters
    ----------
    inputs : placeholder or tensor
        The input of a network.
    name : str
        A unique layer name.

    """

    def __init__(self, inputs, name='input'):
        super(InputLayer, self).__init__(prev_layer=inputs, name=name)

        # logging.info("InputLayer  %s: %s" % (self.name, inputs.get_shape()))

        self.outputs = inputs

        self._add_layers(self.outputs)


class Conv2d(Layer):
    """Simplified version of :class:`Conv2dLayer`.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
    n_filter : int
        The number of filters.
    filter_size : tuple of int
        The filter size (height, width).
    strides : tuple of int
        The sliding window strides of corresponding input dimensions.
        It must be in the same order as the ``shape`` parameter.
    act : activation function
        The activation function of this layer.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    W_init : initializer
        The initializer for the the weight matrix.
    b_init : initializer or None
        The initializer for the the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for the weight matrix initializer (for TF < 1.5).
    b_init_args : dictionary
        The arguments for the bias vector initializer (for TF < 1.5).
    use_cudnn_on_gpu : bool
        Default is False (for TF < 1.5).
    data_format : str
        "NHWC" or "NCHW", default is "NHWC" (for TF < 1.5).
    name : str
        A unique layer name.

    Returns
    -------
    :class:`Layer`
        A :class:`Conv2dLayer` object.

    Examples
    --------
    >>> x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    >>> net = InputLayer(x, name='inputs')
    >>> net = Conv2d(net, 64, (3, 3), act=tf.nn.relu, name='conv1_1')
    >>> net = Conv2d(net, 64, (3, 3), act=tf.nn.relu, name='conv1_2')
    >>> net = MaxPool2d(net, (2, 2), name='pool1')
    >>> net = Conv2d(net, 128, (3, 3), act=tf.nn.relu, name='conv2_1')
    >>> net = Conv2d(net, 128, (3, 3), act=tf.nn.relu, name='conv2_2')
    >>> net = MaxPool2d(net, (2, 2), name='pool2')

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            n_filter=32,
            filter_size=(3, 3),
            strides=(1, 1),
            act=None,
            padding='SAME',
            dilation_rate=(1, 1),
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,
            b_init_args=None,
            use_cudnn_on_gpu=None,
            data_format=None,
            name='conv2d',
    ):
        # if len(strides) != 2:
        #     raise ValueError("len(strides) should be 2, Conv2d and Conv2dLayer are different.")

        # try:
        #     pre_channel = int(layer.outputs.get_shape()[-1])

        # except Exception:  # if pre_channel is ?, it happens when using Spatial Transformer Net
        #     pre_channel = 1
        #     logging.info("[warnings] unknow input channels, set to 1")

        super(Conv2d, self
              ).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, b_init_args=b_init_args, name=name)

        # logging.info(
        #     "Conv2d %s: n_filter: %d filter_size: %s strides: %s pad: %s act: %s" % (
        #         self.name, n_filter, str(filter_size), str(strides), padding, self.act.__name__
        #         if self.act is not None else 'No Activation'
        #     )
        # )
        # with tf.variable_scope(name) as vs:
        conv2d = tf.layers.Conv2D(
            # inputs=self.inputs,
            filters=n_filter,
            kernel_size=filter_size,
            strides=strides,
            padding=padding,
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=self.act,
            use_bias=(False if b_init is None else True),
            kernel_initializer=W_init,  # None,
            bias_initializer=b_init,  # f.zeros_initializer(),
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            trainable=True,
            name=name,
            # reuse=None,
        )
        self.outputs = conv2d(self.inputs)  # must put before ``new_variables``
        # new_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=self.name)  #vs.name)
        new_variables = get_collection_trainable(self.name)
        # new_variables = []
        # for p in tf.trainable_variables():
        #     # print(p.name.rpartition('/')[0], self.name)
        #     if p.name.rpartition('/')[0] == self.name:
        #         new_variables.append(p)
        # exit()
        # TF_GRAPHKEYS_VARIABLES  TF_GRAPHKEYS_VARIABLES
        # print(self.name, name)
        # print(tf.trainable_variables())#tf.GraphKeys.TRAINABLE_VARIABLES)
        # print(new_variables)
        # print(conv2d.weights)

        self._add_layers(self.outputs)
        self._add_params(new_variables)  # conv2d.weights)


class BatchNormLayer(Layer):
    """
    The :class:`BatchNormLayer` is a batch normalization layer for both fully-connected and convolution outputs.
    See ``tf.nn.batch_normalization`` and ``tf.nn.moments``.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        The previous layer.
    decay : float
        A decay factor for `ExponentialMovingAverage`.
        Suggest to use a large value for large dataset.
    epsilon : float
        Eplison.
    act : activation function
        The activation function of this layer.
    is_train : boolean
        Is being used for training or inference.
    beta_init : initializer or None
        The initializer for initializing beta, if None, skip beta.
        Usually you should not skip beta unless you know what happened.
    gamma_init : initializer or None
        The initializer for initializing gamma, if None, skip gamma.
        When the batch normalization layer is use instead of 'biases', or the next layer is linear, this can be
        disabled since the scaling can be done by the next layer. see `Inception-ResNet-v2 <https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_resnet_v2.py>`__
    name : str
        A unique layer name.

    References
    ----------
    - `Source <https://github.com/ry/tensorflow-resnet/blob/master/resnet.py>`__
    - `stackoverflow <http://stackoverflow.com/questions/38312668/how-does-one-do-inference-with-batch-normalization-with-tensor-flow>`__

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            decay=0.9,
            epsilon=0.00001,
            act=None,
            is_train=False,
            beta_init=tf.zeros_initializer,
            gamma_init=tf.random_normal_initializer(mean=1.0, stddev=0.002),
            moving_mean_init=tf.zeros_initializer(),
            name='batchnorm_layer',
    ):
        super(BatchNormLayer, self).__init__(prev_layer=prev_layer, act=act, name=name)

        # logging.info(
        #     "BatchNormLayer %s: decay: %f epsilon: %f act: %s is_train: %s" %
        #     (self.name, decay, epsilon, self.act.__name__ if self.act is not None else 'No Activation', is_train)
        # )

        x_shape = self.inputs.get_shape()
        params_shape = x_shape[-1:]

        with tf.variable_scope(name):
            axis = list(range(len(x_shape) - 1))
            # 1. beta, gamma
            variables = []

            if beta_init:

                if beta_init == tf.zeros_initializer:
                    beta_init = beta_init()

                beta = tf.get_variable(
                    'beta', shape=params_shape, initializer=beta_init, dtype=LayersConfig.tf_dtype, trainable=is_train
                )

                variables.append(beta)

            else:
                beta = None

            if gamma_init:
                gamma = tf.get_variable(
                    'gamma',
                    shape=params_shape,
                    initializer=gamma_init,
                    dtype=LayersConfig.tf_dtype,
                    trainable=is_train,
                )
                variables.append(gamma)
            else:
                gamma = None

            # 2.

            moving_mean = tf.get_variable(
                'moving_mean', params_shape, initializer=moving_mean_init, dtype=LayersConfig.tf_dtype, trainable=False
            )

            moving_variance = tf.get_variable(
                'moving_variance',
                params_shape,
                initializer=tf.constant_initializer(1.),
                dtype=LayersConfig.tf_dtype,
                trainable=False,
            )

            # 3.
            # These ops will only be preformed when training.
            mean, variance = tf.nn.moments(self.inputs, axis)

            update_moving_mean = moving_averages.assign_moving_average(
                moving_mean, mean, decay, zero_debias=False
            )  # if zero_debias=True, has bias

            update_moving_variance = moving_averages.assign_moving_average(
                moving_variance, variance, decay, zero_debias=False
            )  # if zero_debias=True, has bias

            def mean_var_with_update():
                with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                    return tf.identity(mean), tf.identity(variance)

            if is_train:
                mean, var = mean_var_with_update()
            else:
                mean, var = moving_mean, moving_variance

            self.outputs = self._apply_activation(
                tf.nn.batch_normalization(self.inputs, mean, var, beta, gamma, epsilon)
            )

            variables.extend([moving_mean, moving_variance])

        self._add_layers(self.outputs)
        self._add_params(variables)


class DepthwiseConv2d(Layer):
    """Separable/Depthwise Convolutional 2D layer, see `tf.nn.depthwise_conv2d <https://www.tensorflow.org/versions/master/api_docs/python/tf/nn/depthwise_conv2d>`__.

    Input:
        4-D Tensor (batch, height, width, in_channels).
    Output:
        4-D Tensor (batch, new height, new width, in_channels * depth_multiplier).

    Parameters
    ------------
    prev_layer : :class:`Layer`
        Previous layer.
    filter_size : tuple of int
        The filter size (height, width).
    stride : tuple of int
        The stride step (height, width).
    act : activation function
        The activation function of this layer.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    dilation_rate: tuple of 2 int
        The dilation rate in which we sample input values across the height and width dimensions in atrous convolution. If it is greater than 1, then all values of strides must be 1.
    depth_multiplier : int
        The number of channels to expand to.
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip bias.
    W_init_args : dictionary
        The arguments for the weight matrix initializer.
    b_init_args : dictionary
        The arguments for the bias vector initializer.
    name : str
        A unique layer name.

    Examples
    ---------
    >>> net = InputLayer(x, name='input')
    >>> net = Conv2d(net, 32, (3, 3), (2, 2), b_init=None, name='cin')
    >>> net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bnin')
    ...
    >>> net = DepthwiseConv2d(net, (3, 3), (1, 1), b_init=None, name='cdw1')
    >>> net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn11')
    >>> net = Conv2d(net, 64, (1, 1), (1, 1), b_init=None, name='c1')
    >>> net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn12')
    ...
    >>> net = DepthwiseConv2d(net, (3, 3), (2, 2), b_init=None, name='cdw2')
    >>> net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn21')
    >>> net = Conv2d(net, 128, (1, 1), (1, 1), b_init=None, name='c2')
    >>> net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn22')

    References
    -----------
    - tflearn's `grouped_conv_2d <https://github.com/tflearn/tflearn/blob/3e0c3298ff508394f3ef191bcd7d732eb8860b2e/tflearn/layers/conv.py>`__
    - keras's `separableconv2d <https://keras.io/layers/convolutional/#separableconv2d>`__

    """  # # https://zhuanlan.zhihu.com/p/31551004  https://github.com/xiaohu2015/DeepLearning_tutorials/blob/master/CNNs/MobileNet.py

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            shape=(3, 3),
            strides=(1, 1),
            act=None,
            padding='SAME',
            dilation_rate=(1, 1),
            depth_multiplier=1,
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,
            b_init_args=None,
            name='depthwise_conv2d',
    ):
        super(DepthwiseConv2d, self
              ).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, b_init_args=b_init_args, name=name)

        # logging.info(
        #     "DepthwiseConv2d %s: shape: %s strides: %s pad: %s act: %s" % (
        #         self.name, str(shape), str(strides), padding, self.act.__name__
        #         if self.act is not None else 'No Activation'
        #     )
        # )

        try:
            pre_channel = int(prev_layer.outputs.get_shape()[-1])
        except Exception:  # if pre_channel is ?, it happens when using Spatial Transformer Net
            pre_channel = 1
            # logging.info("[warnings] unknown input channels, set to 1")

        shape = [shape[0], shape[1], pre_channel, depth_multiplier]

        if len(strides) == 2:
            strides = [1, strides[0], strides[1], 1]

        if len(strides) != 4:
            raise AssertionError("len(strides) should be 4.")

        with tf.variable_scope(name):

            W = tf.get_variable(
                name='W_depthwise2d', shape=shape, initializer=W_init, dtype=LayersConfig.tf_dtype, **self.W_init_args
            )  # [filter_height, filter_width, in_channels, depth_multiplier]

            self.outputs = tf.nn.depthwise_conv2d(self.inputs, W, strides=strides, padding=padding, rate=dilation_rate)

            if b_init:
                b = tf.get_variable(
                    name='b_depthwise2d', shape=(pre_channel * depth_multiplier), initializer=b_init,
                    dtype=LayersConfig.tf_dtype, **self.b_init_args
                )

                self.outputs = tf.nn.bias_add(self.outputs, b, name='bias_add')

            self.outputs = self._apply_activation(self.outputs)

        self._add_layers(self.outputs)

        if b_init:
            self._add_params([W, b])
        else:
            self._add_params(W)
