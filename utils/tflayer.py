#! /usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import tensorflow as tf

set_keep = globals()
set_keep['_layers_name_list'] = []
set_keep['name_reuse'] = False


class Layer(object):
    def __init__(self, inputs=None, name='layer'):
        # 1.set inputs
        self.inputs = inputs
        # 2. set name
        scope_name = tf.get_variable_scope().name
        if scope_name:
            name = scope_name + '/' + name
        if (name in set_keep['_layers_name_list']) and set_keep['name_reuse'] == False:
            raise Exception("Layers %s exists" % (name))
        else:
            self.name = name
            if name not in ['', None, False]:
                set_keep['_layers_name_list'].append(name)

    def print_params(self, details=True):
        for i, p in enumerate(self.all_params):
            if details:
                pass
            else:
                print("param {:3}: {:15} {}".format(i, str(p.get_shape()), p.name))
        print("nums of params %d: " % (self.count_params()))

    def count_params(self):
        n_params = 0
        for i, p in enumerate(self.all_params):
            n = 1
            for s in p.get_shape():
                try:
                    s = int(s)
                except:
                    s = 1
                if s:
                    s = n * s
            n_params = n_params + n
        return n_params

    def print_layers(self):
        for i, l in enumerate(self.all_layers):
            print("layers %d, %s" % (i, str(l)))

    def __str__(self):
        return "layer %s " % (self.__class__.__name__)


class InputLayer(Layer):
    def __init__(self, inputs=None, name='input_layer'):
        Layer.__init__(self, inputs=inputs, name=name)
        print("[DeepCV] InputLayer %s: %s" % (self.name, inputs.get_shape()))

        self.all_params = []
        self.all_layers = []
        self.all_drop = {}

        self.outputs = inputs


class OneHotInputLayer(Layer):
    '''
    eg.
        Suppose that:
            indices = [0, 2, -1, 1]
            depth = 3
            on_value = 5.0
            off_value = 0.0
            axis = -1
        Then output is [4 x 3]:
            output =
            [5.0 0.0 0.0]  // one_hot(0)
            [0.0 0.0 5.0]  // one_hot(2)
            [0.0 0.0 0.0]  // one_hot(-1)
            [0.0 5.0 0.0]  // one_hot(1)
    '''

    def __init__(self, inputs=None, depth=None, on_value=None, off_value=None, axis=None, dtype=None,
                 name='onehot_input_layer'):
        Layer.__init__(self, inputs=inputs, name=name)
        print("[DeepCV] OneHotInputLayer %s: %s" % (self.name, inputs.get_shape()))

        self.all_params = []
        self.all_layers = []
        self.all_drop = {}

        assert depth != None, "depth is not given"
        self.outputs = tf.one_hot(inputs, depth=depth, on_value=on_value, off_value=off_value, axis=axis, dtype=dtype)


class DenseLayer(Layer):
    def __init__(self, layer=None, out_units=100, act=tf.identity,
                 W_init=tf.truncated_normal_initializer(stddev=0.1),
                 b_init=tf.constant_initializer(value=0.0),
                 W_init_args={},
                 b_init_args={},
                 name='dense_layer'):

        Layer.__init__(self, inputs=layer.outputs, name=name)
        # self.inputs = layer.outputs
        if self.inputs.get_shape().ndims != 2:
            raise Exception("the input dimension must be rank 2!")
        in_units = self.inputs.get_shape()[-1]
        self.out_units = out_units
        print("[DeepCV] DenseLayer %s: units:%d activation:%s" % (self.name, self.out_units, act.__name__))

        self.all_params = list(layer.all_params)
        self.all_layers = list(layer.all_layers)
        self.all_drop = dict(layer.all_drop)

        with tf.variable_scope(name) as vs:
            W = tf.get_variable(name='W', shape=(in_units, out_units), initializer=W_init)
            self.outputs = act(tf.matmul(self.inputs, W))
            self.all_params.extend([W])
            if b_init:
                b = tf.get_variable(name='b', shape=out_units, initializer=b_init)
                self.outputs = act(self.outputs + b)
                self.all_params.extend([W, b])
            self.all_layers.extend([self.outputs])


class Conv2dLayer(Layer):
    def __init__(self, layer=None, act=tf.identity, shape=None, strides=[1, 1, 1, 1], padding='SAME',
                 W_init=tf.truncated_normal_initializer(stddev=0.2),
                 b_init=tf.constant_initializer(value=0.0),
                 w_init_args={},
                 b_init_args={},
                 use_cudnn_on_gpu=None,
                 data_format=None,
                 name='conv2d_layer'):
        Layer.__init__(self, inputs=layer.outputs, name=name)
        # self.inputs = layer.outputs
        print("[DeepCV] Conv2dLayer %s: shape:%s strides:%s padding:%s act:%s"
              % (self.name, str(shape), str(strides), padding, act.__name__))

        self.all_params = list(layer.all_params)
        self.all_layers = list(layer.all_layers)
        self.all_drop = dict(layer.all_drop)

        with tf.variable_scope(name) as vs:
            W = tf.get_variable(name='W', shape=shape, initializer=W_init)
            self.outputs = act(tf.nn.conv2d(self.inputs, W, strides=strides, padding=padding,
                                            use_cudnn_on_gpu=use_cudnn_on_gpu, data_format=data_format))
            self.all_params.extend([W])
            if b_init:
                self.outputs = act(self.outputs + b)
                self.all_params.extend([W, b])
            self.all_layers.extend([self.outputs])


class PoolLayer(Layer):
    def __int__(self, layer=None, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                pool=tf.nn.max_pool, name='pool_layer'):
        Layer.__init__(self, inputs=layer.outputs, name=name)
        print("[DeepCV] PoolLayer %s: ksize:%s strides:%s padding:%s pool:%s" %
              (self.name, str(ksize), str(strides), padding, pool.__name__))

        self.all_params = list(layer.all_params)
        self.all_layers = list(layer.all_layers)
        self.all_drop = dict(layer.all_drop)

        self.outputs = pool(self.inputs, ksize=ksize, strides=strides, padding=padding, name=name)
        self.all_layers.extend([self.outputs])


class DropoutLayer(Layer):
    def __init__(self, layer=None, keep=0.5, seed=None, is_fix=False, is_train=True, name='dropout_layer'):
        Layer.__init__(self, inputs=layer.outputs, name=name)
        self.all_params = list(layer.all_params)
        self.all_layers = list(layer.all_layers)
        self.all_drop = dict(layer.all_drop)
        if is_train:
            print("[DeepCV] DropoutLayer %s: keep:%s is_fix:%s" % (self.name, str(keep), str(is_fix)))
            if is_fix:
                self.outputs = tf.nn.dropout(self.inputs, keep, seed=seed, name=name)
            else:
                set_keep[name] = tf.placeholder(tf.float32)
                self.outputs = tf.nn.dropout(self.inputs, set_keep[name], seed=seed, name=name)
                self.all_drop.update({set_keep[name]: keep})
            self.all_layers.extend([self.outputs])

        else:
            print("[DeepCV] skip dropout!")
            self.outputs = layer.outputs
