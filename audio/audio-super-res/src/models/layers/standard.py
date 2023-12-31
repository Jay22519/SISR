import numpy as np
import tensorflow as tf
from keras.layers.advanced_activations import PReLU

from .summarization import create_var_summaries

# ----------------------------------------------------------------------------


def conv1d(x, n_filters, n_size, stride=1, nl="relu", name="conv1d"):
    n_batch, n_dim, n_input_chan = x.get_shape()
    with tf.compat.v1.variable_scope(name):
        # create and track weights
        with tf.compat.v1.name_scope("weights"):
            W = tf.compat.v1.get_variable(
                "W",
                shape=[n_size, n_input_chan, n_filters],
                initializer=tf.compat.v1.random_normal_initializer(stddev=1e-3),
            )
            create_var_summaries(W)

        # create and track biases
        with tf.compat.v1.name_scope("biases"):
            b = tf.compat.v1.get_variable(
                "b", [n_filters], initializer=tf.compat.v1.constant_initializer(0.0)
            )
            create_var_summaries(b)

        # create and track pre-activations
        with tf.compat.v1.name_scope("preactivations"):
            x = tf.nn.conv1d(input=x, filters=W, stride=1, padding="SAME")
            x = tf.nn.bias_add(x, b)
            tf.compat.v1.summary.histogram("preactivations", x)

        # create and track activations
        if nl == "relu":
            x = tf.nn.relu(x)
        elif nl == "prelu":
            x = PReLU()(x)
        elif nl == None:
            pass
        else:
            raise ValueError("Invalid non-linearity")

        tf.compat.v1.summary.histogram("activations", x)

    return x


def deconv1d(x, r, n_chan, n_in_dim, n_in_chan, name="deconv1d"):
    x = tf.reshape(x, [128, 1, n_in_dim, n_in_chan])
    with tf.compat.v1.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        W = tf.compat.v1.get_variable(
            "W",
            shape=[1, r, n_chan, n_in_chan],
            initializer=tf.compat.v1.random_normal_initializer(stddev=1e-3),
        )
        b = tf.compat.v1.get_variable(
            "b", [n_chan], initializer=tf.compat.v1.constant_initializer(0.0)
        )

        x = tf.nn.conv2d_transpose(
            x, W, output_shape=(128, 1, r * n_in_dim, n_chan), strides=[1, 1, r, 1]
        )
        x = tf.nn.bias_add(x, b)

    return tf.reshape(x, [-1, r * n_in_dim, n_chan])
