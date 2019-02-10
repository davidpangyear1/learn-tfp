#!venv/bin/python
# https://medium.com/tensorflow/introducing-tensorflow-probability-dca4c304e245
import logging
import tensorflow as tf
import tensorflow_probability as tfp
from cifar10_web import cifar10

logging.basicConfig(level=logging.DEBUG)
logging.debug("loading data...")
train_images, train_labels, test_images, test_labels = cifar10(path="./data/cifar/")

logging.debug("creating model...")
# Flipout estimator:
# https://arxiv.org/abs/1803.04386
model = tf.keras.Sequential(
    [
        tf.keras.layers.Reshape([32, 32, 3]),
        tfp.layers.Convolution2DFlipout(
            64, kernel_size=5, padding='SAME', activation=tf.nn.relu
        ),
        tf.keras.layers.MaxPool2D(pool_size=[2, 2],
                                  strides=[2, 2],
                                  padding='SAME'),
        tf.keras.layers.Reshape([16 * 16 * 64]),
        tfp.layers.DenseFlipout(10)
    ]
)

logits = model(train_images)
neg_log_likelihood = tf.nn.softmax_cross_entropy_with_logits_v2(labels=train_labels, logits=logits)
kl = sum(model.get_losses_for(inputs=None))
loss = neg_log_likelihood + kl
train_op = tf.train.AdamOptimizer().minimize(loss)

logging.debug("running...")
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    sess.run(train_op)

logging.debug("done")
