#!venv/bin/python
# https://medium.com/tensorflow/introducing-tensorflow-probability-dca4c304e245
import logging
import tensorflow as tf
import tensorflow_probability as tfp
from cifar10_web import cifar10
import numpy as np

logging.basicConfig(level=logging.DEBUG)

#
# load data
#
logging.debug("loading data...")
train_images, train_labels, test_images, test_labels = cifar10(path="./data/cifar/")

# reduce the size
SIZE = np.shape(train_images)[0]
train_images = train_images[:SIZE]
logging.debug(np.shape(train_images))
# features = train_images
num_input = 3072  # np.shape(train_images)[1]
features = tf.placeholder("float", [None, num_input])

train_labels = train_labels[:SIZE]
logging.debug(np.shape(train_labels))
# labels = train_labels
num_classes = 10  # np.shape(train_labels)[1]
labels = tf.placeholder("float", [None, num_classes])

#
# create model
#
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

#
# criticize model
#
logits = model(features)
neg_log_likelihood = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
kl = sum(model.get_losses_for(inputs=None))
loss = neg_log_likelihood + kl
train_op = tf.train.AdamOptimizer().minimize(loss)

#
# start
#
logging.debug("running...")
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    batch_size = 10
    for n in range(0, SIZE, batch_size):
        sess.run(train_op, feed_dict={features: train_images[n:n + batch_size], labels: train_labels[n:n + batch_size]})
        if n % 100 == 0:
            loss_value = sess.run(
                [loss], feed_dict={features: train_images[n:n + 1], labels: train_labels[n:n + 1]})
            print(loss_value)
            print("n:%d, Loss:%f" % (n, loss_value[0]))
logging.debug("done")
