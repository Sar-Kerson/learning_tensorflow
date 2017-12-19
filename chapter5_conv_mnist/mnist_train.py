#coding=utf-8
import os

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DEC = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVG_DEC = 0.99

MODEL_SAVE_PATH = "/tmp/model_mnist"
MODEL_NAME = "model_mnist.ckpt"

def train(mnist):
    x = tf.placeholder(
        tf.float32,
        shape=[BATCH_SIZE, mnist_inference.IMAGE_SIZE,
         mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS],
        name="x_input")

    y_true = tf.placeholder(
        tf.float32,
        shape=[None, mnist_inference.OUTPUT_NODE],
        name="y_input")

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    y = mnist_inference.inference(x, reuse=False, train=True, regularizer=regularizer)

    global_step = tf.Variable(0, trainable=False)

    # 损失函数,滑动平均,训练过程
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVG_DEC, global_step)
    variable_average_op = variable_average.apply(tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.argmax(y_true, 1),
        logits=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))

    learning_rate = tf.train.exponential_decay(
        learning_rate=LEARNING_RATE_BASE,
        global_step=global_step,
        decay_steps=mnist.train.num_examples / BATCH_SIZE,
        decay_rate=LEARNING_RATE_DEC)
    train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss, global_step)

    # 与train_op = tf.group(train_step, variables_averages_op)是等价的
    with tf.control_dependencies([train_step, variable_average_op]):
        train_op = tf.no_op(name="train")

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            xs_reshaped = np.reshape(xs, (BATCH_SIZE,
                                          mnist_inference.IMAGE_SIZE,
                                          mnist_inference.IMAGE_SIZE,
                                          mnist_inference.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x: xs_reshaped, y_true: ys})

            if i % 1000 == 0:
                print("After %d training steps, loss on training batch is %g."
                      % (step, loss_value))

                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),
                           global_step=global_step)

def main(argv=None):
    mnist = input_data.read_data_sets("/tmp/mnist", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()