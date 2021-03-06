#coding=utf-8
import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DEC = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVG_DEC = 0.99

DATA = "/tmp/mnist"
# MODEL_NAME = "model_mnist.ckpt"

def train(mnist):

    with tf.name_scope('input'):
        x = tf.placeholder(
            tf.float32,
            [None, mnist_inference.INPUT_NODE],
            name="x_input")

        y_true = tf.placeholder(
            tf.float32,
            [None, mnist_inference.OUTPUT_NODE],
            name="y_input")

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    y = mnist_inference.inference(x, regularizer=regularizer)

    global_step = tf.Variable(0, trainable=False)

    # 损失函数,滑动平均,训练过程
    with tf.name_scope('moving_avg'):
        variable_average = tf.train.ExponentialMovingAverage(MOVING_AVG_DEC, global_step)
        variable_average_op = variable_average.apply(tf.trainable_variables())

    with tf.name_scope('loss_func'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.argmax(y_true, 1),
            logits=y)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))

    with tf.name_scope('train_step'):
        learning_rate = tf.train.exponential_decay(
            learning_rate=LEARNING_RATE_BASE,
            global_step=global_step,
            decay_steps=mnist.train.num_examples / BATCH_SIZE,
            decay_rate=LEARNING_RATE_DEC)
        train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss, global_step)

        # 与train_op = tf.group(train_step, variables_averages_op)是等价的
        with tf.control_dependencies([train_step, variable_average_op]):
            train_op = tf.no_op(name="train")

        # saver = tf.train.Saver()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            writer = tf.summary.FileWriter(os.path.join(DATA, 'log'), tf.get_default_graph())

            for i in range(TRAINING_STEPS):
                xs, ys = mnist.train.next_batch(BATCH_SIZE)
                _, loss_value, step = sess.run([train_op, loss, global_step],
                                               feed_dict={x: xs, y_true: ys})
                # add---2
                if i % 1000 == 0:
                    '''add---2'''
                    run_option = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    _, loss_value, step = sess.run([train_op, loss, global_step],
                                                   feed_dict={x:xs, y_true: ys},
                                                   options=run_option,
                                                   run_metadata=run_metadata)
                    writer.add_run_metadata(run_metadata, 'step%03d' % i)

                    print("After %d training steps, loss on training batch is %g."
                          % (step, loss_value))

                else:
                    _, loss_value, step = sess.run([train_op, loss, global_step],
                                                   feed_dict={x:xs, y_true: ys})
                    '''add---2'''

                    # saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),
                    #            global_step=global_step)
            writer.close()

def main(argv=None):
    mnist = input_data.read_data_sets(DATA, one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()