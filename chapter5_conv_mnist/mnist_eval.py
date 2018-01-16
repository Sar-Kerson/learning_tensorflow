#coding=utf-8
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

import mnist_inference
import mnist_train



# load the model every 10 sec, and test the accuracy on the test set
EVAL_INTERVAL_SRCS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:

        x = tf.placeholder(dtype=tf.float32,
                           shape=[mnist_train.BATCH_SIZE, mnist_inference.IMAGE_SIZE,
                                  mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS],
                           name="x_input")

        y_true = tf.placeholder(dtype=tf.float32,
                                shape=[None, mnist_inference.OUTPUT_NODE],
                                name="y_input")

        xv, yv = mnist.validation.next_batch(mnist_train.BATCH_SIZE)
        reshaped_xv = np.reshape(xv, (mnist_train.BATCH_SIZE,
                                      mnist_inference.IMAGE_SIZE,
                                      mnist_inference.IMAGE_SIZE,
                                      mnist_inference.NUM_CHANNELS))

        validate_feed = {x: reshaped_xv,
                         y_true: yv}

        y = mnist_inference.inference(x, reuse=False, train=False, regularizer=None)


        correct_prediction = tf.equal(tf.argmax(y, 1),
                                      tf.argmax(y_true, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 共用前向传播的过程
        variable_average = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVG_DEC)
        variables_to_restore = variable_average.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                # 自动寻找最新模型
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy,
                                              feed_dict=validate_feed)
                    print("After %s training steps, validation accuracy =%g"
                          % (global_step, accuracy_score))
                else:
                    print("No checkpoint file found")
                    return
                time.sleep(EVAL_INTERVAL_SRCS)


def main(argv=None):
    mnist = input_data.read_data_sets(mnist_train.MNIST_PATH, one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
    tf.app.run()