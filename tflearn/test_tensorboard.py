#coding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


DATASET = "/home/yaoleiqi/tmp/mnist"
SUMMARY_DIR = "/home/yaoleiqi/tmp/mnist/log"
BATCH_SIZE = 100
TRAIN_STEPS = 30000


# name: 可视化结果中显示的图表名称
def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        # 会写入HISTOGRAMS栏下
        tf.summary.histogram(name=name, values=var)
        # 计算mean，同一命名空间下的变量会整合到一起
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        # 计算标准差
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddef/' + name, stddev)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):

        with tf.name_scope('weights'):
            weights = tf.Variable(tf.truncated_normal(shape=[input_dim, output_dim], stddev=0.1))
            variable_summaries(weights, layer_name + '/weight')

        with tf.name_scope('biases'):
            biases = tf.Variable(tf.constant(0.0, shape=[output_dim]))
            variable_summaries(biases, layer_name + '/biases')

        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(weights, input_tensor) + biases
            # 记录激活前的分布
            tf.summary.histogram(layer_name + '/preactivations', preactivate)

        activations = act(preactivate, name='activation')
        # 记录激活之后的分布
        tf.summary.histogram(layer_name + '/activations', activations)
        return activations


if __name__ == '__main__':
    mnist = input_data.read_data_sets(DATASET, one_hot=True)
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y = tf.placeholder(tf.float32, [None, 10], name='y-input')

    with tf.name_scope('input-reshape'):
        img_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image(name='input', tensor=img_shaped_input, max_outputs=10)

    hidden1 = nn_layer(x, 784, 500, 'layer1')
    y_ = nn_layer(hidden1, 500, 10, 'layer2', act=tf.identity)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))
        tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdagradOptimizer(0.001).minimize(cross_entropy)

    # 计算正确率并生成监控日志
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_pred = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    # merge_all 隐式调用了所有的summary，不用一一调用它们
    merged = tf.summary.merge_all()

    with tf.Session() as sess:

        writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
        tf.global_variables_initializer().run()

        for i in range(TRAIN_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)

            summary, _ = sess.run([merged, train_step], feed_dict={x: xs, y: ys})

            writer.add_summary(summary, i)

    writer.close()






