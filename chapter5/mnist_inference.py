#coding=utf-8
import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500



def get_weight_variable(shape, regularizer):
    weights = tf.get_variable(
        "weights",
        shape=shape,
        initializer=tf.truncated_normal_initializer(stddev=0.1))

    # 当给出 正规化 ，将当前变量的“正则化损失”加入losses集合
    if regularizer != None:
        tf.add_to_collection("losses", regularizer(weights))

    return weights

def inference(input_tensor, regularizer):
    with tf.variable_scope("layer1"):
        # 这里get_variable与Variable无区别，因为没有在同一个程序里多次调用
        # 同一程序中多次调用，第一次调用之后reuse设置Ture
        weights = get_weight_variable(
            [INPUT_NODE, LAYER1_NODE], regularizer=regularizer)

        biases = tf.get_variable(
            "biases",
            [LAYER1_NODE],
            initializer=tf.constant_initializer(0.0))

        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope("layer2"):
        weights = get_weight_variable(
            [LAYER1_NODE, OUTPUT_NODE], regularizer=regularizer)

        biases = tf.get_variable(
            "biases",
            [OUTPUT_NODE],
            initializer=tf.constant_initializer(0.0))

        layer2 = tf.matmul(layer1, weights) + biases

    return layer2