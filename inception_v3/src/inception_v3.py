# coding=utf-8
import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

# Inception-v3 模型瓶颈层节点数
BOTTLENECT_TENSOR_SIZE = 2048

# Inception-v3 模型瓶颈层结果的tensor名: pool_3/_reshape:0
BOTTLENECT_TENSOR_NAME = 'pool_3/_reshape:0'

# 图像input的tensor名
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

# inception-v3模型目录
MODEL_DIR = '/home/sar/Downloads/inception_dec_2015'

# inception-v3模型文件名
MODEL_FILE = 'tensorflow_inception_graph.pb'

# 将原始图像计算得到的特征tensor保存到文件中免去重复计算？
CACHE_DIR = '/tmp/bottleneck'

# 图片数据集  每个子目录代表一个class
INPUT_DATA = '/home/sar/Downloads/flower_photos'

# 验证数据百分比
VALIDATION_PERCENTAGE = 10
# 测试数据百分比
TEST_PERCENTAGE = 10

# 定义神经网络设置
LEARNING_RATE = 0.01
STEPS = 4000
BATCH = 100

# 从数据文件读取所有图片列表
# 并按训练 验证 测试数据分开
def create_image_lists(testing_percentage, validation_percentage):
    # key 为类别  value为图片文件名
    result = {}
    # 获取当前目录下所有子目录 x[0]为目录名   x[1]为该目录下的子目录list  x[2]为该目录下的文件list
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    is_root_dir = True

    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        # 获取当前目录下的有效图片
        extensions = ['jpg', 'JPG', 'jpeg', 'JPEG']
        file_list = []
        # 目录名称
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)  # 获取某类文件的所有文件
            file_list.extend(glob.glob(file_glob))  # ??
        if not file_list: continue

        label_name = dir_name.lower()     # =dir_name ??
        # 初始化训练集 测试集 验证集
        training_imgs = []
        testing_imgs = []
        validation_imgs = []
        for file_name in file_list:                     # am I right ?
            base_name = os.path.basename(file_name)     # 遍历所有图片
            chance = np.random.randint(100)
            if chance < validation_percentage:          # 10%
                validation_imgs.append(base_name)
            if chance < (testing_percentage + validation_percentage):   # 20%
                testing_imgs.append(base_name)
            else:                                       # 70%
                training_imgs.append(base_name)

        result[label_name] = {
            'dir': dir_name,                            # 目录名 == label_name ?
            'training': training_imgs,                  # 图片文件名
            'testing': testing_imgs,
            'validation': validation_imgs,
        }
    return  result

# 通过类名称 所属数据集 图片编号  获取图片路径
def get_image_path(image_lists, image_dir, label_name, index, category):
    # 给定类别所有图片
    label_lists = image_lists[label_name]       # flower_a {dir, training, testing...}
    # 数据集
    category_liist = label_lists[category]      # training [ a.jpg, b.JPG, c.JPEG, d.jpeg ]
    mod_index = index % len(category_liist)
    # 获取文件名
    base_name = category_liist[mod_index]       # x.jpg
    sub_dir = label_lists['dir']                # x_label
    # 文件地址
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path

# 通过类名称 所属数据集 图片编号 获得特征向量路径
def get_bottleneck_path(image_lists, label_name, index, category):
    return get_image_path(image_lists, CACHE_DIR, label_name, index, category) + '.txt'

# 使用inception-v3处理一张图片，得到特征向量
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    bottleneck_values = sess.run(bottleneck_tensor,
                                 { image_data_tensor: image_data })
    # 将4维的卷积层的结果压缩成1维数组
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values

# 获取一张图片的特征向量
# 若之前计算过，则直接使用
# 否则，重新计算，并保存文件
def get_or_create_bottleneck(sess, image_lists, label_name, index,
                             category, jpeg_data_tensor, bottleneck_tensor):
    # 获取图片对应特征向量文件名
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(CACHE_DIR, sub_dir)
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index, category)

    # 如果文件不存在
    if not os.path.exists(bottleneck_path):
        # 原始图片路径
        image_path = get_image_path(image_lists, INPUT_DATA, label_name, index, category)
        # 获取图片
        image_data = gfile.FastGFile(image_path, 'rb').read()
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)  # 将向量写入文件，对称写法，参考
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        # 直接读取
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]    # 对称写法，参考
    return bottleneck_values

# 随机选择一个batch文件作为训练数据
def get_random_cached_bottlenecks(sess, n_classes, image_lists, batch_size,
                                  category, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    for _ in range(batch_size):
        label_index = random.randrange(n_classes)                       # randint ?
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65536)                           # 不必担心数过大，会取模
        bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, image_index,
                                              category, jpeg_data_tensor, bottleneck_tensor)    # 特征向量
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    return bottlenecks, ground_truths

# 获取全部测试数据  计算测试集上的正确率
def get_test_bottlenecks(sess, image_lists, n_classes,
                         jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    label_names = list(image_lists.keys())
    for label_index, label_name in enumerate(label_names):             # flower_a, flower_b
        category = 'testing'
        for index, unusesd_base_name in enumerate(image_lists[label_name][category]):   # flower_a_testing_1, flower_a_testing_2...
            bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, index, # index: 图片编号
                                                  category, jpeg_data_tensor, bottleneck_tensor)
            ground_truth = np.zeros(n_classes, np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)

    return bottlenecks, ground_truths

def main(argv):
    # 读取所有图片
    image_lists = create_image_lists(TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
    n_classes = len(image_lists.keys())

    # 读取inception-v3模型
    # 保存了每一个节点的计算方法及取值
    with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # 加载模型，返回结果对应的tensor
    bottlenect_tensor, jpeg_data_tensor = tf.import_graph_def(
        graph_def, return_elements=[BOTTLENECT_TENSOR_NAME, JPEG_DATA_TENSOR_NAME]
    )

    # 定义新的神经网络的input
    # input即为图片经前向传播到瓶颈层的节点取值
    bottlenect_input = tf.placeholder(tf.float32, [None, BOTTLENECT_TENSOR_SIZE], name='BottlenectInput')
    ground_truth_input = tf.placeholder(tf.float32, [None, n_classes], name='GroundTruthInput')

    # 定义全连接层解决分类问题
    with tf.name_scope('final_training_ops'):
        weights = tf.Variable(tf.truncated_normal([BOTTLENECT_TENSOR_SIZE, n_classes], stddev=0.001))
        biases = tf.Variable(tf.zeros([n_classes]))
        logits = tf.matmul(bottlenect_input, weights) + biases
        softmax = tf.nn.softmax(logits)

    # 损失函数
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth_input)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)

    # 计算正确率
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(softmax, 1),
                                      tf.argmax(ground_truth_input, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        # train
        for i in range(STEPS):
            # 获取batch数据
            train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(
                sess, n_classes, image_lists, BATCH,
                'training', jpeg_data_tensor, bottlenect_tensor
            )

            sess.run(train_step,
                     feed_dict={bottlenect_input: train_bottlenecks,
                                ground_truth_input: train_ground_truth})

            # 在验证集上测试正确率
            if i % 100 == 0 or i + 1 == STEPS:
                validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(
                    sess, n_classes, image_lists, BATCH,
                    'validation', jpeg_data_tensor, bottlenect_tensor
                )
                validation_accuracy = sess.run(evaluation_step,
                                                feed_dict={bottlenect_input: validation_bottlenecks,
                                                ground_truth_input: validation_ground_truth})
                print('Step %s: Validation accuracy is %d examples = %.1f%%' % (i, BATCH, validation_accuracy * 100))

        # 在测试数据上测试
        test_bottlenecks, test_ground_truth = get_test_bottlenecks(sess, image_lists, n_classes,
                                                                   jpeg_data_tensor, bottlenect_tensor)
        test_accuracy = sess.run(evaluation_step,
                                 feed_dict={bottlenect_input: test_bottlenecks,
                                            ground_truth_input: test_ground_truth})
        print('Final test accuracy = %.lf%%' % (test_accuracy * 100))


if __name__ == '__main__':
    tf.app.run()


