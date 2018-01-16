#coding=utf-8
import tensorflow as tf

RECORD = '/tmp/TFRecord/output.tfrecords'

reader = tf.TFRecordReader()

# queue to maintain the input file list
filename_queue = tf.train.string_input_producer([RECORD])

# 读入一个example
_, serialized_example = reader.read(filename_queue)
# 解析读入的一个example
features = tf.parse_single_example(
    serialized_example,
    features={
        'pixel': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
    }
)

pixels = tf.decode_raw(features['pixels'], tf.uint8)
labels = tf.cast(features['labels'], tf.int32)
images = tf.decode_raw(features['image_raw'], tf.int32)

sess = tf.Session()

# 启动多线程处理数据
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for i in range(10):
    image, label, pixel = sess.run([images, labels, pixels])