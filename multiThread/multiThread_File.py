# coding=utf-8

import tensorflow as tf
import numpy as np


def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image=image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image=image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image=image, max_delta=0.2)
        image = tf.image.random_contrast(image=image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_saturation(image=image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image=image, max_delta=32. / 255.)
        image = tf.image.random_hue(image=image, max_delta=0.2)
        image = tf.image.random_contrast(image=image, lower=0.5, upper=1.5)
    elif color_ordering == 2:
        image = tf.image.random_saturation(image=image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image=image, max_delta=0.2)
        image = tf.image.random_brightness(image=image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image=image, lower=0.5, upper=1.5)
    else:
        image = tf.image.random_saturation(image=image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image=image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image=image, max_delta=32. / 255.)
        image = tf.image.random_hue(image=image, max_delta=0.2)
    return tf.clip_by_value(image, 0.0, 1.0)

def preprocess(image, height, width, box):
    if box is None:
        box = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])

    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    box_begin, box_size, _ = tf.image.sample_distorted_bounding_box(tf.shape(image),
                                                                    bounding_boxes=box)
    distorted_img = tf.slice(image, box_begin, box_size)
    distorted_img = tf.image.resize_images(distorted_img, [height, width], method=np.random.randint(4))
    distorted_img = tf.image.random_flip_left_right(distorted_img)
    distorted_img = distort_color(distorted_img, np.random.randint(4))
    return distorted_img



files = tf.train.match_filenames_once('./output-*')
filename_queue = tf.train.string_input_producer(files, shuffle=False)

# image -----  raw_data
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized=serialized_example,
    features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
        'pixel': tf.FixedLenFeature([], tf.int64),
    }
)

images, labels = features['image_raw'], features['label']
pixels = tf.cast(features['pixel'], tf.float32)
col, row, channel = tf.sqrt(pixels), tf.sqrt(pixels), 1

with tf.Session() as sess:
    col, row = sess.run([col, row])

print('row: %d, col: %d, c: %d' % (row, col, channel))

decoded_img = tf.decode_raw(images, tf.uint8)
decoded_img.set_shape([row, col, channel])

distored_img = preprocess(decoded_img, row, col, None)

MIN_AFTER_DEQUEUE = 10000
BATCH_SIZE = 100

CAPACITY = MIN_AFTER_DEQUEUE + 3 * BATCH_SIZE

img_batch, label_batch = tf.train.shuffle_batch([distored_img, labels],
                                                batch_size=BATCH_SIZE,
                                                capacity=CAPACITY,
                                                min_after_dequeue=MIN_AFTER_DEQUEUE)

import reference

print('begin...')

logit = reference.inference(img_batch)

with tf.Session() as sess:
    logit = sess.run(logit)