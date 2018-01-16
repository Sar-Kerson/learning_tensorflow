# coding=utf-8
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt


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

if __name__ == '__main__':
    img_raw_data = tf.gfile.FastGFile('/home/sar/SarKerson/dehaze/darkchannel_prior/dehazeProcessor/input/sea_7.png','r').read()
    with tf.Session() as sess:
        img_data = tf.image.decode_png(img_raw_data)
        boxes = tf.constant([[[0.3, 0.3, 0.6, 0.6]]])

        for i in range(6):
            result = preprocess(image=img_data, height=200, width=200, box=boxes)
            plt.imshow(result.eval())
            plt.show()