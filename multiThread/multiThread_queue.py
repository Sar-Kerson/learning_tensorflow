# coding=utf-8
import tensorflow as tf
import numpy as np
import threading
import time

def test_que():
    q = tf.FIFOQueue(2, "int32")
    init = q.enqueue_many(([0, 10],))

    x = q.dequeue()

    y = x + 1

    q_inc = q.enqueue([y])

    with tf.Session() as sess:
        sess.run(init)
        for _ in range(5):
            v, _ = sess.run([x, q_inc])
            print v

def test_coordinate():
    def MyLoop(coord, worker_id):
        # 判断当前线程是否需要停止
        while not coord.should_stop():
            # 随机停止所有线程
            if np.random.rand() < 0.1:
                print("Stop from id: %d\n" % worker_id)
                coord.request_stop()
            else:
                print("Work on id: %d\n" % worker_id)
            time.sleep(1)

    coord = tf.train.Coordinator()

    threads = [threading.Thread(target=MyLoop, args=(coord, i)) for i in xrange(5)]

    # 启动所有线程
    for t in threads:
        t.start()

    coord.join(threads)

def test_queRunner():
    # 创建float型的100个元素的队列
    queue = tf.FIFOQueue(100, "float")

    #定义操作
    enqueue_op = queue.enqueue(tf.random_normal([1]))

    # 需要启动5个线程，均执行enqueue_op操作
    qr = tf.train.QueueRunner(queue, [enqueue_op] * 5)
    tf.train.add_queue_runner(qr)

    out_tensor = queue.dequeue()

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for _ in range(3):
            print(sess.run(out_tensor))

        coord.request_stop()
        coord.join(threads)

test_queRunner()