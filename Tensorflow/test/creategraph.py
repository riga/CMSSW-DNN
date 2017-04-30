# -*- coding: utf-8 -*-

import os
import tensorflow as tf

thisdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(os.path.dirname(thisdir), "data")

x_ = tf.placeholder(tf.float32, [None, 10], name="input")
W = tf.Variable(tf.ones([10, 1]), name="weights")
b = tf.Variable(tf.ones([1]), name="biases")
y = tf.add(tf.matmul(x_, W), b, name="output")

init_op = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init_op)

print(sess.run(y, feed_dict={x_: [range(10)]})[0][0])

saver = tf.train.Saver()
saver.save(sess, os.path.join(datadir, "simplegraph"))
