# -*- coding: utf-8 -*-

import os
import tensorflow as tf

thisdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(os.path.dirname(thisdir), "data")

x_ = tf.placeholder(tf.float32, [None, 100], name="input")

W0 = tf.Variable(tf.random_normal([100, 200]))
b0 = tf.Variable(tf.random_normal([200]))
h0 = tf.nn.elu(tf.matmul(x_, W0) + b0)

W1 = tf.Variable(tf.random_normal([200, 200]))
b1 = tf.Variable(tf.random_normal([200]))
h1 = tf.nn.elu(tf.matmul(h0, W1) + b1)

W2 = tf.Variable(tf.random_normal([200, 200]))
b2 = tf.Variable(tf.random_normal([200]))
h2 = tf.nn.elu(tf.matmul(h1, W2) + b2)

W3 = tf.Variable(tf.random_normal([200, 200]))
b3 = tf.Variable(tf.random_normal([200]))
h3 = tf.nn.elu(tf.matmul(h2, W3) + b3)

W4 = tf.Variable(tf.random_normal([200, 200]))
b4 = tf.Variable(tf.random_normal([200]))
h4  = tf.nn.elu(tf.matmul(h3, W4) + b4)

W5 = tf.Variable(tf.random_normal([200, 10]))
b5 = tf.Variable(tf.random_normal([10]))
y  = tf.nn.softmax(tf.matmul(h4, W5) + b5, name="output")

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(y, feed_dict={x_: [range(100)]})[0])

saver = tf.train.Saver()
saver.save(sess, os.path.join(datadir, "largegraph"))
