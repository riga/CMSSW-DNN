# -*- coding: utf-8 -*-

import os
import tensorflow as tf

thisdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(os.path.dirname(thisdir), "data")

saver = tf.train.import_meta_graph(os.path.join(datadir, "simplegraph.pb"))
graph = tf.get_default_graph()

x_ = graph.get_tensor_by_name("input:0")
y = graph.get_tensor_by_name("output:0")

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(y, feed_dict={x_: [range(10)]})[0][0])
