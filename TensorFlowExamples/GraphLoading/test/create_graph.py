# -*- coding: utf-8 -*-

"""
Script that creates and stores a minimal tensorflow graph.
"""


import os
import sys

import tensorflow as tf


def create_graph(graph_path):
    x_ = tf.placeholder(tf.float32, [None, 10], name="input")

    W = tf.Variable(tf.ones([10, 1]))
    b = tf.Variable(tf.ones([1]))
    tf.add(tf.matmul(x_, W), b, name="output")

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    outputs = ["output"]
    constant_graph = tf.graph_util.convert_variables_to_constants(
        sess, sess.graph.as_graph_def(), outputs)
    tf.train.write_graph(constant_graph, *os.path.split(graph_path), as_text=False)

    sess.close()


if __name__ == "__main__":
    if len(sys.argv) == 2:
        graph_path = sys.argv[1]
    else:
        graph_path = os.path.abspath("graph.pb")

    if os.path.exists(graph_path):
        os.remove(graph_path)

    print("create tensorflow graph at {}".format(graph_path))
    create_graph(graph_path)
