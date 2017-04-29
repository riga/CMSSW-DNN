/*
 * Generic Tensorflow graph representation.
 *
 * Author:
 *   Marcel Rieger
 */

#ifndef DNN_TENSORFLOW_GRAPH_H
#define DNN_TENSORFLOW_GRAPH_H

#include <fstream>
#include <stdexcept>
#include <string>

#include "DNN/Base/interface/Logger.h"
#include "DNN/Base/interface/PythonInterface.h"

namespace DNN
{

class TensorflowGraph
{
public:
    TensorflowGraph();
    TensorflowGraph(const std::string& filename);
    TensorflowGraph(LogLevel logLevel);
    TensorflowGraph(const std::string& filename, LogLevel logLevel);
    virtual ~TensorflowGraph();

    // void load(const std::string& filename);
    void load(std::string filename);
    void defineInputs(const std::vector<std::string>& inputs);
    void defineOutputs(const std::vector<std::string>& outputs);
    void startSession();
    int call();

    PythonInterface getPythonInterface();

private:
    void init(const std::string& filename);

    Logger logger;
    PythonInterface python;

    size_t nInputs;
    size_t nOutputs;
};

static std::string embeddedTensorflowScript = "\
import os, sys, numpy as np\n\
tf = saver = graph = sess = inputs = outputs = None\n\
\n\
def insert_path(path):\n\
    path = os.path.expandvars(os.path.expanduser(path))\n\
    sys.path.append(path)\n\
\n\
def import_tf():\n\
    global tf\n\
    import tensorflow as tf\n\
\n\
def load_graph(path):\n\
    global saver, graph\n\
    path = os.path.expandvars(os.path.expanduser(path))\n\
    saver = tf.train.import_meta_graph(path)\n\
    graph = tf.get_default_graph()\n\
\n\
def define_inputs(*names):\n\
    global inputs\n\
    inputs = [graph.get_tensor_by_name(name) for name in names]\n\
\n\
def define_outputs(*names):\n\
    global outputs\n\
    outputs = [graph.get_tensor_by_name(name) for name in names]\n\
\n\
def start_session():\n\
    global sess\n\
    sess = tf.Session()\n\
    sess.run(tf.global_variables_initializer())\n\
\n\
def call(*values):\n\
    results = sess.run(inputs, feed_dict=dict(zip(inputs, values)))\n\
    return len(results)\n\
";

} // namepace DNN

#endif // DNN_TENSORFLOW_GRAPH_H
