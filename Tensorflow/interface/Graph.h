/*
 * Generic Tensorflow graph representation.
 *
 * Author:
 *   Marcel Rieger
 */

#ifndef DNN_TENSORFLOW_GRAPH_H
#define DNN_TENSORFLOW_GRAPH_H

#include <fstream>
#include <map>
#include <stdexcept>
#include <string>

#include "Python.h"

#include "DNN/Base/interface/Logger.h"
#include "DNN/Base/interface/PythonInterface.h"
#include "DNN/Tensorflow/interface/Tensor.h"

namespace dnn
{

namespace tf
{

class Graph
{
public:
    Graph();
    Graph(const std::string& filename);
    Graph(LogLevel logLevel);
    Graph(const std::string& filename, LogLevel logLevel);
    virtual ~Graph();

    Tensor* defineInput(Tensor* tensor);
    Tensor* defineOutput(Tensor* tensor);

    void removeInput(const std::string& name);
    void removeOutput(const std::string& name);

    inline bool hasInput(const std::string& name) const;
    inline bool hasOutput(const std::string& name) const;

    Tensor* getInput(const std::string& name);
    Tensor* getOutput(const std::string& name);

    void load(const std::string& filename);
    void eval();

    PythonInterface& getPythonInterface();

private:
    void init(const std::string& filename);

    Logger logger;
    PythonInterface python;

    std::map<std::string, Tensor*> inputs;
    std::map<std::string, Tensor*> outputs;

    PyObject* pyInputs;
    PyObject* pyOutputs;
    PyObject* pyEvalSession;
};

static std::string embeddedTensorflowScript = "\
import os, sys, numpy as np\n\
tf = sess = saver = None\n\
\n\
def insert_path(path):\n\
    path = os.path.expandvars(os.path.expanduser(path))\n\
    sys.path.append(path)\n\
\n\
def import_tf():\n\
    global tf\n\
    import tensorflow as tf\n\
\n\
def start_session():\n\
    global sess\n\
    sess = tf.Session()\n\
\n\
def load_graph(path):\n\
    global saver\n\
    path = os.path.expandvars(os.path.expanduser(path))\n\
    saver = tf.train.import_meta_graph(path + '.meta')\n\
    saver.restore(sess, path)\n\
\n\
def eval_session(inputs, outputs):\n\
    outputs.update(sess.run(dict(zip(outputs.keys(), outputs.keys())), feed_dict=inputs))\n\
";

} // namepace tf

} // namepace dnn

#endif // DNN_TENSORFLOW_GRAPH_H
