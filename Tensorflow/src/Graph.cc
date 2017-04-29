/*
 * Generic Tensorflow graph representation.
 *
 * Author:
 *   Marcel Rieger
 */

#include "DNN/Tensorflow/interface/Graph.h"

namespace DNN
{

TensorflowGraph::TensorflowGraph()
    : logger(Logger("TensorflowGraph"))
    , python()
    , nInputs(0)
    , nOutputs(0)
{
    init("");
}

TensorflowGraph::TensorflowGraph(const std::string& filename)
    : logger(Logger("TensorflowGraph"))
    , python()
    , nInputs(0)
    , nOutputs(0)
{
    init(filename);
}

TensorflowGraph::TensorflowGraph(LogLevel logLevel)
    : logger(Logger("TensorflowGraph", logLevel))
    , python(PythonInterface(logLevel))
    , nInputs(0)
    , nOutputs(0)
{
    init("");
}

TensorflowGraph::TensorflowGraph(const std::string& filename, LogLevel logLevel)
    : logger(Logger("TensorflowGraph", logLevel))
    , python(PythonInterface(logLevel))
    , nInputs(0)
    , nOutputs(0)
{
    init(filename);
}

TensorflowGraph::~TensorflowGraph()
{
}

void TensorflowGraph::init(const std::string& filename)
{
    python.runScript(embeddedTensorflowScript);

    // update the python path to find tensorflow
    std::string cmsswBase = std::string(getenv("CMSSW_BASE"));
    std::string pythonPath = cmsswBase + "/python/DNN/Tensorflow";
    PyObject* result1 = python.call("insert_path", pythonPath);
    python.release(result1);

    // import tensorflow
    PyObject* result2 = python.call("import_tf");
    python.release(result2);

    if (!filename.empty())
    {
        load(filename);
    }
}


void TensorflowGraph::load(std::string filename)
{
    PyObject* result = python.call("load_graph", filename);
    python.release(result);
}

void TensorflowGraph::defineInputs(const std::vector<std::string>& inputs)
{
    PyObject* args = python.createTuple(inputs);
    PyObject* result = python.call("define_inputs", args);
    python.release(result);
    python.release(args);

    nInputs = inputs.size();
}

void TensorflowGraph::defineOutputs(const std::vector<std::string>& outputs)
{
    PyObject* args = python.createTuple(outputs);
    PyObject* result = python.call("define_outputs", args);
    python.release(result);
    python.release(args);

    nOutputs = outputs.size();
}

void TensorflowGraph::startSession()
{
    PyObject* result = python.call("start_session");
    python.release(result);
}

int TensorflowGraph::call()
{
    return 27;
}

PythonInterface TensorflowGraph::getPythonInterface()
{
    return python;
}

} // namespace DNN
