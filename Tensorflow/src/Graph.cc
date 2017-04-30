/*
 * Generic Tensorflow graph representation.
 *
 * Author:
 *   Marcel Rieger
 */

#include "DNN/Tensorflow/interface/Graph.h"

namespace dnn
{

namespace tf
{

Graph::Graph()
    : logger(Logger("tf::Graph"))
    , python()
    , pyInputs(0)
    , pyOutputs(0)
    , pyEvalArgs(0)
    , pyEvalSession(0)
{
    init("");
}

Graph::Graph(const std::string& filename)
    : logger(Logger("tf::Graph"))
    , python()
    , pyInputs(0)
    , pyOutputs(0)
    , pyEvalArgs(0)
    , pyEvalSession(0)
{
    init(filename);
}

Graph::Graph(LogLevel logLevel)
    : logger(Logger("tf::Graph", logLevel))
    , python(PythonInterface(logLevel))
    , pyInputs(0)
    , pyOutputs(0)
    , pyEvalArgs(0)
    , pyEvalSession(0)
{
    init("");
}

Graph::Graph(const std::string& filename, LogLevel logLevel)
    : logger(Logger("tf::Graph", logLevel))
    , python(PythonInterface(logLevel))
    , pyInputs(0)
    , pyOutputs(0)
    , pyEvalArgs(0)
    , pyEvalSession(0)
{
    init(filename);
}

Graph::~Graph()
{
    // cleanup python objects
    python.release(pyInputs);
    python.release(pyOutputs);
    python.release(pyEvalArgs);
}

void Graph::init(const std::string& filename)
{
    logger.debug("initialize graph");
    python.runScript(embeddedTensorflowScript);

    // update the python path to find tensorflow
    std::string cmsswBase = std::string(getenv("CMSSW_BASE"));
    std::string pythonPath = cmsswBase + "/python/DNN/Tensorflow";
    PyObject* result1 = python.call("insert_path", pythonPath);
    python.release(result1);

    // import tensorflow
    PyObject* result2 = python.call("import_tf");
    python.release(result2);

    // create the evaluation arguments
    pyInputs = PyDict_New();
    pyOutputs = PyDict_New();
    pyEvalArgs = PyTuple_New(2);
    PyTuple_SetItem(pyEvalArgs, 0, pyInputs);
    Py_INCREF(pyInputs);
    PyTuple_SetItem(pyEvalArgs, 1, pyOutputs);
    Py_INCREF(pyOutputs);

    if (!filename.empty())
    {
        load(filename);
    }
}

Tensor* Graph::defineInput(Tensor* tensor)
{
    if (tensor)
    {
        removeInput(tensor->getName());
        inputs[tensor->getName()] = tensor;
    }
    return tensor;
}

Tensor* Graph::defineOutput(Tensor* tensor)
{
    if (tensor)
    {
        removeOutput(tensor->getName());
        outputs[tensor->getName()] = tensor;
    }
    return tensor;
}

bool Graph::removeInput(const std::string& name)
{
    if (!hasInput(name))
    {
        return false;
    }
    inputs.erase(name);
    return true;
}

bool Graph::removeOutput(const std::string& name)
{
    if (!hasOutput(name))
    {
        return false;
    }
    outputs.erase(name);
    return true;
}

bool Graph::hasInput(const std::string& name) const
{
    return inputs.find(name) != inputs.end();
}

bool Graph::hasOutput(const std::string& name) const
{
    return outputs.find(name) != outputs.end();
}

Tensor* Graph::getInput(const std::string& name)
{
    if (!hasInput(name))
    {
        return 0;
    }
    return inputs.find(name)->second;
}

Tensor* Graph::getOutput(const std::string& name)
{
    if (!hasOutput(name))
    {
        return 0;
    }
    return outputs.find(name)->second;
}

void Graph::load(const std::string& filename)
{
    logger.info("load graph from " + filename);

    PyObject* result = python.call("start_session");
    python.release(result);

    result = python.call("load_graph", filename);
    python.release(result);

    pyEvalSession = python.get("eval_session");
}

void Graph::buildArgs()
{
    logger.debug("building evaluation args");

    PyDict_Clear(pyInputs);
    PyDict_Clear(pyOutputs);

    for (std::map<std::string, Tensor*>::iterator it = inputs.begin(); it != inputs.end(); it++)
    {
        logger.debug("use array of tensor '" + it->first + "' as input");
        if (it->second->isEmpty())
        {
            throw std::runtime_error("cannot set non-initialized tensor as input");
        }
        PyDict_SetItem(pyInputs, PyString_FromString(it->first.c_str()), it->second->data);
    }

    for (std::map<std::string, Tensor*>::iterator it = outputs.begin(); it != outputs.end(); it++)
    {
        logger.debug("use array of tensor '" + it->first + "' as output");
        PyObject* item = it->second->isEmpty() ? Py_None : it->second->data;
        PyDict_SetItem(pyOutputs, PyString_FromString(it->first.c_str()), item);
    }
}

void Graph::eval()
{
    logger.debug("evaluate");

    if (!pyEvalSession || !pyEvalArgs)
    {
        throw std::runtime_error("cannot eval session, graph not loaded yet");
    }

    if (PyDict_Size(pyOutputs) == 0)
    {
        buildArgs();
    }

    // actual evaluation
    python.call(pyEvalSession, pyEvalArgs);

    // update output tensors
    for (std::map<std::string, Tensor*>::iterator it = outputs.begin(); it != outputs.end(); it++)
    {
        PyObject* data = PyDict_GetItem(pyOutputs, PyString_FromString(it->first.c_str()));
        python.except(data, "evaluation for tensor '" + it->first + "' failed");

        it->second->setArray(data);
    }
}

PythonInterface Graph::getPythonInterface()
{
    return python;
}

} // namespace tf

} // namespace dnn
