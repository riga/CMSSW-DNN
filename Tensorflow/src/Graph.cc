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
    : python(0)
    , nInputs(0)
    , nOutputs(0)
{
    python = new PythonInterface();
    python->runScript(embeddedTensorflowScript);
}

TensorflowGraph::~TensorflowGraph()
{
    if (python)
    {
        delete python;
    }
}

void TensorflowGraph::defineInputs(const std::vector<std::string>& inputs)
{
}

void TensorflowGraph::defineOutputs(const std::vector<std::string>& outputs)
{
}

} // namespace DNN
