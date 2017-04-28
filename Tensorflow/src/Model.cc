/*
 * Generic Tensorflow model.
 *
 * Author:
 *   Marcel Rieger
 */

#include "DNN/Tensorflow/interface/Model.h"

namespace DNN
{

Model::Model()
    : python(0)
    , nInputs(0)
    , nOutputs(0)
{
    python = new PythonInterface();
    python->runScript(embeddedTensorflowScript);
}

Model::~Model()
{
    if (python)
    {
        delete python;
    }
}

void Model::defineInputs(const std::vector<std::string>& inputs)
{

}

void Model::defineOutputs(const std::vector<std::string>& outputs)
{
    
}

} // namespace DNN
