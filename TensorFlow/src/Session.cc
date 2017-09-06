/*
 * TensorFlow session interface.
 * Based on TensorFlow C API 1.1.
 * For more info, see https://gitlab.cern.ch/mrieger/CMSSW-DNN.
 *
 * Author: Marcel Rieger
 */

#include "DNN/TensorFlow/interface/Session.h"

namespace tf
{

Session::Session()
    : prepared_(false)
    , tf_session_(nullptr)
    , graph_(nullptr)
{
}

Session::Session(Graph* graph, bool threads)
    : prepared_(false)
    , tf_session_(nullptr)
    , graph_(nullptr)
{
    init(graph, threads);
}

Session::~Session()
{
    reset();
}

void Session::init(Graph* graph, bool threads)
{
    reset();

    // status to check tensorflow calls
    TF_Status* status = TF_NewStatus();

    // session options
    TF_SessionOptions* tf_sessionOptions = TF_NewSessionOptions();

    // disable multi-threading when threads is false
    if (!threads) {
        // serialized protobuf representation of
        // intra_op_parallelism_threads = 1 (index 2)
        // inter_op_parallelism_threads = 1 (index 5)
        const char opts[] = "\020\001\050\001";
        TF_SetConfig(tf_sessionOptions, opts, 4, status);
        if (TF_GetCode(status) != TF_OK)
        {
            throw cms::Exception("InvalidSessionOptions") << "error while setting session options: "
                << TF_Message(status);
        }
    }

    // create the tensorflow session object
    tf_session_ = TF_NewSession(graph->getTFGraph(), tf_sessionOptions, status);
    if (TF_GetCode(status) != TF_OK)
    {
        throw cms::Exception("InvalidSession") << "error while creating new session: "
            << TF_Message(status);
    }

    // initialize all variables
    TF_Operation* tf_initOp = graph->getTFOperation("init");
    TF_SessionRun(
        tf_session_,
        nullptr, // run options
        nullptr, nullptr, 0, // inputs
        nullptr, nullptr, 0, // outputs
        &tf_initOp, 1, // target ops, number of targets
        nullptr, // run metadata
        status);
    if (TF_GetCode(status) != TF_OK)
    {
        throw cms::Exception("InvalidSession") << "error while initializing variables: "
            << TF_Message(status);
    }

    // store the graph pointer
    graph_ = graph;

    // some cleanup
    TF_DeleteSessionOptions(tf_sessionOptions);
    TF_DeleteStatus(status);
}

void Session::reset()
{
    prepared_ = false;

    // clear all inputs
    while (nInputs() > 0)
    {
        removeInput(inputs_[0]);
    }

    // clear all outputs
    while (nOutputs() > 0)
    {
        removeOutput(outputs_[0]);
    }

    // close and delete the session object
    if (tf_session_)
    {
        TF_Status* status = TF_NewStatus();

        TF_CloseSession(tf_session_, status);
        if (TF_GetCode(status) != TF_OK)
        {
            throw cms::Exception("InvalidSession") << "error while closing session: "
                << TF_Message(status);
        }

        TF_DeleteSession(tf_session_, status);
        if (TF_GetCode(status) != TF_OK)
        {
            throw cms::Exception("InvalidSession") << "error while deleting session: "
                << TF_Message(status);
        }
        tf_session_ = nullptr;

        TF_DeleteStatus(status);
    }

    // reset the default graph, do not delete it
    graph_ = nullptr;
}

IO* Session::createIO(Tensor* tensor, const std::string& opName, int opIndex) const
{
    // session must not be empty
    if (empty())
    {
        throw cms::Exception("InvalidSession")
            << "cannot create IO object for uninitialized session";
    }

    // get a pointer to the associated operation
    TF_Operation* operation = graph_->getTFOperation(opName);
    if (!operation)
    {
        throw cms::Exception("InvalidOperation") << "no such operation in graph: " << opName;
    }

    // create the IO object
    IO* io = new IO(tensor, operation, opName, opIndex);

    return io;
}

IO* Session::addInput(Tensor* tensor, const std::string& opName, int opIndex)
{
    // the tensor must be initialized
    if (tensor->empty())
    {
        throw cms::Exception("InvalidTensor")
            << "cannot create input using uninitialized tensor for operation: " << opName << ":"
            << opIndex;
    }

    // check for duplicate
    if (hasInput(tensor, opName, opIndex))
    {
        throw cms::Exception("InvalidInput") << "duplicate input tensor for operation: "
            << opName << ":" << opIndex;
    }

    // create the input object
    IO* input = createIO(tensor, opName, opIndex);

    // store it
    inputs_.push_back(input);
    prepared_ = false;

    return input;
}

IO* Session::addOutput(Tensor* tensor, const std::string& opName, int opIndex)
{
    // check for duplicates
    if (hasOutput(tensor, opName, opIndex))
    {
        throw cms::Exception("InvalidOutput") << "duplicate output tensor for operation: "
            << opName << ":" << opIndex;
    }

    // create the output object
    IO* output = createIO(tensor, opName, opIndex);

    // store it
    outputs_.push_back(output);
    prepared_ = false;

    return output;
}

void Session::removeInput(Tensor* tensor, const std::string& opName, int opIndex)
{
    for (size_t i = 0, n = nInputs(); i < n; i++)
    {
        if (inputs_[i]->getTensor() == tensor && inputs_[i]->getOpName() == opName
            && inputs_[i]->getOpIndex() == opIndex)
        {
            delete inputs_[i];
            inputs_.erase(inputs_.begin() + i);
            prepared_ = false;
            break;
        }
    }
}

void Session::removeInput(IO* input)
{
    IOs::iterator it = std::find(inputs_.begin(), inputs_.end(), input);
    if (it != inputs_.end())
    {
        delete *it;
        inputs_.erase(it);
        prepared_ = false;
    }
}

void Session::removeOutput(Tensor* tensor, const std::string& opName, int opIndex)
{
    for (size_t i = 0, n = nOutputs(); i < n; i++)
    {
        if (outputs_[i]->getTensor() == tensor && outputs_[i]->getOpName() == opName
            && outputs_[i]->getOpIndex() == opIndex)
        {
            delete outputs_[i];
            outputs_.erase(outputs_.begin() + i);
            prepared_ = false;
            break;
        }
    }
}

void Session::removeOutput(IO* output)
{
    IOs::iterator it = std::find(outputs_.begin(), outputs_.end(), output);
    if (it != outputs_.end())
    {
        delete *it;
        outputs_.erase(it);
        prepared_ = false;
    }
}

bool Session::hasInput(IO* input) const
{
    return std::find(inputs_.begin(), inputs_.end(), input) != inputs_.end();
}

bool Session::hasOutput(IO* output) const
{
    return std::find(outputs_.begin(), outputs_.end(), output) != outputs_.end();
}

bool Session::hasInput(Tensor* tensor, const std::string& opName, int opIndex) const
{
    for (size_t i = 0, n = nInputs(); i < n; i++)
    {
        if (inputs_[i]->getTensor() == tensor && inputs_[i]->getOpName() == opName
            && inputs_[i]->getOpIndex() == opIndex)
        {
            return true;
        }
    }
    return false;
}

bool Session::hasOutput(Tensor* tensor, const std::string& opName, int opIndex) const
{
    for (size_t i = 0, n = nOutputs(); i < n; i++)
    {
        if (outputs_[i]->getTensor() == tensor && outputs_[i]->getOpName() == opName
            && outputs_[i]->getOpIndex() == opIndex)
        {
            return true;
        }
    }
    return false;
}

void Session::run()
{
    if (empty())
    {
        throw cms::Exception("InvalidSession") << "cannot run uninitialized session";
    }

    // prepare the cache objects
    prepare();

    size_t nIn = nInputs();
    size_t nOut = nOutputs();

    // reset previous outputs
    outputTensors_.clear();
    outputTensors_.resize(nOut, nullptr);
    for (size_t i = 0; i < nOut; i++)
    {
        outputs_[i]->getTensor()->reset();
    }

    // actual run call
    TF_Status* status = TF_NewStatus();
    TF_SessionRun(
        tf_session_,
        nullptr, // run options
        nIn == 0 ? nullptr : &inputOutputs_[0], nIn == 0 ? nullptr : &inputTensors_[0], nIn,
        nOut == 0 ? nullptr : &outputOutputs_[0], nOut == 0 ? nullptr : &outputTensors_[0], nOut,
        nullptr, 0, // target ops, number of targets
        nullptr, // run metadata
        status);

    // check the status
    if (TF_GetCode(status) != TF_OK)
    {
        throw cms::Exception("InvalidSession") << "error during stateful evaluation: "
            << (TF_Message(status));
    }

    // sync outputs
    for (size_t i = 0; i < nOut; i++)
    {
        outputs_[i]->getTensor()->init(outputTensors_[i]);
    }

    // cleanup
    TF_DeleteStatus(status);
}

void Session::run(const IOs inputs, const IOs& outputs) const
{
    if (empty())
    {
        throw cms::Exception("InvalidSession") << "cannot run uninitialized session";
    }

    // objects that will be passed to TF_SessionRun
    std::vector<TF_Output> inputOutputs;
    std::vector<TF_Output> outputOutputs;
    std::vector<TF_Tensor*> inputTensors;
    std::vector<TF_Tensor*> outputTensors;

    // fill input objects
    size_t nIn = inputs.size();
    for (size_t i = 0; i < nIn; i++) {
        // set the input TF_Output object and the input tensor
        inputOutputs.push_back(inputs[i]->getTFOutput());
        inputTensors.push_back(inputs[i]->getTensor()->getTFTensor());
    }

    // fill output objects
    size_t nOut = outputs.size();
    outputTensors.resize(nOut, nullptr);
    for (size_t i = 0; i < nOut; i++)
    {
        // set the output TF_Output object
        outputOutputs.push_back(outputs[i]->getTFOutput());

        // reset output tensors
        outputs[i]->getTensor()->reset();
    }

    // actual run call
    TF_Status* status = TF_NewStatus();
    TF_SessionRun(
        tf_session_,
        nullptr, // run options
        nIn == 0 ? nullptr : &inputOutputs[0], nIn == 0 ? nullptr : &inputTensors[0], nIn,
        nOut == 0 ? nullptr : &outputOutputs[0], nOut == 0 ? nullptr : &outputTensors[0], nOut,
        nullptr, 0, // target ops, number of targets
        nullptr, // run metadata
        status);

    // check the status
    if (TF_GetCode(status) != TF_OK)
    {
        throw cms::Exception("InvalidSession") << "error during stateless evaluation: "
            << (TF_Message(status));
    }

    // sync outputs
    for (size_t i = 0; i < nOut; i++)
    {
        outputs[i]->getTensor()->init(outputTensors[i]);
    }

    // cleanup
    TF_DeleteStatus(status);
}

void Session::prepare()
{
    if (prepared_)
    {
        return;
    }

    // reset input objects and tensors
    inputOutputs_.clear();
    inputTensors_.clear();
    for (size_t i = 0, n = nInputs(); i < n; i++) {
        inputOutputs_.push_back(inputs_[i]->getTFOutput());
        inputTensors_.push_back(inputs_[i]->getTensor()->getTFTensor());
    }

    // reset output objects
    outputOutputs_.clear();
    for (size_t i = 0, n = nOutputs(); i < n; i++)
    {
        outputOutputs_.push_back(outputs_[i]->getTFOutput());
    }

    prepared_ = true;
}

} // namespace tf
