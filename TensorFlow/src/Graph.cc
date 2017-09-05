/*
 * TensorFlow graph interface.
 * Based on TensorFlow C API 1.1.
 * For more info, see https://gitlab.cern.ch/mrieger/CMSSW-DNN.
 *
 * Author: Marcel Rieger
 */

#include "DNN/TensorFlow/interface/Graph.h"

namespace tf
{

Graph::Graph()
    : tf_graph_(nullptr)
{
}

Graph::Graph(const std::string& filename, const std::string& tag)
    : tf_graph_(nullptr)
{
    // initialize the graph
    init(filename, tag);
}

Graph::~Graph()
{
    reset();
}

void Graph::init(const std::string& exportDir, const std::string& tag)
{
    reset();

    // disable tensorflow logging by default
    setenv("TF_CPP_MIN_LOG_LEVEL", "3", 0);

    // status to check TF calls
    TF_Status* status = TF_NewStatus();

    // graph tags
    const char* tags[] = { tag.c_str() };

    // session options
    TF_SessionOptions* tf_sessionOptions = TF_NewSessionOptions();

    // initialize an empty graph
    tf_graph_ = TF_NewGraph();

    // load the SavedModel into tf_graph_
    // we need a dummy session as there is currently no other way to load a graph
    TF_Session* tf_session = TF_LoadSessionFromSavedModel(
        tf_sessionOptions, nullptr, exportDir.c_str(), tags, 1, tf_graph_, nullptr, status);
    if (TF_GetCode(status) != TF_OK)
    {
        throw cms::Exception("InvalidGraph") << "error while loading graph from SavedModel: "
            << TF_Message(status);
    }

    // close and delete the session again
    TF_CloseSession(tf_session, status);
    if (TF_GetCode(status) != TF_OK)
    {
        throw cms::Exception("InvalidGraph") << "error while closing dummy session: "
            << TF_Message(status);
    }

    TF_DeleteSession(tf_session, status);
    if (TF_GetCode(status) != TF_OK)
    {
        throw cms::Exception("InvalidGraph") << "error while deleting dummy session: "
            << TF_Message(status);
    }

    // some cleanup
    TF_DeleteStatus(status);
    TF_DeleteSessionOptions(tf_sessionOptions);
}

void Graph::reset()
{
    // delete the graph object
    if (tf_graph_)
    {
        TF_DeleteGraph(tf_graph_);
        tf_graph_ = nullptr;
    }
}

TF_Operation* Graph::getTFOperation(const std::string& name)
{
    if (empty())
    {
        throw cms::Exception("InvalidGraph") << "cannot find operation in uninitialized graph";
    }

    return TF_GraphOperationByName(tf_graph_, name.c_str());
}

} // namespace tf
