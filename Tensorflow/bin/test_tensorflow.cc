/*
 * Simple test of the tensorflow interface.
 *
 * Graph (from data/simplegraph.pb, created by test/creategraph.py):
 *   Input:
 *     - name = "input:0", shape = (batch, 10)
 *   Output:
 *     - name = "output:0", shape = (batch, 1)
 *
 * Usage:
 *   > test_tensorflow
 *
 * Author:
 *   Marcel Rieger
 */

#include <iostream>
#include <string>
#include <vector>

#include "DNN/Tensorflow/interface/Graph.h"

int main(int argc, char* argv[])
{
    std::cout << "tensorflow test" << std::endl;

    // get the file containing the graph
    std::string cmsswBase = std::string(getenv("CMSSW_BASE"));
    std::string dataDir = cmsswBase + "/src/DNN/Tensorflow/data";
    std::string graphFile = dataDir + "/simplegraph.pb";
    std::cout << "load graph " << graphFile << std::endl;

    // load and initialize graph
    DNN::TensorflowGraph g(graphFile, DNN::LogLevel::ALL);
    g.defineInputs({"input:0"});
    g.defineOutputs({"output:0"});
    g.startSession();

    // evaluation calls
    // TODO

    return 0;
}
