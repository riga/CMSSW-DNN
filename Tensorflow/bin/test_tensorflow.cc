/*
 * Simple test of the tensorflow interface.
 *
 * Usage:
 *   > test_tensorflow
 *
 * Author:
 *   Marcel Rieger
 */

#include <iostream>

#include "DNN/Tensorflow/interface/Graph.h"

int main(int argc, char* argv[])
{
    std::cout << "tensorflow test" << std::endl;

    std::string cmsswBase = std::string(getenv("CMSSW_BASE"));
    std::string dataDir = cmsswBase + "/src/DNN/Tensorflow/data";
    std::string graphFile = dataDir + "/simplegraph.pb";
    std::cout << "load graph " << graphFile << std::endl;

    DNN::TensorflowGraph g();


    return 0;
}
