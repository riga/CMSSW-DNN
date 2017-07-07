/*
 * Simple performance test of lwtnn with the same graph as used in test_tfperf.cc.
 *
 * Usage:
 *   > test_lwtnnperf
 *
 * Author:
 *   Marcel Rieger
 */

#include <iostream>
#include <string>
#include <vector>
#include <limits>
#include <fstream>
#include <sys/time.h>

#include "lwtnn/LightweightNeuralNetwork.hh"
#include "lwtnn/parse_json.hh"

long int getTime(timeval* t)
{
    gettimeofday(t, NULL);
    return t->tv_sec * 1000 + t->tv_usec / 1000;
}

int main(int argc, char* argv[])
{
    std::cout << std::endl << "test lwtnn performance" << std::endl;

    struct timeval t;

    // get the json file containing the model
    std::string cmsswBase = std::string(getenv("CMSSW_BASE"));
    std::string dataDir = cmsswBase + "/src/DNN/Tensorflow/bin/data";
    std::string jsonFile = dataDir + "/lwtnn_5.json";
    std::cout << "load json file " << jsonFile << std::endl;

    // setup the lwt nn
    std::ifstream inputStream(jsonFile);
    lwt::JSONConfig config = lwt::parse_json(inputStream);
    lwt::LightweightNeuralNetwork nn(config.inputs, config.layers, config.outputs);

    // do the testing only with a single batch (batching is actually not documented!?)
    int n = 10000;
    std::cout << "run " << n << " evaluations for batch size 1" << std::endl;

    // update inputs
    std::map<std::string, double> inValues;
    inValues["input_0"] = 1.0;

    // prepare output
    std::map<std::string, double> outValues;

    // actual testing
    long int t0 = getTime(&t);
    for (int i = 0; i < n; i++)
    {
        inValues["input_0"] = 1.0;
        outValues = nn.compute(inValues);
    }
    long int t1 = getTime(&t);
    std::cout << "-> " << (t1 - t0) / (float)n << " ms per batch" << std::endl << std::endl;

    std::cout << std::endl << "done" << std::endl;

    return 0;
}
