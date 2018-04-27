/*
 * Example module that shows how to load and evaluate a constant TensorFlow graph in a CMSSW plugin.
 * If you are aiming to use the TensorFlow interface in your personal plugin (!), make sure to
 * include the following lines in your /plugins/BuildFile.xml:
 *
 *     <use name="PhysicsTools/TensorFlow" />
 *
 * Important: If you are using the TensorFlow interface in a file in the /src/ or /interface/
 * directory of your module, make sure to create a (global) /BuildFile.xml containing (at least):
 *
 *     <use name="PhysicsTools/TensorFlow" />
 *     <export>
 *         <lib name="1" />
 *     </export>
 *
 * Author: Marcel Rieger <marcel.rieger@cern.ch>
 */

#include <memory>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

class GraphLoading : public edm::one::EDAnalyzer<>
{
public:
    explicit GraphLoading(const edm::ParameterSet&);
    ~GraphLoading();

private:
    void beginJob();
    void analyze(const edm::Event&, const edm::EventSetup&);
    void endJob();

    std::string graphPath_;
    tensorflow::GraphDef* graphDef_;
    tensorflow::Session* session_;
};

GraphLoading::GraphLoading(const edm::ParameterSet& config)
    : graphPath_(config.getParameter<std::string>("graphPath"))
    , graphDef_(nullptr)
    , session_(nullptr)
{
    // show tf debug logs
    tensorflow::setLogging("0");
}

GraphLoading::~GraphLoading()
{
}

void GraphLoading::beginJob()
{
    // load the graph
    std::cout << "loading graph from " << graphPath_ << std::endl;
    graphDef_ = tensorflow::loadGraphDef(graphPath_);

    // create a new session and add the graphDef
    session_ = tensorflow::createSession(graphDef_);
}

void GraphLoading::endJob()
{
    // close the session
    tensorflow::closeSession(session_);
    session_ = nullptr;

    // delete the graph
    delete graphDef_;
    graphDef_ = nullptr;
}

void GraphLoading::analyze(const edm::Event& event, const edm::EventSetup& setup)
{
    // define a tensor and fill it with range(10)
    tensorflow::Tensor input(tensorflow::DT_FLOAT, { 1, 10 });
    float* d = input.flat<float>().data();
    for (float i = 0; i < 10; i++, d++)
    {
        *d = i;
    }

    // define the output and run
    std::cout << "session.run" << std::endl;
    std::vector<tensorflow::Tensor> outputs;
    tensorflow::run(session_, { { "input", input } }, { "output" }, &outputs);

    // check and print the output
    std::cout << " -> " << outputs[0].matrix<float>()(0, 0) << std::endl << std::endl;
}

DEFINE_FWK_MODULE(GraphLoading);
