/*
 * HelloWorld test of the TensorFlow interface.
 * Based on TensorFlow C++ API 1.3.
 * For more info, see https://gitlab.cern.ch/mrieger/CMSSW-DNN.
 *
 * Author: Marcel Rieger
 */

#include <boost/filesystem.hpp>
#include <cppunit/extensions/HelperMacros.h>
#include <stdexcept>

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"

using tensorflow::Status;
using tensorflow::GraphDef;
using tensorflow::Session;
using tensorflow::SessionOptions;
using tensorflow::RunOptions;
using tensorflow::SavedModelBundle;
using tensorflow::Tensor;

class testSession : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(testSession);
    CPPUNIT_TEST(checkAll);
    CPPUNIT_TEST_SUITE_END();

public:
    std::string dataPath;

    void setUp();
    void tearDown();
    void checkAll();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testSession);

void testSession::setUp()
{
    dataPath = std::string(getenv("CMSSW_BASE")) + "/test/" + std::string(getenv("SCRAM_ARCH"))
        + "/" + boost::filesystem::unique_path().string();

    // create the graph
    std::string testPath = std::string(getenv("CMSSW_BASE")) + "/src/DNN/TensorFlow/test";
    std::string cmd = "python " + testPath + "/creategraph.py " + dataPath;
    std::array<char, 128> buffer;
    std::string result;
    std::shared_ptr<FILE> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe)
    {
        throw std::runtime_error("popen() failed!");
    }
    while (!feof(pipe.get()))
    {
        if (fgets(buffer.data(), 128, pipe.get()) != NULL)
        {
            result += buffer.data();
        }
    }
    std::cout << std::endl
              << result << std::endl;
}

void testSession::tearDown()
{
    if (boost::filesystem::exists(dataPath))
    {
        boost::filesystem::remove_all(dataPath);
    }
}

void testSession::checkAll()
{
    std::string modelDir = dataPath + "/simplegraph";

    Status status;
    SessionOptions sessionOptions;
    RunOptions runOptions;
    SavedModelBundle bundle;

    status = LoadSavedModel(sessionOptions, runOptions, modelDir, { "serve" }, &bundle);
    if (!status.ok())
    {
        std::cout << status.ToString() << std::endl;
        return;
    }

    Session* session = bundle.session.release();
    GraphDef graphDef = bundle.meta_graph_def.graph_def();

    Tensor input(tensorflow::DT_FLOAT, { 1, 10 });
    float* d = input.flat<float>().data();
    for (size_t i = 0; i < 10; i++, d++)
    {
        *d = float(i);
    }

    Tensor scale(tensorflow::DT_FLOAT, {});
    scale.scalar<float>()() = 1.0;

    std::vector<tensorflow::Tensor> outputs;

    session->Run({ { "input", input }, { "scale", scale } }, { "output" }, {}, &outputs);

    std::cout << outputs[0].DebugString() << std::endl;

    session->Close();
    delete session;
}
