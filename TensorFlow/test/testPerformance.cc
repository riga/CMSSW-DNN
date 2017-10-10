/*
 * Test of the TensorFlow interface performance.
 * Based on TensorFlow C++ API 1.3.
 * For more info, see https://gitlab.cern.ch/mrieger/CMSSW-DNN.
 *
 * Author: Marcel Rieger
 */

#include <boost/filesystem.hpp>
#include <cppunit/extensions/HelperMacros.h>
#include <stdexcept>
#include <sys/time.h>

#include "DNN/TensorFlow/interface/TensorFlow.h"

class testPerformance : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(testPerformance);
    CPPUNIT_TEST(checkAll);
    CPPUNIT_TEST_SUITE_END();

public:
    std::string dataPath;

    void setUp();
    void tearDown();
    void checkAll();

    static long int getTime(timeval* t)
    {
        gettimeofday(t, NULL);
        return t->tv_sec * 1000 + t->tv_usec / 1000;
    }

    void runBatches(bool singleThreaded);
};

CPPUNIT_TEST_SUITE_REGISTRATION(testPerformance);

void testPerformance::setUp()
{
    dataPath = std::string(getenv("CMSSW_BASE")) + "/test/" + std::string(getenv("SCRAM_ARCH"))
        + "/" + boost::filesystem::unique_path().string();

    // create the graph
    std::string testPath = std::string(getenv("CMSSW_BASE")) + "/src/DNN/TensorFlow/test";
    std::string cmd = "python " + testPath + "/createlargegraph.py " + dataPath;
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

void testPerformance::tearDown()
{
    if (boost::filesystem::exists(dataPath))
    {
        boost::filesystem::remove_all(dataPath);
    }
}

void testPerformance::runBatches(bool multiThreaded)
{
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << (multiThreaded ? "multi" : "single") << "-threaded performance:" << std::endl
              << std::endl;

    struct timeval t;

    // do the testing for various different batch sizes
    int n = 1000;
    int batchSizes[] = { 1, 10, 100, 1000 };

    tf::setLogging();
    std::string exportDir = dataPath + "/largegraph";
    tf::MetaGraphDef* metaGraph = tf::loadMetaGraph(exportDir, multiThreaded);
    tf::Session* session = tf::createSession(metaGraph, exportDir, multiThreaded);

    for (size_t i = 0; i < 4; i++)
    {
        std::cout << "run " << n << " evaluations for batch size " << batchSizes[i] << std::endl;

        // create the input tensor and add it to the graph
        tf::Tensor x(tf::DT_FLOAT, { batchSizes[i], 100 });
        float* d = x.flat<float>().data();
        for (size_t i = 0; i < 10; i++, d++)
        {
            *d = float(i);
        }

        // set values
        for (int b = 0; b < batchSizes[i]; b++)
        {
            for (int j = 0; j < 100; j++)
            {
                x.matrix<float>()(b, j) = (float)(b + j);
            }
        }

        // measure the run time with n repititions
        long int t0 = getTime(&t);
        for (int j = 0; j < n; j++)
        {
            tf::run(session, { { "input", x } }, { "output:0" }, nullptr);
        }
        long int t1 = getTime(&t);
        std::cout << "-> " << (t1 - t0) / (float)n << " ms per batch" << std::endl
                  << std::endl;
    }

    // cleanup
    session->Close();
    delete session;
    delete metaGraph;
}

void testPerformance::checkAll()
{
    // single-threaded
    runBatches(false);

    // multi-threaded
    runBatches(true);
}
