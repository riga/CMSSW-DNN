/*
 * Python interface.
 */

#ifndef DNN_BASE_PYTHONINTERFACE_H
#define DNN_BASE_PYTHONINTERFACE_H

#include <fstream>
#include <stdexcept>
#include <string>

#include "Python.h"

namespace DNN
{

enum LogLevel
{
    ALL = 0,
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

class PythonInterface
{
private:
    PyObject* context_;
    LogLevel logLevel_;

public:
    PythonInterface();
    virtual ~PythonInterface();

    static void initialize();
    static void finalize();

    static void except(PyObject* obj, const std::string& msg);

    bool hasContext();
    void checkContext();
    void startContext();

    void runScript(const std::string& script);
    void runFile(const std::string& filename);

    PyObject* call(PyObject* callable, PyObject* args = 0);

    inline void setLogLevel(LogLevel level)
    {
        logLevel_ = level;
    }

    inline LogLevel getLogLevel()
    {
        return logLevel_;
    }
};

} // namepace DNN

#endif // DNN_BASE_PYTHONINTERFACE_H
