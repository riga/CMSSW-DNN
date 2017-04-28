/*
 * Python interface.
 */

#ifndef DNN_BASE_PYTHONINTERFACE_H
#define DNN_BASE_PYTHONINTERFACE_H

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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

    void log_(const LogLevel& level, const std::string& msg) const;

public:
    PythonInterface();
    virtual ~PythonInterface();

    void initialize() const;
    void finalize() const;

    void except(PyObject* obj, const std::string& msg) const;
    void releaseObject(PyObject*& ptr) const;

    PyObject* call(PyObject* callable, PyObject* args = 0) const;

    PyObject* createTuple(const std::vector<int>& v) const;
    PyObject* createTuple(const std::vector<double>& v) const;

    bool hasContext() const;
    void checkContext() const;
    void startContext();

    void runScript(const std::string& script);
    void runFile(const std::string& filename);

    inline void setLogLevel(LogLevel& level)
    {
        log_(DEBUG, "set log level to " + std::to_string(level));
        logLevel_ = level;
    }

    inline LogLevel getLogLevel() const
    {
        return logLevel_;
    }
};

} // namepace DNN

#endif // DNN_BASE_PYTHONINTERFACE_H
