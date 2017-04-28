/*
 * Python interface.
 *
 * Author:
 *   Marcel Rieger
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
public:
    PythonInterface(const LogLevel& level = INFO);
    virtual ~PythonInterface();

    void except(PyObject* obj, const std::string& msg) const;
    void release(PyObject*& ptr) const;

    PyObject* get(const std::string& name) const;
    PyObject* call(PyObject* callable, PyObject* args = 0) const;
    PyObject* call(const std::string& name, PyObject* args = 0) const;

    void runScript(const std::string& script);
    void runFile(const std::string& filename);

    PyObject* createTuple(const std::vector<int>& v) const;
    PyObject* createTuple(const std::vector<double>& v) const;

    void setLogLevel(LogLevel& level);
    LogLevel getLogLevel() const;

private:
    static size_t nConsumers;

    PyObject* context;
    LogLevel logLevel;

    void initialize() const;
    void finalize() const;

    bool hasContext() const;
    void checkContext() const;
    void startContext();

    void log(const LogLevel& level, const std::string& msg) const;
};

size_t PythonInterface::nConsumers = 0;

} // namepace DNN

#endif // DNN_BASE_PYTHONINTERFACE_H
