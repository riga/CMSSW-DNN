/*
 * Python interface.
 */

#include "DNN/Base/interface/PythonInterface.h"

namespace DNN
{

PythonInterface::PythonInterface()
    : context_(0)
    , logLevel_(INFO)
{
}

PythonInterface::~PythonInterface()
{
}

void PythonInterface::initialize()
{
    PyEval_InitThreads();
    Py_Initialize();
}

void PythonInterface::finalize()
{
    Py_Finalize();
}

void PythonInterface::except(PyObject* obj, const std::string& msg)
{
    // in Python we know that an error occured when an object is NULL
    if (obj == NULL)
    {
        // check if there is a python error on the stack
        if (PyErr_Occurred() != NULL)
        {
            PyErr_PrintEx(0);
        }
        throw std::runtime_error("a python error occured: " + msg);
    }
}

bool PythonInterface::hasContext()
{
    return context_ != 0;
}

void PythonInterface::checkContext()
{
    if (!hasContext())
    {
        throw std::runtime_error("python context not yet started");
    }
}

void PythonInterface::startContext()
{
    // TODO: remember the main object?

    if (hasContext())
    {
        throw std::runtime_error("python context already started");
    }

    // create the main module and globals dict
    PyObject* main = PyImport_AddModule("__main__");
    PyObject* globals = PyModule_GetDict(main);

    // copy the global dict to create a new reference rather than borrowing one
    // this will be our context
    context_ = PyDict_Copy(globals);

    // decrease borrowed references
    Py_DECREF(globals);
}

void PythonInterface::runScript(const std::string& script)
{
    checkContext();

    // run the script in our context
    PyObject* result = PyRun_String(script.c_str(), Py_file_input, context_, context_);
    except(result, "error during execution of script");

    // decrease borrowed references
    Py_DECREF(result);
}

void PythonInterface::runFile(const std::string& filename)
{
    // read the content of the file
    std::ifstream ifs(filename);
    std::string script;
    script.assign(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());

    // run the script
    runScript(script);
}

PyObject* PythonInterface::call(PyObject* callable, PyObject* args)
{
    // simply call the callable with args and check for errors afterwards
    PyObject* result = PyObject_CallObject(callable, args);
    except(result, "error during invocation of callable");

    return result;
}

} // namespace DNN
