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

void PythonInterface::initialize() const
{
    log_(INFO, "initialize python");
    PyEval_InitThreads();
    Py_Initialize();
}

void PythonInterface::finalize() const
{
    log_(INFO, "finalize python");
    Py_Finalize();
}

void PythonInterface::except(PyObject* obj, const std::string& msg) const
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

void PythonInterface::releaseObject(PyObject*& ptr) const
{
    Py_XDECREF(ptr);
    ptr = 0;
}

PyObject* PythonInterface::call(PyObject* callable, PyObject* args) const
{
    // check if args is a tuple
    size_t nArgs = 0;
    if (args)
    {
        if (!PyTuple_Check(args))
        {
            throw std::runtime_error("args for function call is not a tuple");
        }
        nArgs = PyTuple_Size(args);
    }

    log_(DEBUG, "invoke callable with " + std::to_string(nArgs) + " argument(s)");

    // simply call the callable with args and check for errors afterwards
    PyObject* result = PyObject_CallObject(callable, args);
    except(result, "error during invocation of callable");

    return result;
}

PyObject* PythonInterface::createTuple(const std::vector<int>& v) const
{
    PyObject* tpl = PyTuple_New(v.size());
    for (size_t i = 0; i < v.size(); i++)
    {
        PyTuple_SetItem(tpl, i, PyInt_FromLong(v[i]));
    }
    return tpl;
}

PyObject* PythonInterface::createTuple(const std::vector<double>& v) const
{
    PyObject* tpl = PyTuple_New(v.size());
    for (size_t i = 0; i < v.size(); i++)
    {
        PyTuple_SetItem(tpl, i, PyFloat_FromDouble(v[i]));
    }
    return tpl;
}

bool PythonInterface::hasContext() const
{
    return context_ != 0;
}

void PythonInterface::checkContext() const
{
    if (!hasContext())
    {
        throw std::runtime_error("python context not yet started");
    }
}

void PythonInterface::startContext()
{
    log_(INFO, "start context");

    if (hasContext())
    {
        throw std::runtime_error("python context already started");
    }

    // create the main module and globals dict
    // TODO: remember the main object?
    PyObject* main = PyImport_AddModule("__main__");
    PyObject* globals = PyModule_GetDict(main);

    // copy the global dict to create a new reference rather than borrowing one
    // this will be our context
    context_ = PyDict_Copy(globals);

    // decrease borrowed references
    releaseObject(globals);
}

void PythonInterface::runScript(const std::string& script)
{
    log_(INFO, "run script");

    checkContext();

    // run the script in our context
    PyObject* result = PyRun_String(script.c_str(), Py_file_input, context_, context_);
    except(result, "error during execution of script");

    // decrease borrowed references
    releaseObject(result);
}

void PythonInterface::runFile(const std::string& filename)
{
    log_(INFO, "run file from " + filename);

    // read the content of the file
    std::ifstream ifs(filename);
    std::string script;
    script.assign(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());

    // run the script
    runScript(script);
}

void PythonInterface::log_(const LogLevel& level, const std::string& msg) const
{
    if (level >= logLevel_)
    {
        if (level <= INFO)
        {
            std::cout << "PythonInterface: " << msg << std::endl;
        }
        else
        {
            std::cerr << "PythonInterface: " << msg << std::endl;
        }
    }
}

} // namespace DNN
