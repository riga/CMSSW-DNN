/*
 * Generic Tensorflow tensor representation.
 *
 * Author:
 *   Marcel Rieger
 */

#include "DNN/Tensorflow/interface/Tensor.h"

namespace dnn
{

namespace tf
{

Tensor::Tensor(const std::string& name)
    : name(name)
    , data(0)
    , arrData(0)
{
    init(-1, 0);
}

Tensor::Tensor(int rank, npy_intp* shape, int typenum)
    : name("")
    , data(0)
    , arrData(0)
{
    init(rank, shape, typenum);
}

Tensor::Tensor(const std::string& name, int rank, npy_intp* shape, int typenum)
    : name(name)
    , data(0)
    , arrData(0)
{
    init(rank, shape, typenum);
}

Tensor::~Tensor()
{
    if (!isEmpty())
    {
        Py_DECREF(data);
    }
}

void Tensor::init(int rank, npy_intp* shape, int typenum)
{
    if (PyArray_API == NULL)
    {
        import_array();
    }

    if (rank >= 0)
    {
        data = PyArray_ZEROS(rank, shape, typenum, 0);
        arrData = (PyArrayObject*)data;
    }
}

bool Tensor::isEmpty() const
{
    return !data || !arrData;
}

std::string Tensor::getName() const
{
    return name;
}

void Tensor::setName(const std::string& name)
{
    this->name = name;
}

int Tensor::getRank() const
{
    if (isEmpty())
    {
        return -1;
    }
    return arrData->nd;
}

const npy_intp* Tensor::getShape() const
{
    if (isEmpty())
    {
        return 0;
    }
    return arrData->dimensions;
}

npy_intp Tensor::getShape(int axis) const
{
    if (isEmpty())
    {
        return -1;
    }
    return (arrData->dimensions)[axis];
}

void* Tensor::getPtrAtPos(npy_intp* pos)
{
    if (isEmpty())
    {
        return 0;
    }

    return PyArray_GetPtr(arrData, pos);
}

void* Tensor::getPtr()
{
    return getPtrAtPos(0);
}

void* Tensor::getPtr(npy_intp i)
{
    npy_intp pos[1] = {i};
    return getPtrAtPos(pos);
}

void* Tensor::getPtr(npy_intp i, npy_intp j)
{
    npy_intp pos[2] = {i, j};
    return getPtrAtPos(pos);
}

void* Tensor::getPtr(npy_intp i, npy_intp j, npy_intp k)
{
    npy_intp pos[3] = {i, j, k};
    return getPtrAtPos(pos);
}

void* Tensor::getPtr(npy_intp i, npy_intp j, npy_intp k, npy_intp l)
{
    npy_intp pos[4] = {i, j, k, l};
    return getPtrAtPos(pos);
}

void* Tensor::getPtr(npy_intp i, npy_intp j, npy_intp k, npy_intp l, npy_intp m)
{
    npy_intp pos[5] = {i, j, k, l, m};
    return getPtrAtPos(pos);
}

PyArrayObject* Tensor::getArray()
{
    return arrData;
}

void Tensor::setArray(PyObject* data)
{
    if (!isEmpty())
    {
        Py_DECREF(data);
    }

    this->data = data;
    arrData = (PyArrayObject*)data;
}

} // namespace tf

} // namespace dnn
