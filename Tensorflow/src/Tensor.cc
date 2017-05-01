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
    , array(0)
{
    init(-1, 0);
}

Tensor::Tensor(int rank, npy_intp* shape, int typenum)
    : name("")
    , array(0)
{
    init(rank, shape, typenum);
}

Tensor::Tensor(const std::string& name, int rank, npy_intp* shape, int typenum)
    : name(name)
    , array(0)
{
    init(rank, shape, typenum);
}

Tensor::~Tensor()
{
    if (!isEmpty())
    {
        Py_DECREF(array);
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
        array = (PyArrayObject*)PyArray_ZEROS(rank, shape, typenum, 0);;
    }
}

bool Tensor::isEmpty() const
{
    return !array;
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
    return array->nd;
}

const npy_intp* Tensor::getShape() const
{
    if (isEmpty())
    {
        return 0;
    }
    return array->dimensions;
}

npy_intp Tensor::getShape(int axis) const
{
    if (isEmpty())
    {
        return -1;
    }
    return (array->dimensions)[axis];
}

void* Tensor::getPtrAtPos(npy_intp* pos)
{
    if (isEmpty())
    {
        return 0;
    }

    return PyArray_GetPtr(array, pos);
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
    return array;
}

void Tensor::setArray(PyArrayObject* array)
{
    if (!isEmpty())
    {
        Py_DECREF(this->array);
    }

    this->array = array;
}

void Tensor::setArray(PyObject* array)
{
    setArray((PyArrayObject*)array);
}

} // namespace tf

} // namespace dnn
