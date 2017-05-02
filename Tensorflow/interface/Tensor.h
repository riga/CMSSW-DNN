/*
 * Generic Tensorflow tensor representation.
 *
 * Author:
 *   Marcel Rieger
 */

#ifndef DNN_TENSORFLOW_TENSOR_H_FOO
#define DNN_TENSORFLOW_TENSOR_H_FOO

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Python.h"
#include "numpy/arrayobject.h"

namespace dnn
{

namespace tf
{

class Tensor
{
public:
    Tensor(const std::string& name);
    Tensor(int rank, npy_intp* shape, int typenum = NPY_FLOAT);
    Tensor(const std::string& name, int rank, npy_intp* shape, int typenum = NPY_FLOAT);
    virtual ~Tensor();

    std::string getName() const;
    void setName(const std::string& name);

    bool isEmpty() const;

    inline PyArrayObject* getArray()
    {
        return array;
    }
    void setArray(PyArrayObject* array); // steals reference
    void setArray(int rank, npy_intp* shape, int typenum = NPY_FLOAT);

    int getRank() const;
    const npy_intp* getShape() const;
    npy_intp getShape(int axis) const;

    void* getPtrAtPos(npy_intp* pos);
    void* getPtr();
    void* getPtr(npy_intp i);
    void* getPtr(npy_intp i, npy_intp j);
    void* getPtr(npy_intp i, npy_intp j, npy_intp k);
    void* getPtr(npy_intp i, npy_intp j, npy_intp k, npy_intp l);
    void* getPtr(npy_intp i, npy_intp j, npy_intp k, npy_intp l, npy_intp m);

    template <typename T>
    T getValueAtPos(npy_intp* pos);
    template <typename T>
    T getValue();
    template <typename T>
    T getValue(npy_intp i);
    template <typename T>
    T getValue(npy_intp i, npy_intp j);
    template <typename T>
    T getValue(npy_intp i, npy_intp j, npy_intp k);
    template <typename T>
    T getValue(npy_intp i, npy_intp j, npy_intp k, npy_intp l);
    template <typename T>
    T getValue(npy_intp i, npy_intp j, npy_intp k, npy_intp l, npy_intp m);

    template <typename T>
    void setValueAtPos(npy_intp* pos, T value);
    template <typename T>
    void setValue(T value);
    template <typename T>
    void setValue(npy_intp i, T value);
    template <typename T>
    void setValue(npy_intp i, npy_intp j, T value);
    template <typename T>
    void setValue(npy_intp i, npy_intp j, npy_intp k, T value);
    template <typename T>
    void setValue(npy_intp i, npy_intp j, npy_intp k, npy_intp l, T value);
    template <typename T>
    void setValue(npy_intp i, npy_intp j, npy_intp k, npy_intp l, npy_intp m, T value);

    template <typename T>
    std::vector<T> getVectorAtPos(int axis, npy_intp* pos);
    template <typename T>
    std::vector<T> getVector(); // rank 1
    template <typename T>
    std::vector<T> getVector(int axis, npy_intp a); // rank 2
    template <typename T>
    std::vector<T> getVector(int axis, npy_intp a, npy_intp b); // rank 3
    template <typename T>
    std::vector<T> getVector(int axis, npy_intp a, npy_intp b, npy_intp c); // rank 4
    template <typename T>
    std::vector<T> getVector(int axis, npy_intp a, npy_intp b, npy_intp c, npy_intp d); // rank 5

private:
    void init(int rank, npy_intp* shape, int typenum = NPY_FLOAT);

    std::string name;
    PyArrayObject* array;
};

template <typename T>
T Tensor::getValueAtPos(npy_intp* pos)
{
    return *((T*)(getPtrAtPos(pos)));
}

template <typename T>
T Tensor::getValue()
{
    return *((T*)(getPtrAtPos(0)));
}

template <typename T>
T Tensor::getValue(npy_intp i)
{
    return *((T*)(getPtr(i)));
}

template <typename T>
T Tensor::getValue(npy_intp i, npy_intp j)
{
    return *((T*)(getPtr(i, j)));
}

template <typename T>
T Tensor::getValue(npy_intp i, npy_intp j, npy_intp k)
{
    return *((T*)(getPtr(i, j, k)));
}

template <typename T>
T Tensor::getValue(npy_intp i, npy_intp j, npy_intp k, npy_intp l)
{
    return *((T*)(getPtr(i, j, k, l)));
}

template <typename T>
T Tensor::getValue(npy_intp i, npy_intp j, npy_intp k, npy_intp l, npy_intp m)
{
    return *((T*)(getPtr(i, j, k, l, m)));
}

template <typename T>
void Tensor::setValueAtPos(npy_intp* pos, T value)
{
    getPtrAtPos(pos) = (void*)(&value);
}

template <typename T>
void Tensor::setValue(T value)
{
    getPtrAtPos(0) = (void*)(&value);
}

template <typename T>
void Tensor::setValue(npy_intp i, T value)
{
    getPtr(i) = (void*)(&value);
}

template <typename T>
void Tensor::setValue(npy_intp i, npy_intp j, T value)
{
    *((T*)(getPtr(i, j))) = value;
}

template <typename T>
void Tensor::setValue(npy_intp i, npy_intp j, npy_intp k, T value)
{
    *((T*)(getPtr(i, j, k))) = value;
}

template <typename T>
void Tensor::setValue(npy_intp i, npy_intp j, npy_intp k, npy_intp l, T value)
{
    *((T*)(getPtr(i, j, k, l))) = value;
}

template <typename T>
void Tensor::setValue(npy_intp i, npy_intp j, npy_intp k, npy_intp l, npy_intp m, T value)
{
    *((T*)(getPtr(i, j, k, l, m))) = value;
}

template <typename T>
std::vector<T> Tensor::getVectorAtPos(int axis, npy_intp* pos)
{
    const int rank = getRank();
    if (axis < 0)
    {
        axis = rank + axis;
    }
    if (axis >= rank)
    {
        throw std::runtime_error("axis " + std::to_string(axis) + " invalid for rank " 
                                 + std::to_string(rank));
    }

    npy_intp pos2[rank];
    for (int i = 0; i < rank; i++)
    {
        if (i < axis)
        {
            pos2[i] = pos[i];
        }
        else if (i == axis)
        {
            pos2[i] = axis;
        }
        else
        {
            pos2[i] = pos[i-1];
        }
    }

    std::vector<T> v;
    for (npy_intp i = 0; i < getShape(axis); i++)
    {
        pos2[axis] = i;
        v.push_back(*((T*)(getPtrAtPos(pos2))));
    }
    return v;
}

template <typename T>
std::vector<T> Tensor::getVector()
{
    return getVectorAtPos<T>(0, 0);
}

template <typename T>
std::vector<T> Tensor::getVector(int axis, npy_intp a)
{
    npy_intp pos[1] = {a};
    return getVectorAtPos<T>(axis, pos);
}

template <typename T>
std::vector<T> Tensor::getVector(int axis, npy_intp a, npy_intp b)
{
    npy_intp pos[2] = {a, b};
    return getVectorAtPos<T>(axis, pos);
}

template <typename T>
std::vector<T> Tensor::getVector(int axis, npy_intp a, npy_intp b, npy_intp c)
{
    npy_intp pos[3] = {a, b, c};
    return getVectorAtPos<T>(axis, pos);
}

template <typename T>
std::vector<T> Tensor::getVector(int axis, npy_intp a, npy_intp b, npy_intp c, npy_intp d)
{
    npy_intp pos[4] = {a, b, c, d};
    return getVectorAtPos<T>(axis, pos);
}

} // namepace tf

} // namepace dnn

#endif // DNN_TENSORFLOW_TENSOR_H
