## DNN / Tensorflow Interface for CMSSW&nbsp;&nbsp;&nbsp;&nbsp;[![build status](https://gitlab.cern.ch/mrieger/CMSSW-DNN/badges/master/build.svg)](https://gitlab.cern.ch/mrieger/CMSSW-DNN/pipelines)

- Main repository & issues: [gitlab.cern.ch/mrieger/CMSSW-DNN](https://gitlab.cern.ch/mrieger/CMSSW-DNN)
- Code mirror: [github.com/riga/CMSSW-DNN](https://github.com/riga/CMSSW-DNN)

This project provides a simple yet fast interface to [Tensorflow](https://www.tensorflow.org) graphs and tensors which lets you evaluate trained models right within CMSSW. It **does not depend** on a converter library or custom NN implementation. By using the C-API's of both Python and NumPy (available via `/cvmfs`), you can essentially load and evaluate every model that was previously saved via [`tf.train.Saver.save()`](https://www.tensorflow.org/api_docs/python/tf/train/Saver#save).

Tensorflow is available starting from CMSSW 9.0.X ([PR](https://github.com/cms-sw/cmsdist/pull/2824)). The software bundle for CMSSW 8.0.X is provided by [M. Harrendorf](https://github.com/mharrend).


##### Features in a nutshell

- Direct interface to Tensorflow / NumPy objects, no intermediate converter library required.
- Fast data access via Numpy arrays and in-place operations.
- Evaluation with multiple input and output tensors.
- Batching.
- **GPU support**.


### Usage

##### Save your Model (in Python)

```python
import tensorflow as tf

# define your model here
# example:
x_ = tf.placeholder(tf.float32, [None, 10], name="input")
W = tf.Variable(tf.ones([10, 1]))
b = tf.Variable(tf.ones([1]))
y = tf.add(tf.matmul(x_, W), b, name="output")

# create a session and initialize everything
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train your model here
...

# save your model
saver = tf.train.Saver()
saver.save(sess, "/path/to/simplegraph")
```


##### Evaluate your Model (in CMSSW)

(from [`test_tfgraph.cc`](./Tensorflow/bin/test_tfgraph.cc))

```cpp
//
// setup
//

// load and initialize the graph
dnn::tf::Graph graph("/path/to/simplegraph");

// prepare input and output tensors
// no shape info required for output
npy_intp xShape[] = {1, 10}; // 1 = single batch
dnn::tf::Tensor* x = graph.defineInput(new dnn::tf::Tensor("input:0", 2, xShape));
dnn::tf::Tensor* y = graph.defineOutput(new dnn::tf::Tensor("output:0"));

//
// evaluation
//

// example: fill a single batch of the input tensor with consecutive numbers
// -> [[0, 1, 2, ...]]
for (int i = 0; i < x->getShape(1); i++)
    x->setValue<float>(0, i, (float)i);

// evaluation call
// this does not return anything but changes the output tensor(s) in-place
graph.eval();

// print the output
// -> [[float]]
std::cout << y->getValue<float>(0, 0) << std::endl;

// cleanup
delete x;
delete y;
```


##### Note on Keras

As Keras can be backed by Tensorflow, the model saving process is identical:

```python
import tensorflow as tf
sess = tf.Session()

from keras import backend as K
K.set_session(sess)

# define and train your keras model here
...

# save your model
saver = tf.train.Saver()
saver.save(sess, "/path/to/simplegraph")
```


### Performance

A performance test (CPU only for now) is located at [`test_tfperf.cc`](./Tensorflow/bin/test_tfperf.cc) and runs 10k evaluations of a feed-forward network with 100 input features, 5 hidden elu layers with 200 nodes each, a softmax output with 10 nodes, and multiple batch sizes.

```shell
> test_tfperf
...
run 10000 evaluations for batch size 1
-> 1.4844 ms per batch

run 10000 evaluations for batch size 10
-> 3.4295 ms per batch

run 10000 evaluations for batch size 100
-> 7.563 ms per batch

run 10000 evaluations for batch size 1000
-> 15.7977 ms per batch
```


### Installation

```bash
source /cvmfs/cms.cern.ch/cmsset_default.sh

export SCRAM_ARCH="slc6_amd64_gcc530"
export CMSSW_VERSION="CMSSW_8_0_26_patch2"

cmsrel $CMSSW_VERSION
cd $CMSSW_VERSION/src
cmsenv

git clone https://gitlab.cern.ch/mrieger/CMSSW-DNN.git DNN
./DNN/setup.sh

scram b -j
```
