## TensorFlow Interface for CMSSW

[![build status](https://gitlab.cern.ch/mrieger/CMSSW-DNN/badges/tf_py_cpython/build.svg)](https://gitlab.cern.ch/mrieger/CMSSW-DNN/pipelines)

- Main repository & issues: [gitlab.cern.ch/mrieger/CMSSW-DNN](https://gitlab.cern.ch/mrieger/CMSSW-DNN)
- Code mirror: [github.com/riga/CMSSW-DNN](https://github.com/riga/CMSSW-DNN)

### Note

The interface was merged under [PhysicsTools/TensorFlow](https://github.com/cms-sw/cmssw/tree/master/PhysicsTools/TensorFlow) on Jan 25 2018 into [CMSSW\_10\_1\_X](https://github.com/cms-sw/cmssw/pull/19893) and backported to [CMSSW\_9\_4\_X](https://github.com/cms-sw/cmssw/pull/22042) on Feb 15 2018. For development purposes, the include paths in this repository point to `DNN/TensorFlow`.

---

This interface provides simple and fast access to [TensorFlow](https://www.tensorflow.org) in CMSSW and lets you evaluate trained models right within your C++ modules. It **does not depend** on a converter library or custom NN implementation. In fact, it is a thin layer on top of TensorFlow's C++ API (available via exernals in `/cvmfs`) which handles session / graph loading & cleanup, exceptions, and thread management within CMSSW. As a result, you can load and evaluate every model that was previously trained and saved in Python (or C++).

Due to the development of the CMS software environment since 8\_0\_X, there are multiple versions of this interface. But since the C++ API was added in 9\_4\_X, the interface API is stable and should handle all changes within TensorFlow internally. The following table summarizes all available versions, mapped to CMSSW version and SCRAM\_ARCH:

| CMSSW version |     SCRAM\_ARCH     | TF API & version (externals) |                          Interface branch                         |
| ------------- | ------------------- | ---------------------------- | ----------------------------------------------------------------- |
| t.b.a.        | slc6\_amd64\_gcc630 | C++, 1.6.0                   | [tf\_cc\_1.6](/../tree/tf_cc_1.6)                                 |
| 10\_1\_X      | slc6\_amd64\_gcc630 | C++, 1.5.0                   | [tf\_cc\_1.5](/../tree/tf_cc_1.5)                                 |
| 9\_4\_X       | slc6\_amd64\_gcc630 | C++, 1.3.0                   | [tf\_cc\_1.3](/../tree/tf_cc_1.3) / **[master](/../tree/master)** |
| 9\_3\_X       | slc6\_amd64\_gcc630 | C, 1.1.0                     | [tf\_c](/../tree/tf_c)                                            |
| 8\_0\_X       | slc6\_amd64\_gcc530 | Py + CPython, 1.1.0          | [tf\_py\_cpython](/../tree/tf_py_cpython)                         |


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
dnn::tf::Shape xShape[] = {1, 10}; // 1 = single batch
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

##### 93X

93X is not released yet. However, you can test pre-releases such as `CMSSW_9_3_X_2017-07-20-1100` on lxplus.

```bash
source /cvmfs/cms.cern.ch/cmsset_default.sh

export SCRAM_ARCH="slc6_amd64_gcc630"
export CMSSW_VERSION="CMSSW_9_3_X_2017-07-20-1100" # pre-release

cmsrel $CMSSW_VERSION
cd $CMSSW_VERSION/src
cmsenv

git clone https://gitlab.cern.ch/mrieger/CMSSW-DNN.git DNN

scram b -j
```


##### 80X

```bash
source /cvmfs/cms.cern.ch/cmsset_default.sh

export SCRAM_ARCH="slc6_amd64_gcc530"
export CMSSW_VERSION="CMSSW_8_0_26_patch2"

cmsrel $CMSSW_VERSION
cd $CMSSW_VERSION/src
cmsenv

git clone https://gitlab.cern.ch/mrieger/CMSSW-DNN.git DNN
./DNN/setup_legacy.sh # only difference to 93X setup

scram b -j
```
