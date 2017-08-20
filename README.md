## DNN / TensorFlow Interface for CMSSW&nbsp;&nbsp;&nbsp;&nbsp;[![build status](https://gitlab.cern.ch/mrieger/CMSSW-DNN/badges/master/build.svg)](https://gitlab.cern.ch/mrieger/CMSSW-DNN/pipelines)

- Main repository & issues: [gitlab.cern.ch/mrieger/CMSSW-DNN](https://gitlab.cern.ch/mrieger/CMSSW-DNN)
- Code mirror: [github.com/riga/CMSSW-DNN](https://github.com/riga/CMSSW-DNN)

This project provides a simple yet fast interface to [TensorFlow](https://www.tensorflow.org) graphs and tensors which lets you evaluate trained models right within CMSSW. It **does not depend** on a converter library or custom NN implementation. By using TensorFlow's C API (available via `/cvmfs`), you can essentially load and evaluate every model that was previously saved in both C **or** Python.

This interface requires CMSSW 9.3.X or greater. For lower versions see the [80X branch](/../tree/80X).


##### Features in a nutshell

- Native supports for arbitrary network architectures.
- Direct interface to TensorFlow, no intermediate converter library required.
- Fast data access and in-place operations.
- Evaluation with multiple input and output tensors (and tensors defined as inputs multiple times).
- Batching.
- **GPU support**.


### Usage

Model saving and loading makes use of TensorFlow's [``SavedModel`` serialization format](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md).


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
builder = tf.saved_model.builder.SavedModelBuilder("/path/to/simplegraph")
builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
builder.save()
```

The tag passed to `add_meta_graph_and_variables` serves as an identifier for your graph in the saved file which potentially can contain multiple graphs. `tf.saved_model.tag_constants.SERVING` ("serve") is a commonly used tag, but you are free to use any value here.


##### Evaluate your Model (in CMSSW)

```cpp
//
// setup
//

// load and initialize the graph
tf::Graph graph("/path/to/simplegraph");

// prepare input and output tensors
tf::Shape xShape[] = {1, 10}; // 1 = single batch
tf::Tensor* x = new tf::Tensor(2, xShape);
graph.defineInput(x, "input");

// no shape info required for output
tf::Tensor* y = new tf::Tensor();
graph.defineOutput(y, "output");


//
// evaluation
//

// example: fill a single batch of the input tensor with consecutive numbers
// -> [[0, 1, 2, ...]]
std::vector<float> values = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
x->setVector<float>(1, 0, values); // axis 1, batch 0, values

// evaluation call
// this does not return anything but changes the output tensor(s) in-place
graph.eval();

// print the output
// -> [[float]]
std::cout << *y->getPtr<float>(0, 0) << std::endl;

// cleanup
delete x;
delete y;
```

For more examples, see [`test/testTensor.cc`](./TensorFlow/test/testTensor.cc) and [`test/testGraph.cc`](./TensorFlow/test/testGraph.cc).


##### Note on Keras

As Keras can be backed by TensorFlow, the model saving process is identical:

```python
import tensorflow as tf
sess = tf.Session()

from keras import backend as K
K.set_session(sess)

# define and train your keras model here
...

# save your model
builder = tf.saved_model.builder.SavedModelBuilder("/path/to/simplegraph")
builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
builder.save()
```


### Performance

A performance test (CPU only for now) is located at [`test/testPerformance.cc`](./TensorFlow/test/testPerformance.cc) and runs 1k evaluations of a feed-forward network with 100 input features, 5 hidden elu layers with 200 nodes each, a softmax output with 10 nodes, and multiple batch sizes. Of course, the actual performance is hardware dependent.

```
run 1000 evaluations for batch size 1
-> 0.134 ms per batch

run 1000 evaluations for batch size 10
-> 0.440 ms per batch

run 1000 evaluations for batch size 100
-> 1.860 ms per batch

run 1000 evaluations for batch size 1000
-> 8.065 ms per batch
```


### Installation

93X is not released yet. However, you can test pre-releases such as `CMSSW_9_3_X_2017-08-19-1100` on lxplus.

```bash
source /cvmfs/cms.cern.ch/cmsset_default.sh

export SCRAM_ARCH="slc6_amd64_gcc630"
export CMSSW_VERSION="CMSSW_9_3_X_2017-08-19-1100" # pre-release

cmsrel $CMSSW_VERSION
cd $CMSSW_VERSION/src
cmsenv

git clone https://gitlab.cern.ch/mrieger/CMSSW-DNN.git DNN

scram b
```
