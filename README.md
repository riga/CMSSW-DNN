## TensorFlow Interface for CMSSW

[![build status](https://gitlab.cern.ch/mrieger/CMSSW-DNN/badges/tf_c/build.svg)](https://gitlab.cern.ch/mrieger/CMSSW-DNN/pipelines)

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

There are two ways to evaluate your model: *stateful* and *stateless*.

Stateful means that you define inputs and outputs to the computational graph on the session object before the first `run()` call. Overhead due to repeatedly performed sanity checks is minimized. However, this approach cannot be used if thread-safety is required.

For those cases, a second `run()` method is provided that leaves the session constant. See below for examples.

```cpp
#include "DNN/TensorFlow/interface/TensorFlow.h"

//
// setup (common)
//

// load the graph
tf::Graph graph("/path/to/simplegraph");

// create a session
tf::Session session(&graph);

// prepare input tensors
tf::Shape xShape[] = { 1, 10 }; // 1 = single batch
tf::Tensor* x = new tf::Tensor(2, xShape);

// example: fill a single batch of the input tensor with consecutive numbers
// -> [[0, 1, 2, ...]]
std::vector<float> values = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
x->setVector<float>(1, 0, values); // axis 1, batch 0, values

// prepare output tensors, no shape required
tf::Tensor* y = new tf::Tensor();


//
// stateful evaluation
//

// define input and outputs on the session
session.defineInput(x, "input");
session.defineOutput(y, "output");

// run it
session.run();


//
// stateless evaluation
//

// define input and outputs
tf::IOs inputs = { session.createIO(x, "input") };
tf::IOs outputs = { session.createIO(y, "output") };

// run it
session.run(inputs, outputs);


//
// process outputs (common)
//

// print the output
// -> [[float]]
std::cout << *y->getPtr<float>(0, 0) << std::endl;

// cleanup
delete x;
delete y;
```

For more examples, see [`test/testSession.cc`](./TensorFlow/test/testSession.cc) and [`test/testTensor.cc`](./TensorFlow/test/testTensor.cc).


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

Any CMSSW 93X version will work:

```bash
source /cvmfs/cms.cern.ch/cmsset_default.sh

export SCRAM_ARCH="slc6_amd64_gcc630"
export CMSSW_VERSION="CMSSW_9_3_0"

cmsrel $CMSSW_VERSION
cd $CMSSW_VERSION/src
cmsenv

git clone https://gitlab.cern.ch/mrieger/CMSSW-DNN.git DNN

scram b
```


### Important Notes

##### Multi-threading

If you want TensorFlow to use multiple threads, you can pass `true` as the second argument to the Session constructor. By default, only one thread is used.

```cpp
// load the graph
tf::Graph graph("/path/to/graph");

// create a session and use multi-threading
tf::Session session(&graph, true);

// proceed as usual
...
```


##### Logging

By default, only error logs from the TensorFlow C API are shown. This can be changed via setting the `TF_CPP_MIN_LOG_LEVEL` environment varibale before calling (e.g.) `cmsRun`:

| `TF_CPP_MIN_LOG_LEVEL` value | Verbosity level |
| ---------------------------- | --------------- |
| 0                            | debug           |
| 1                            | info            |
| 2                            | warning         |
| 3 (default)                  | error           |
| 4                            | none            |

Forwarding to the `MessageLogger` service is not yet possible.
