## DNN / TensorFlow Interface for CMSSW&nbsp;&nbsp;&nbsp;&nbsp;[![build status](https://gitlab.cern.ch/mrieger/CMSSW-DNN/badges/master/build.svg)](https://gitlab.cern.ch/mrieger/CMSSW-DNN/pipelines)

- Main repository & issues: [gitlab.cern.ch/mrieger/CMSSW-DNN](https://gitlab.cern.ch/mrieger/CMSSW-DNN)
- Code mirror: [github.com/riga/CMSSW-DNN](https://github.com/riga/CMSSW-DNN)

This project provides a simple yet fast interface to [TensorFlow](https://www.tensorflow.org) and lets you evaluate trained models right within CMSSW. It **does not depend** on a converter library or custom NN implementation. By using TensorFlow's C++ API (available via `/cvmfs`), you can essentially load and evaluate every model that was previously saved in both C++ **or** Python.

This interface requires CMSSW 9.4.X or greater. For the C API based version see the [c_api branch](/../tree/c_api). For lower versions see the [80X branch](/../tree/80X).


### Usage

TensorFlow provides multiple ways to save a computational graph. Depending on which method is used, the API calls to load a graph in CMSSW vary.

After defining and training a neural network in Python, you typically don't want to continue training within CMSSW. If this is the case, you want to save a [constant graph](#constant-graph). Otherwise, jump to the [`SavedModel` format](#savedmodel-format).


#### Constant Graphs

A constant graph is saved in a single protobuf file. During the saving process, variables are converted to constant tensors, and ops and tensors that are no longer required (cost function, optimizer, etc.) are removed. The memory consumption during evaluation in CMSSW - especially in multi-threaded mode - can greatly benefit from this conversion.


###### Saving

```python
import tensorflow as tf

# define your model here
x_ = tf.placeholder(tf.float32, [None, 10], name="input")
W = tf.Variable(tf.ones([10, 1]))
b = tf.Variable(tf.ones([1]))
y = tf.add(tf.matmul(x_, W), b, name="output")

# create a session and initialize everything
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# training
...

# convert and save it
outputs = ["output"] # names of output operations you want to use later
constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), outputs)
tf.train.write_graph(constant_graph, "/path/to", "constantgraph.pb", as_text=False)
```

###### Loading and Evaluation

```cpp
#include "DNN/TensorFlow/interface/TensorFlow.h"

//
// setup
//

// load the graph definition, i.e. an object that contains the computational graph
tensorflow::GraphDef* graphDef = tensorflow::loadGraphDef("/path/to/constantgraph.pb");

// create a session
tensorflow::Session* session = tensorflow::createSession(graphDef);

// create an input tensor
tensorflow::Tensor input(tensorflow::DT_FLOAT, { 1, 10 }); // single batch of dimension 10

// example: fill a single batch of the input tensor with consecutive numbers
// -> [[0, 1, 2, ...]]
for (size_t i = 0; i < 10; i++) input.matrix<float>()(0, i) = float(i);


//
// evaluation
//

std::vector<tensorflow::Tensor> outputs;
tensorflow::run(session, { { "input", input } }, { "output" }, &outputs);


//
// process outputs
//

// print the output
// -> [[float]]
std::cout << outputs[0].matrix<float>()(0, 0) << std::endl;
// -> 46.

// cleanup
tensorflow::closeSession(session);
delete graphDef;
```

For more examples, see [`test/testGraphLoading.cc`](./TensorFlow/test/testGraphLoading.cc).


#### `SavedModel` Format


TensorFlow's [``SavedModel`` serialization format](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md) is a more generic


###### Saving

```python
import tensorflow as tf

# define your model here
x_ = tf.placeholder(tf.float32, [None, 10], name="input")
W = tf.Variable(tf.ones([10, 1]))
b = tf.Variable(tf.ones([1]))
y = tf.add(tf.matmul(x_, W), b, name="output")

# create a session and initialize everything
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# training
...

# save it
builder = tf.saved_model.builder.SavedModelBuilder("/path/to/simplegraph")
builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
builder.save()
```

The tag passed to `add_meta_graph_and_variables` serves as an identifier for your graph in the saved file which potentially can contain multiple graphs. `tf.saved_model.tag_constants.SERVING` ("serve") is a commonly used tag, but you are free to use any value here.


###### Loading and Evaluation

```cpp
#include "DNN/TensorFlow/interface/TensorFlow.h"

//
// setup
//

// load the meta graph, i.e. an object that contains the computational graph
// as well as some meta information
tensorflow::MetaGraphDef* metaGraph = tensorflow::loadMetaGraph("/path/to/simplegraph");

// create a session
// we need to pass the export directory again so that the session can
// properly initialize all variables
tensorflow::Session* session = tensorflow::createSession(metaGraph, "/path/to/simplegraph");

// create an input tensor
tensorflow::Tensor input(tensorflow::DT_FLOAT, { 1, 10 }); // single batch of dimension 10

// example: fill a single batch of the input tensor with consecutive numbers
// -> [[0, 1, 2, ...]]
for (size_t i = 0; i < 10; i++) input.matrix<float>()(0, i) = float(i);


//
// evaluation
//

std::vector<tensorflow::Tensor> outputs;
tensorflow::run(session, { { "input", input } }, { "output" }, &outputs);


//
// process outputs
//

// print the output
// -> [[float]]
std::cout << outputs[0].matrix<float>()(0, 0) << std::endl;
// -> 46.

// cleanup
tensorflow::closeSession(session);
delete graphDef;
```

For more examples, see [`test/testMetaGraphLoading.cc`](./TensorFlow/test/testMetaGraphLoading.cc).


### Note on Keras

As Keras can be backed by TensorFlow, the model saving process is identical:

```python
import tensorflow as tf
sess = tf.Session()

from keras import backend as K
K.set_session(sess)

# define and train your keras model here
...

# save at as a constant graph
outputs = [...]
constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), outputs)
tf.train.write_graph(constant_graph, "/path/to", "constantgraph.pb", as_text=False)

# save it as a SavedModel
builder = tf.saved_model.builder.SavedModelBuilder("/path/to/simplegraph")
builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
builder.save()
```


### Performance

A performance test (CPU only for now) is located at [`test/testPerformance.cc`](./TensorFlow/test/testPerformance.cc) and runs 1k evaluations of a feed-forward network with 100 input features, 5 hidden elu layers with 200 nodes each, a softmax output with 10 nodes, and multiple batch sizes. Of course, the actual performance is hardware dependent.

```
single-threaded performance:

run 1000 evaluations for batch size 1
-> 0.118 ms per batch

run 1000 evaluations for batch size 10
-> 0.607 ms per batch

run 1000 evaluations for batch size 100
-> 3.83 ms per batch

run 1000 evaluations for batch size 1000
-> 36.687 ms per batch

--------------------------------------------
multi-threaded performance:

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

Any CMSSW 94X version will work (currently only available on integration branches):

```bash
source /cvmfs/cms.cern.ch/cmsset_default.sh

export SCRAM_ARCH="slc6_amd64_gcc630"
export CMSSW_VERSION="CMSSW_9_4_X_2017-11-09-1100"

cmsrel $CMSSW_VERSION
cd $CMSSW_VERSION/src
cmsenv

git clone https://gitlab.cern.ch/mrieger/CMSSW-DNN.git DNN

scram b
```


### Important Notes

##### Multi-threading

You can set the number of treads when loading a `GraphDef`, `MetaGraphDef`, and `Session`. By default, only one thread is used.

When loading a constant graph:

```cpp
// load the graph
tensorflow::GraphDef* graphDef = tensorflow::loadGraphDef("/path/to/constantgraph.pb");

// create a session and use multi-threading
tensorflow::Session* session = tensorflow::createSession(graphDef, 4);

// proceed as usual
...
```


When loading a saved model:

```cpp
// load the meta graph
tensorflow::MetaGraphDef* metaGraph = tensorflow::loadMetaGraph("/path/to/simplegraph", 4);

// create a session and use multi-threading
tensorflow::Session* session = tensorflow::createSession(metaGraph, "/path/to/simplegraph", 4);

// proceed as usual
...
```


##### Logging

By default, TensorFlow logging is quite verbose. This can be changed via setting the `TF_CPP_MIN_LOG_LEVEL` environment varibale before calling (e.g.) `cmsRun`, or via calling `tensorflow::setLogging(level)` in your code. Log levels:

| `TF_CPP_MIN_LOG_LEVEL` value | Verbosity level |
| ---------------------------- | --------------- |
| "0"                          | debug           |
| "1" (default)                | info            |
| "2"                          | warning         |
| "3"                          | error           |
| "4"                          | none            |

Forwarding to the `MessageLogger` service is not possible yet.
