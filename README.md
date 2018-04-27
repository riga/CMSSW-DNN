## TensorFlow Interface for CMSSW

[![build status](https://gitlab.cern.ch/mrieger/CMSSW-DNN/badges/master/build.svg)](https://gitlab.cern.ch/mrieger/CMSSW-DNN/pipelines)

- Main repository & issues: [gitlab.cern.ch/mrieger/CMSSW-DNN](https://gitlab.cern.ch/mrieger/CMSSW-DNN)
- Code mirror: [github.com/riga/CMSSW-DNN](https://github.com/riga/CMSSW-DNN)

### Note

The interface was merged under [PhysicsTools/TensorFlow](https://github.com/cms-sw/cmssw/tree/master/PhysicsTools/TensorFlow) on Jan 25 2018 into [CMSSW\_10\_1\_X](https://github.com/cms-sw/cmssw/pull/19893) and backported to [CMSSW\_9\_4\_X](https://github.com/cms-sw/cmssw/pull/22042) on Feb 15 2018.

---

This interface provides simple and fast access to [TensorFlow](https://www.tensorflow.org) in CMSSW and lets you evaluate trained models right within your C++ modules. It **does not depend** on a converter library or custom NN implementation. In fact, it is a thin layer on top of TensorFlow's C++ API (available via exernals in `/cvmfs`) which handles session / graph loading & cleanup, exceptions, and thread management within CMSSW. As a result, you can load and evaluate every model that was previously trained and saved in Python (or C++).

Due to the development of the CMS software environment since 8\_0\_X, there are multiple versions of this interface. But since the C++ API was added in 9\_4\_X, the interface API is stable and should handle all changes within TensorFlow internally. The following table summarizes all available versions, mapped to CMSSW version and SCRAM\_ARCH:

| CMSSW version |     SCRAM\_ARCH     | TF API & version (externals) |                          Interface branch                         |
| ------------- | ------------------- | ---------------------------- | ----------------------------------------------------------------- |
| t.b.a.        | slc6\_amd64\_gcc630 | C++, 1.6.0                   | [tf\_cc\_1.6](/../tree/tf_cc_1.6)                                 |
| 10\_1\_X      | slc6\_amd64\_gcc630 | C++, 1.5.0                   | [tf\_cc\_1.5](/../tree/tf_cc_1.5)                                 |
| 10\_0\_X      | slc6\_amd64\_gcc630 | C++, 1.3.0                   | [tf\_cc\_1.3](/../tree/tf_cc_1.3) / **[master](/../tree/master)** |
| 9\_4\_X       | slc6\_amd64\_gcc630 | C++, 1.3.0                   | [tf\_cc\_1.3](/../tree/tf_cc_1.3) / **[master](/../tree/master)** |
| 9\_3\_X       | slc6\_amd64\_gcc630 | C, 1.1.0                     | [tf\_c](/../tree/tf_c)                                            |
| 8\_0\_X       | slc6\_amd64\_gcc530 | Py + CPython, 1.1.0          | [tf\_py\_cpython](/../tree/tf_py_cpython)                         |


### Examples

- [`TensorFlowExamples/GraphLoading`](./TensorFlowExamples/GraphLoading): Graph loading and evaluation in a CMSSW plugin.


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
constant_graph = tf.graph_util.convert_variables_to_constants(
    sess, sess.graph.as_graph_def(), outputs)
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

For more examples, see [`TensorFlow/test/testGraphLoading.cc`](./TensorFlow/test/testGraphLoading.cc) or a complete CMSSW example plugin at [`TensorFlowExamples/GraphLoading`](./TensorFlowExamples/GraphLoading).


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


### Installation

Any CMSSW version starting from 9.4.X will work:

```bash
source /cvmfs/cms.cern.ch/cmsset_default.sh

export SCRAM_ARCH="slc6_amd64_gcc630"
export CMSSW_VERSION="CMSSW_9_4_6_patch1"

cmsrel "$CMSSW_VERSION"
cd "$CMSSW_VERSION/src"
cmsenv

git clone https://gitlab.cern.ch/mrieger/CMSSW-DNN.git DNN

scram b
```


### Important Notes

#### Keras

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
constant_graph = tf.graph_util.convert_variables_to_constants(
    sess, sess.graph.as_graph_def(), outputs)
tf.train.write_graph(constant_graph, "/path/to", "constantgraph.pb", as_text=False)

# save it as a SavedModel
builder = tf.saved_model.builder.SavedModelBuilder("/path/to/simplegraph")
builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
builder.save()
```

#### `BuildFile.xml`'s

If you are aiming to use the TensorFlow interface in your personal CMSSW plugin (!), make sure to include the following lines in your `/plugins/BuildFile.xml`:

```xml
<use name="PhysicsTools/TensorFlow" />
```

If you are using the interface in a file in the `/src/` or `/interface/` directory of your module, make sure to create a (global) `/BuildFile.xml` containing (at least):

```xml
<use name="PhysicsTools/TensorFlow" />

<export>
    <lib name="1" />
</export>
```


#### TensorFlow in `cmsRun` config files

Please make sure you do not import the TensorFlow python module in a CMSSW Python configuration files. TensorFlow will crash when it is loaded again within C++ code. Currently, there is no way to shutdown the TensorFlow environment within Python.


#### Multi-threading

You can set the number of threads when loading a `GraphDef`, `MetaGraphDef`, and `Session`. By default, only one thread is used but it's easy to use more.

When loading a constant graph:

```cpp
// load the graph
tensorflow::GraphDef* graphDef = tensorflow::loadGraphDef("/path/to/constantgraph.pb");

// create a session and use 4 threads
tensorflow::Session* session = tensorflow::createSession(graphDef, 4);

// proceed as usual
...
```


When loading a saved model:

```cpp
// load the meta graph
tensorflow::MetaGraphDef* metaGraph = tensorflow::loadMetaGraph("/path/to/simplegraph", 4);

// create a session and use 4 threads
tensorflow::Session* session = tensorflow::createSession(metaGraph, "/path/to/simplegraph", 4);

// proceed as usual
...
```


#### Logging

By default, TensorFlow logging is quite verbose. This can be changed via setting the `TF_CPP_MIN_LOG_LEVEL` environment varibale before calling (e.g.) `cmsRun`, or via calling `tensorflow::setLogging(level)` in your code. Log levels:

| `TF_CPP_MIN_LOG_LEVEL` value | Verbosity level |
| ---------------------------- | --------------- |
| "0"                          | debug           |
| "1" (default)                | info            |
| "2"                          | warning         |
| "3"                          | error           |
| "4"                          | none            |

Forwarding to the `MessageLogger` service is not possible yet.
