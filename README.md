## DNN Interface in CMSSW

#### Installation

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


#### Authors

- [Marcel Rieger](mailto:marcel.rieger@cern.ch): Tensorflow interface
- [Marco Harrendorf](mailto:marco.harrendorf@cern.ch): Tensorflow bundle
