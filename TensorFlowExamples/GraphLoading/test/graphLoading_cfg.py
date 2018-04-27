# -*- coding: utf-8 -*-

"""
Test config to run the GraphLoading example.
"""


import os
import subprocess

import FWCore.ParameterSet.Config as cms


# helper to determine the location if _this_ file
def get_this_dir():
    if "__file__" in globals():
        return os.path.dirname(os.path.abspath(__file__))
    else:
        return os.path.expandvars("$CMSSW_BASE/src/TensorFlowExamples/GraphLoading/test")


# ensure that the graph exists
# if not, call the create_graph.py script in a subprocess since tensorflow complains
# when its loaded twice (once here in python, once in c++)
graph_path = os.path.abspath("graph.pb")
if not os.path.exists(graph_path):
    script_path = os.path.join(get_this_dir(), "create_graph.py")
    code = subprocess.call(["python", script_path, graph_path])
    if code != 0:
        raise Exception("create_graph.py failed")


# define the process to run
process = cms.Process("TF")

# minimal configuration
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(10))
process.source = cms.Source("PoolSource", fileNames=cms.untracked.vstring("root://xrootd-cms.infn.it//store/mc/RunIIFall17MiniAOD/DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/94X_mc2017_realistic_v10-v2/00000/9A439935-1FFF-E711-AE07-D4AE5269F5FF.root"))

# load the graphLoading example module
process.load("TensorFlowExamples.GraphLoading.graphLoading_cfi")
process.graphLoading.graphPath = cms.string(graph_path)

# define the path to run
process.p = cms.Path(process.graphLoading)
