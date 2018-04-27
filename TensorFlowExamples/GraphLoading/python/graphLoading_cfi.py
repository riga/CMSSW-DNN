# -*- coding: utf-8 -*-

"""
Initialization file for the GraphLoading example module.
"""


__all__ = ["graphLoading"]


import FWCore.ParameterSet.Config as cms


graphLoading = cms.EDAnalyzer("GraphLoading",
    graphPath=cms.string("graph.pb"),
)
