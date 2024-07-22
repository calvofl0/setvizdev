#!/bin/env python3
# -*- coding: utf-8 -*-

"""
SetVizDev
=========

Simple tool providing the function `gpus_per_task`, which sets the
CUDA_VISIBLE_DEVICES environment variable so that only the appropriate
gpus are visible to each MPI task.
"""

from importlib.metadata import version

from .setvizdev import *

del setvizdev

__all__ = ["gpus_per_task"]

__version__ = version(__package__)
