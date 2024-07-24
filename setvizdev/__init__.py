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
import os

from .setvizdev import *
from .setvizdev import _ntasks_on_node

try:
    gpus_on_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    gpus_per_task(gpus_on_node // _ntasks_on_node)
except (KeyError, ValueError):
    pass

__all__ = ["gpus_per_task"]

__version__ = version(__package__)

del version, os, setvizdev, _ntasks_on_node
