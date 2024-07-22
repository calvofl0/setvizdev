#!/bin/env python3
# -*- coding: utf-8 -*-

import functools
import os
import socket

import numpy as np
from mpi4py import MPI


_ngpus = 1

# MPI settings
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
sz = comm.Get_size()

# Hostname hashes
# DJB2 python implementation borrowed from https://gist.github.com/amakukha/7854a3e910cb5866b53bf4b2af1af968
djb2_hash = lambda x: functools.reduce(lambda x, c: 0xFFFFFFFF & (x * 33 + c), x, 5381)
localhash = djb2_hash([ord(c) for c in socket.gethostname()])
hosthashes = np.empty(sz, dtype=np.int64)
comm.Allgather(np.array([localhash], dtype=np.int64), hosthashes)

# Compute local (within node) rank
localRank = 0
for r, h in enumerate(hosthashes):
    if rank == r:
        break
    if h == localhash:
        localRank += 1


def gpus_per_task(ngpus: int = None):
    """
    Set the number of gpus to be visible per task

    :param ngpus: Number of gpus (default: 1)

    :return: Number of gpus
    """
    if not ngpus is None:
        _ngpus = ngpus
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(r) for r in range(ngpus * localRank, ngpus * (localRank + 1))]
        )

    return _ngpus


gpus_per_task(_ngpus)

__all__ = ["gpus_per_task"]
