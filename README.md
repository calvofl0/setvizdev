# Simple tool for setting the CUDA visible devices

This tool appropriately sets the `CUDA_VISIBLE_DEVICES` environment variable. When running in a multi-node-multi-gpu setting, the GPUs of a given node might all be available to all tasks resident in the node. If might then be necessary to properly define `CUDA_VISIBLE_DEVICES`, so that any given GPU is assigned to one single MPI task.

The procedure is based on the example available on the [NCCL example](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html#example-2-one-device-per-process-or-thread) in the official documentation.

## Usage

In order to assign one single GPU per task, just import the module:

```python
>>> from setvizdev import *
```

Set a different number of gpus per task `ngpus`:

```python
>>> gpus_per_task(ngpus)
```

Voilà.