from __future__ import annotations

import ctypes
import os
import subprocess

import cpuinfo
import numpy as np
import torch


cpu = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LIBDSC = ctypes.cdll.LoadLibrary("lib/libdsc.so")

LIBDSC.heap_alloc.restype = ctypes.c_void_p
LIBDSC.astar_init_state_map.restype = ctypes.c_void_p
LIBDSC.astar_get_back_actions.restype = ctypes.c_char_p

def ptr(arr: torch.Tensor) -> ctypes:
    return ctypes.c_void_p(arr.data_ptr())

def tensor_size(x: np.ndarray | torch.Tensor) -> int:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.element_size() * x.numel()

def unique(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """ Returns a tensor with the indices of unique elements in x
    as well as a tensor of inverse indices. """
    index = torch.empty(len(x), dtype=torch.long)
    inverse = torch.empty(len(x), dtype=torch.long)

    numel_per_element = np.cumprod((*x.shape[1:], 1))[-1]

    num_unique = LIBDSC.unique(
        ptr(x),
        ctypes.c_size_t(len(x)),
        ctypes.c_size_t(x.element_size() * numel_per_element),
        ptr(index),
        ptr(inverse),
    )

    index = index[:num_unique]

    return index, inverse

class HardwareInfo:

    cpu     = cpuinfo.get_cpu_info()["brand_raw"]
    sockets = int(subprocess.check_output('cat /proc/cpuinfo | grep "physical id" | sort -u | wc -l', shell=True))
    cores   = os.cpu_count()
    gpu     = torch.cuda.get_device_name(device) if torch.cuda.is_available() else None
