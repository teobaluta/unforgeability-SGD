import struct
import numpy as np
import ctypes
from numpy.ctypeslib import ndpointer
import os


def get_lsb_fast(grads: np.ndarray, precision: int, path: str, mode: str):
    rows = grads.shape[0]
    cols = grads.shape[1]

    getLSB = ctypes.cdll.LoadLibrary('./lib/librref.so').getGradsLSB
    getLSB.argtypes = [
        ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
        ctypes.c_uint,
        ctypes.c_uint64,
        ctypes.c_uint64,
    ]
    getLSB.restype = ctypes.c_void_p

    free_pointer = ctypes.cdll.LoadLibrary('./lib/librref.so').free_pointer
    free_pointer.argtypes = [ctypes.c_void_p]
    free_pointer.restype = None

    grads = np.ascontiguousarray(grads)
    s = getLSB(grads, precision, rows, cols)
    decoded_s = ctypes.cast(s, ctypes.c_char_p).value
    decoded_s = decoded_s.decode('utf-8')
    with open(path, mode) as f:
        f.write(decoded_s)
    free_pointer(s)
    del decoded_s