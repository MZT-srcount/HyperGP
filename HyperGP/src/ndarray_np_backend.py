import numpy as np

__device_name__ = "numpy"
_datatype = np.float64
_datatype_size = np.dtype(_datatype).itemsize

class Array:
    def __init__(self, shape):
        self.array = np.empty(shape, _datatype)

    @property
    def size(self):
        return self.array.size

def from_numpy(out, array):
    out.array[:] =  array.flatten()

def to_numpy(a, shape, strides, offset):
    return np.lib.stride_tricks.as_strided(
        a.array[offset:], shape, tuple([s * _datatype_size for s in strides])
    )

def ewise_add(a, b, out):
    out[:] = a + b

def scalar_add(a, val, out):
    out[:] = a + val

def ewise_sub(a, b, out):
    out[:] = a - b

def scalar_sub(a, val, out):
    out[:] = a - val

def ewise_mul(a, b, out):
    out[:] = a * b

def scalar_mul(a, val, out):
    out[:] = a * val

def ewise_div(a, b, out):
    out[:] = a / b

def scalar_div(a, val, out):
    out[:] = a / val

def scalar_div(a, val, out):
    out[:] = a / val

