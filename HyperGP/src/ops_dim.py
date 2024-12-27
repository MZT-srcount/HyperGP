from .ndarray import NDArray, prob, default_device, ops_run
from .data_type import _out_dtype
import numpy as np


def substract(a, b, dim_0=0, dim_1=0):
    return ops_run(a.device.ewise_sub_dim, a, b, dim_0, dim_1)

def add(a, b, dim_0=0, dim_1=0):
    return ops_run(a.device.ewise_add_dim, a, b, dim_0, dim_1)

def multiply(a, b, dim_0=0, dim_1=0):
    return ops_run(a.device.ewise_mul_dim, a, b, dim_0, dim_1)

def divide(a, b, dim_0=0, dim_1=0):
    return ops_run(a.device.ewise_div_dim, a, b, dim_0, dim_1)

def concatenate(arrays:tuple, dim=0, device=None):
    assert isinstance(arrays, tuple), "The inputs should be organized as tuple"
    assert all([isinstance(array, NDArray) for array in arrays]), "{TYPE}".format(TYPE=[type(array) for array in arrays])
    cdds = [i for i in range(len(arrays)) if prob(arrays[i].shape) > 0]
    pre_size = 1 if dim == 0 else prob(arrays[cdds[0]].shape[:dim])
    post_len = prob(arrays[cdds[0]].shape[dim + 1:])
    if not all([arrays[cdd].shape[:dim] == arrays[cdds[0]].shape[:dim] and arrays[cdd].shape[dim + 1:] == arrays[cdds[0]].shape[dim + 1:] for cdd in cdds]):
        raise ValueError("all the input array dimensions except for the concatenation axis must match exactly")
    new_shape = arrays[cdds[0]].shape[:dim] + (np.sum([arrays[cdd].shape[dim] for cdd in cdds]), ) + arrays[cdds[0]].shape[dim + 1:]
    dtype = arrays[0].dtype
    for i in range(len(arrays)):
        dtype = _out_dtype(arrays[i].dtype, dtype)
    new_array = NDArray.make(shape=tuple(new_shape), device=default_device() if device is None else device, dtype=dtype)
    accum_posi = 0
    for cdd in cdds:
        array = arrays[cdd]
        offset = new_array._stride[dim] * new_array._shape[dim]
        new_array.device.concatenate(new_array._handle, array._handle, pre_size, array._stride[dim] * array._shape[dim], accum_posi, offset, array._offset)
        accum_posi += array._stride[dim] * array._shape[dim]
    return new_array
