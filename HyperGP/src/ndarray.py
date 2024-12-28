import numpy as np
from .device import BackendDevice
# from . import ndarray_cpu_backend
# from . import ndarray_np_backend
from functools import reduce
import operator
import time
from .data_type import _supported_dtype, _supported_builtin_dtype, _out_dtype, _dtype_strmap
from .data_type import _bool, _int32, _int64, _int8, _float32, _float64, _uint16, _uint32, _uint64, _uint8
from .data_type import *
from collections.abc import Iterable
import math


__default_devid__ = 0


def prob(x):
    return reduce(operator.mul, x, 1)

def _cumsum(x):
    return reduce(operator.add, x, 0)

def _array(data, dtype=float, device=None, device_id=__default_devid__):
    '''[ ] TODO: dtype should be applied later..'''
    # if 
    return NDArray(data, dtype, device=device, device_id=device_id)

# def _array(dtype, size=0):
#     if dtype="float32":
        


def cpu_numpy():
    return BackendDevice('cpu_numpy', ndarray_np_backend)

def cpu():
    return BackendDevice("cpu", ndarray_cpu_backend)

def gpu():
    from . import ndarray_cuda_backend
    return BackendDevice("gpu", ndarray_cuda_backend)

#[ ]TODO: change the computation way through changing the default_device
def default_device():
    return gpu()

def get_porperties():
    from . import ndarray_cuda_backend
    return BackendDevice("gpu", ndarray_cuda_backend).get_properties()

def check_gpu():
    gpu().check()

def device(device_id):
    global __default_devid__
    __default_devid__ = device_id

def query_device():
    return __default_devid__


def ops_run(runner, a, b, dim_0=0, dim_1=0):
    if not isinstance(a, NDArray):
        a = NDArray(a)

    assert abs(dim_0) < len(a._shape) or (dim_0 < 0 and -dim_0 <= len(a._shape)) or (dim_0 == 0 and len(a._shape) == 0), "input dim should smaller than array shape, where shape: {A} / dim: {B}".format(A=a._shape, B=dim_0)
    if dim_0 != 0:
        pre_dim_a, post_dim_a = prob(a._shape[:dim_0]), prob(a._shape[dim_0:])
    else:
        pre_dim_a, post_dim_a = 1, prob(a._shape)

    if not isinstance(b, NDArray):
        b = NDArray(b)
    
    assert abs(dim_1) < len(b._shape) or (dim_1 < 0 and -dim_1 <= len(b._shape)) or (dim_1 == 0 and len(b._shape) == 0), "input dim should smaller than array shape, where shape: {A} / dim: {B}".format(A=b._shape, B=dim_1)
    if dim_1 != 0:
        pre_dim_b, post_dim_b = prob(b._shape[:dim_1]), prob(b._shape[dim_1:])
    else:
        pre_dim_b, post_dim_b = 1, prob(b._shape)

    out = NDArray.make(a.shape, device=a.device, dtype=_out_dtype(a.dtype, b.dtype)) if prob(a.shape) > prob(b.shape) else NDArray.make(b.shape, device=b.device, dtype=_out_dtype(a.dtype, b.dtype))
    assert (pre_dim_a == pre_dim_b or (pre_dim_a % pre_dim_b == 0 and post_dim_a > post_dim_b) or (pre_dim_a % pre_dim_b == 0 and post_dim_b > post_dim_a), 
            "The input size is not equal {dim_a}:{dim_b}".format(dim_a=a._shape[:dim_0 + 1], dim_b=b._shape[:dim_1 + 1])
            )
    # if pre_dim_a != pre_dim_b and not (pre_dim_a % pre_dim_b == 0 and post_dim_a > post_dim_b) and not (pre_dim_a % pre_dim_b == 0 and post_dim_b > post_dim_a):
    #     UserWarning("The input size is not equal {dim_a}:{dim_b}".format(a._shape[:dim_0 + 1], b._shape[:dim_1 + 1]))
    assert post_dim_a == post_dim_b or post_dim_b == 1 or post_dim_a == 1, "input size not equal to size in dim, %d!=%d"%(post_dim_a, post_dim_b)
    runner(a._handle, b._handle, out._handle, pre_dim_a, post_dim_a, pre_dim_b, post_dim_b, a._offset, b._offset)
    return out

#[ ] TODO: The output dtype should be considered, especially for the opers with 2 dims or more 
class NDArray:
    def __init__(self, other, dtype, device=None, device_id=__default_devid__):
        if isinstance(other, NDArray):
            if device is None:
                self._init(other)
            else:
                self._init(other.to(device))
        elif isinstance(other, np.ndarray):
            device = default_device() if device is None else device
            dtype = _supported_dtype(other.dtype) if dtype==None else dtype
            if datatype_mapping[dtype] != other.dtype:
                other = np.array(other)
            array = NDArray.make(other.shape, device_id=device_id, device=device, dtype=dtype)
            array.device.from_numpy(np.ascontiguousarray(other), array._handle)
            self._init(array)
        else:
            array = NDArray(np.array(other), dtype, device=device)
            self._init(array)

    def is_compact(self):
        """Return true if array is compact in memory and internal size equals product
        of the shape dimensions"""
        return (
            self._stride == NDArray.compact_strides(self._shape)
            and prob(self.shape) == self._handle.size
        )
    
    def compact(self):
        """ Convert a matrix to be compact """
        if self.is_compact():
            return self
        else:
            out = NDArray.make(self.shape, self._handle.dev_id, device=self.device, dtype=self.dtype)
            self.device.compact(
                self._handle, out._handle, self.shape, self._stride, self._offset
            )
            return out

    def _init(self, other):
        self._handle = other._handle
        self._device = other._device
        self._shape = other._shape
        self._stride = other._stride
        self._offset = other._offset
        self._dtype = other._dtype

    def __getstate__(self):
        return (self.numpy(), self._dtype.__name__, self._device.name)

    def __setstate__(self, state):
        array = NDArray(state[0], state[1], gpu() if state[2] == "gpu" else cpu())
        self._init(array)

    
    def _setstate(state):
        return NDArray(state[0], _dtype_strmap[state[1]], gpu() if state[2] == "gpu" else cpu())

    # To get the index list from the input slice
    def _get_unit_list(self, idxs):
        unit_idxs = []
        new_steps = [list(range(sl.start, sl.stop, sl.step)) if isinstance(sl, slice) else list(sl) for sl in idxs]
        stack = [(step, 0) for step in reversed(new_steps[0])]
        accumulation_1 = np.zeros(shape=len(new_steps))
        unit_len = self._stride[len(new_steps) - 1]
        if len(new_steps) == 1:
            unit_idxs.extend([step * self._stride[0] for step in new_steps[0]])
        while len(stack) > 0:
            origin_step, idx = stack.pop()
            accumulation_1[idx] = origin_step * self._stride[idx]
            if idx < len(new_steps) - 2:
                stack.extend([(step, idx + 1) for step in reversed(new_steps[idx + 1])])
            elif idx == len(new_steps) - 2:
                accum_sum = np.sum(accumulation_1)
                unit_idxs.extend([int(accum_sum + step * unit_len) for step in new_steps[idx + 1]])
        return unit_idxs, unit_len
        
    #[ ] TODO: It seems the offset compute in each opers is wrong
    def __getitem__(self, idxs):
        # print("idxs", idxs)
        if not isinstance(idxs, tuple):
            idxs = (idxs, )
        assert len(idxs) <= len(self.shape), "Need indexes leq to number of dimensions"
        
        assert all([isinstance(idx, slice) or isinstance(idx, list) or isinstance(idx, int) for idx in idxs]),\
        "The index type should be the same with seveal selection: int, list, slice, while the current is '{A}'".format(A=[type(idx) for idx in idxs])

        idxs = [
            self._slice_process(sl, i)  if isinstance(sl, slice) else ([sl] if not isinstance(sl, list) else sl)
            for i, sl in enumerate(idxs)
        ]

        new_shape = [(sl.stop - sl.start + sl.step - 1) // sl.step if isinstance(sl, slice) else len(sl) for sl in idxs]
        
        new_shape_tmp = new_shape.copy()
        if (len(idxs) < len(self._shape)):
            begin_posi = len(new_shape)
            new_shape = self._completion(new_shape, begin_posi)
        
        """use _offset to avoid the frequent data transfer"""
        if len(idxs) == 1:
            if isinstance(idxs[0], list) and isinstance(idxs[0][0], int):
                idx = idxs[0][0]
                new_array = NDArray.make(tuple(new_shape), self._handle.dev_id, handle=self._handle, offset=idx * self._stride[0] + self._offset, device=self.device, dtype=self.dtype)
                return new_array
            if isinstance(idxs[0], slice) and idxs[0].step == 1:
                new_array = NDArray.make(tuple(new_shape), self._handle.dev_id, handle=self._handle, offset=idxs[0].start * self._stride[0] + self._offset, device=self.device, dtype=self.dtype)
                return new_array
        else:
            if all([isinstance(idx, list) and len(idx) == 1 for idx in idxs]):
                new_array = NDArray.make(tuple(new_shape), self._handle.dev_id, handle=self._handle, offset=sum([idx[0] * self._stride[i] for i, idx in enumerate(idxs)]) + self._offset, device=self.device, dtype=self.dtype)
                return new_array
        
        new_array = NDArray.make(tuple(new_shape), self._handle.dev_id, device=self.device, dtype=self.dtype)

        step_list = [idx * self._stride[i] for i, sl in enumerate(idxs) for idx in (range(sl.start, sl.stop, sl.step) if isinstance(sl, slice) else sl)]
        step_sizes = new_shape_tmp
        self.device.transfer(new_array._handle, self._handle, step_list, step_sizes, self._stride[len(idxs) - 1], self._offset, 0, 0)
        # unit_idxs, unit_len = self._get_unit_list(idxs)
        # self.device.old_transfer(new_array._handle, self._handle, unit_idxs, unit_len, self._offset, 0)

        return new_array
    
    #[ ]TODO: __getitem__ returns a new array instead of the whole origin array
    def __setitem__(self, idxs, other):
        
        if not isinstance(idxs, tuple):
            idxs = (idxs, )
        if not isinstance(other, NDArray):
            other = NDArray(np.array(other), dtype=self._dtype, device=self.device)

        assert not any([isinstance(sl, list) for sl in idxs]), "There should be slice instead of list object in input idxs"
        assert all([isinstance(idx, slice) or isinstance(idx, list) or isinstance(idx, int) for idx in idxs]),\
        "The index type should be the same with seveal selection: int, list, slice, while the current is '{A}'".format(A=[type(idx) for idx in idxs])
        
        idxs = [
            self._slice_process(sl, i)  if isinstance(sl, slice) else ([sl] if not isinstance(sl, list) else sl)
            for i, sl in enumerate(idxs)
        ]

        step_list = [idx * self._stride[i] for i, sl in enumerate(idxs) for idx in (range(sl.start, sl.stop, sl.step) if isinstance(sl, slice) else sl)]
        step_sizes = [(sl.stop - sl.start + sl.step - 1) // sl.step if isinstance(sl, slice) else len(sl) for sl in idxs]
        # unit_idxs, unit_len = self._get_unit_list(idxs)
        # self.device.old_transfer(self._handle, other._handle, unit_idxs, unit_len, self._offset, 1)

        self.device.transfer(self._handle, other._handle, step_list, step_sizes, self._stride[len(idxs) - 1], other._offset, self._offset, 1)

        
        
    def _array(self, size, dtype, device_id):
        if dtype==_float32:
            return self.device.Array_f(size, device_id)
        if dtype==_float64:
            return self.device.Array_d(size, device_id)
        if dtype==_int64:
            return self.device.Array_l(size, device_id)
        if dtype==_int32:
            return self.device.Array_i(size, device_id)
        if dtype==_int8:
            return self.device.Array_a(size, device_id)
        if dtype==_uint8:
            return self.device.Array_h(size, device_id)
        if dtype==_uint16:
            return self.device.Array_t(size, device_id)
        if dtype==_uint32:
            return self.device.Array_j(size, device_id)
        if dtype==_uint64:
            return self.device.Array_m(size, device_id)
        if dtype==_bool:
            return self.device.Array_b(size, device_id)
        raise NotImplementedError("The datatype '{DT}' is not supported in the current version".format(DT=dtype))
    
        
    def make(shape, device_id=__default_devid__, strides=None, device=None, handle=None, offset=0, dtype=None):
        array = NDArray.__new__(NDArray)
        array._shape = shape
        array._stride = NDArray.compact_strides(shape) if strides is None else strides
        array._device = device if device is not None else default_device()
        array._handle = handle if handle is not None else array._array(prob(shape), _supported_dtype(dtype), device_id)
        array._offset = offset
        array._dtype = _supported_dtype(dtype)
        return array

    def compact_strides(shape):
        stride = 1
        res = []
        for i in range(1, len(shape) + 1):
            res.append(stride)
            stride *= shape[-i]
        return tuple(res[::-1])
    
    @property
    def device(self):
        return self._device

    @property
    def shape(self):
        return self._shape

    @property
    def offset(self):
        return self._offset

    def reshape(self, shape:tuple):
        if isinstance(shape, int):
            shape = (shape, )
        s_num = [i for i, s in enumerate(shape) if s == -1]#np.where(shape == -1)[0]
        assert len(s_num) <= 1, "the number of -1 in new_shape should smaller than 1"
        new_prob = prob(shape)
        origin_prob = prob(self._shape)
        if len(s_num) > 0:
            shape = shape[:s_num[0]] + (int(-origin_prob / new_prob), ) + (shape[s_num[0] + 1:] if s_num[0] + 1 < len(shape) else ())
            new_prob = prob(shape)

        assert origin_prob == new_prob, "the size of the new shape should keep the same with the origin shape, origin-new:{A}-{B}".format(A=origin_prob, B=new_prob)
        return NDArray.make(shape, self._handle.dev_id, NDArray.compact_strides(shape), self._device, self._handle, self._offset, dtype=self._dtype)

    def to(self, device):
        if device == self.device:
            return self
        else:
            array = self.device.to_numpy(self._shape, self._stride, self._offset)
            return NDArray(array, device)
    
    def numpy(self):
        return self.device.to_numpy(self._handle, self._shape, self._stride, self._offset)

    def ptr(self):
        return self._handle.ptr()

    def __add__(self, other):
        out = NDArray.make(self.shape, self._handle.dev_id, device=self.device, dtype=_out_dtype(self.dtype, other.dtype) if isinstance(other, NDArray) else self.dtype)
        if isinstance(other, NDArray):
            if other.shape != self.shape:
                ops_run(self.device.ewise_add_dim, self, other)
            else:
                # assert self.shape == other.shape, "operation needs two equal-sized arrays, where a/b:{A1}/{A2}".format(A1=self.shape, A2=other.shape)
                self.device.ewise_add(self._handle, other._handle, out._handle, self._offset, other._offset)
        else:
            self.device.scalar_add(self._handle, other, out._handle, self._offset)
        return out

    def __sub__(self, other):
        out = NDArray.make(self.shape, self._handle.dev_id, device=self.device, dtype=_out_dtype(self.dtype, other.dtype) if isinstance(other, NDArray) else self.dtype)
        if isinstance(other, NDArray):
            if other.shape != self.shape:
                ops_run(self.device.ewise_sub_dim, self, other)
            else:
                # assert self.shape == other.shape, "operation needs two equal-sized arrays, {s1}:{s2}".format(s1=self.shape, s2=other.shape)
                self.device.ewise_sub(self._handle, other._handle, out._handle, self._offset, other._offset)
        else:
            self.device.scalar_sub(self._handle, other, out._handle, self._offset)
        return out

    def __mul__(self, other):
        out = NDArray.make(self.shape, self._handle.dev_id, device=self.device, dtype=_out_dtype(self.dtype, other.dtype) if isinstance(other, NDArray) else self.dtype)
        if isinstance(other, NDArray):
            if other.shape != self.shape:
                ops_run(self.device.ewise_mul_dim, self, other)
            else:
                # assert self.shape == other.shape, "operation needs two equal-sized arrays, where a/b:{A1}/{A2}".format(A1=self.shape, A2=other.shape)
                self.device.ewise_mul(self._handle, other._handle, out._handle, self._offset, other._offset)
        else:
            self.device.scalar_mul(self._handle, other, out._handle, self._offset)
        return out
           
    def __truediv__(self, other):
        out = NDArray.make(self.shape, self._handle.dev_id, device=self.device, dtype=_out_dtype(self.dtype, other.dtype) if isinstance(other, NDArray) else self.dtype)
        if isinstance(other, NDArray):
            if other.shape != self.shape:
                ops_run(self.device.ewise_div_dim, self, other)
            else:
                # assert self.shape == other.shape, "operation needs two equal-sized arrays, where a/b:{A1}/{A2}".format(A1=self.shape, A2=other.shape)
                self.device.ewise_div(self._handle, other._handle, out._handle, self._offset, other._offset)
        else:
            self.device.scalar_div(self._handle, other, out._handle, self._offset)
        return out
    
    def pow(self, other):
        out = NDArray.make(self.shape, self._handle.dev_id, device=self.device, dtype=self._dtype)
        if isinstance(other, NDArray):
            assert self.shape == other.shape, "operation needs two equal-sized arrays, where a/b:{A1}/{A2}".format(A1=self.shape, A2=other.shape)
            self.device.ewise_pow(self._handle, other._handle, out._handle, self._offset, other._offset)
        else:
            self.device.scalar_pow(self._handle, other, out._handle, self._offset)
        return out
    
    def __lt__(self, other):
        out = NDArray.make(self.shape, self._handle.dev_id, device=self.device, dtype=_bool)
        if isinstance(other, NDArray):
            assert self.shape == other.shape, "operation needs two equal-sized arrays, where a/b:{A1}/{A2}".format(A1=self.shape, A2=other.shape)
            self.device.ewise_lt(self._handle, other._handle, out._handle, self._offset, other._offset)
        else:
            self.device.scalar_lt(self._handle, other, out._handle, self._offset)
        return out.numpy()
    
    def __le__(self, other):
        out = NDArray.make(self.shape, self._handle.dev_id, device=self.device, dtype=_bool)
        if isinstance(other, NDArray):
            assert self.shape == other.shape, "operation needs two equal-sized arrays, where a/b:{A1}/{A2}".format(A1=self.shape, A2=other.shape)
            self.device.ewise_le(self._handle, other._handle, out._handle, self._offset, other._offset)
        else:
            self.device.scalar_le(self._handle, other, out._handle, self._offset)
        return out.numpy()
    
    def __gt__(self, other):
        out = NDArray.make(self.shape, self._handle.dev_id, device=self.device, dtype=_bool)
        if isinstance(other, NDArray):
            assert self.shape == other.shape, "operation needs two equal-sized arrays, where a/b:{A1}/{A2}".format(A1=self.shape, A2=other.shape)
            self.device.ewise_ge(self._handle, other._handle, out._handle, self._offset, other._offset)
        else:
            self.device.scalar_ge(self._handle, other, out._handle, self._offset)
        return out.numpy()

    def __iter__(self):
        return iter(self.numpy())

    def __ge__(self, other):
        out = NDArray.make(self.shape, self._handle.dev_id, device=self.device, dtype=_bool)
        if isinstance(other, NDArray):
            assert self.shape == other.shape, "operation needs two equal-sized arrays, where a/b:{A1}/{A2}".format(A1=self.shape, A2=other.shape)
            self.device.ewise_ge(self._handle, other._handle, out._handle, self._offset, other._offset)
        else:
            self.device.scalar_ge(self._handle, other, out._handle, self._offset)
        return out.numpy()

    def __eq__(self, other):
        out = NDArray.make(self.shape, self._handle.dev_id, device=self.device, dtype=_bool)
        if isinstance(other, NDArray):
            assert self.shape == other.shape, "operation needs two equal-sized arrays, where a/b:{A1}/{A2}".format(A1=self.shape, A2=other.shape)
            self.device.ewise_eq(self._handle, other._handle, out._handle, self._offset, other._offset)
        else:
            self.device.scalar_eq(self._handle, other, out._handle, self._offset)
        return out.numpy()
    
    
    def __ne__(self, other):
        out = NDArray.make(self.shape, self._handle.dev_id, device=self.device, dtype=_bool)
        self.device.wait(self._handle)
        if isinstance(other, NDArray):
            assert self.shape == other.shape, "operation needs two equal-sized arrays"
            self.device.ewise_ne(self._handle, other._handle, out._handle, self._offset, other._offset)
        else:
            self.device.scalar_ne(self._handle, other, out._handle, self._offset)
        return out.numpy()
    
    def __neg__(self):
        out = NDArray.make(self.shape, self._handle.dev_id, device=self.device, dtype=self._dtype)
        self.device.ewise_neg(self._handle, out._handle, self._offset)
        return out

    def sin(self):
        out = NDArray.make(self.shape, self._handle.dev_id, device=self.device, dtype=self._dtype)
        self.device.ewise_sin(self._handle, out._handle, self._offset)
        return out

    def cos(self):
        out = NDArray.make(self.shape, self._handle.dev_id, device=self.device, dtype=self._dtype)
        self.device.ewise_cos(self._handle, out._handle, self._offset)
        return out
    
    def loge(self):
        out = NDArray.make(self.shape, self._handle.dev_id, device=self.device, dtype=self._dtype)
        self.device.ewise_loge(self._handle, out._handle, self._offset)
        return out
    
    def log10(self):
        out = NDArray.make(self.shape, self._handle.dev_id, device=self.device, dtype=self._dtype)
        self.device.ewise_log10(self._handle, out._handle, self._offset)
        return out
    
    def log2(self):
        out = NDArray.make(self.shape, self._handle.dev_id, device=self.device, dtype=self._dtype)
        self.device.ewise_log2(self._handle, out._handle, self._offset)
        return out
    
    def tab(self):
        out = NDArray.make(self.shape, self._handle.dev_id, device=self.device, dtype=self._dtype)
        self.device.ewise_tan(self._handle, out._handle, self._offset)
        return out
    
    def sqrt(self):
        out = NDArray.make(self.shape, self._handle.dev_id, device=self.device, dtype=self._dtype)
        self.device.ewise_sqrt(self._handle, out._handle, self._offset)
        return out
    
    def arcsin(self):
        out = NDArray.make(self.shape, self._handle.dev_id, device=self.device, dtype=self._dtype)
        self.device.ewise_arcsin(self._handle, out._handle, self._offset)
        return out

    def arccos(self):
        out = NDArray.make(self.shape, self._handle.dev_id, device=self.device, dtype=self._dtype)
        self.device.ewise_arccos(self._handle, out._handle, self._offset)
        return out
    
    def arctan(self):
        out = NDArray.make(self.shape, self._handle.dev_id, device=self.device, dtype=self._dtype)
        self.device.ewise_arctan(self._handle, out._handle, self._offset)
        return out
    
    def substract(self, other, dim_0=0, dim_1=0):
        assert abs(dim_0) < len(self._shape) or (dim_0 < 0 and -dim_0 <= len(self._shape)), "input dim should smaller than array shape"
        out = NDArray.make(self.shape, self._handle.dev_id, device=self.device, dtype=_out_dtype(self.dtype, other.dtype))
        if dim_0 != 0:
            pre_dim_a, post_dim_a = prob(self._shape[:dim_0]), prob(self._shape[dim_0:])
        else:
            pre_dim_a, post_dim_a = 1, prob(self._shape)

        if not isinstance(other, NDArray):
            other = NDArray(other)
            
        assert abs(dim_1) < len(other._shape) or (dim_1 < 0 and -dim_1 <= len(other._shape)), "input dim should smaller than array shape"
        if dim_1 != 0:
            pre_dim_b, post_dim_b = prob(other._shape[:dim_1]), prob(other._shape[dim_1:])
        else:
            pre_dim_b, post_dim_b = 1, prob(other._shape)

        if pre_dim_a != pre_dim_b and pre_dim_b != 1 and pre_dim_a != 1:
            UserWarning("The input size is not equal {dim_a}:{dim_b}".format(self._shape[:dim_0], other._shape[:dim_1]))
        assert post_dim_a == post_dim_b or post_dim_b == 1 or post_dim_a == 1, "input size not equal to size in dim, %d!=%d"%(post_dim_a, prob(other._shape))
        self.device.ewise_sub_dim(self._handle, other._handle, out._handle, pre_dim_a, post_dim_a, pre_dim_b, post_dim_b, self._offset, other._offset)
        return out

    def _ops_dim_1(self, dim, ops, dtype=None):
        assert abs(dim) < len(self._shape) or (dim < 0 and -dim <= len(self._shape)), "input dim should smaller than array shape, dim/shape: {D}/{S}".format(D=dim, S=self._shape)
        if dim != 0:
            out = NDArray.make(tuple(self._shape[:dim]), self._handle.dev_id, device=self.device, dtype=self._dtype if dtype is None else dtype)
            pre_dim, post_dim = prob(self._shape[:dim]), prob(self._shape[dim:])
        else:
            out = NDArray.make((1, ), self._handle.dev_id, device=self.device, dtype=self._dtype if dtype is None else dtype)
            pre_dim, post_dim = 1, prob(self._shape)
        ops(self._handle, out._handle, pre_dim, post_dim, self._offset)
        return out
    
    def abs(self):
        out = NDArray.make(self.shape, self._handle.dev_id, device=self.device, dtype=self._dtype)
        self.device.ewise_abs(self._handle, out._handle, self._offset)
        return out

    def exp(self):
        out = NDArray.make(self.shape, self._handle.dev_id, device=self.device, dtype=self._dtype)
        self.device.ewise_exp(self._handle, out._handle, self._offset)
        return out

    def ceil(self):
        out = NDArray.make(self.shape, self._handle.dev_id, device=self.device, dtype=self._dtype)
        self.device.ewise_ceil(self._handle, out._handle, self._offset)
        return out

    def floor(self):
        out = NDArray.make(self.shape, self._handle.dev_id, device=self.device, dtype=self._dtype)
        self.device.ewise_floor(self._handle, out._handle, self._offset)
        return out

    def sign(self):
        out = NDArray.make(self.shape, self._handle.dev_id, device=self.device, dtype=self._dtype)
        self.device.ewise_sign(self._handle, out._handle, self._offset)
        return out

    def sum(self, dim=0):
        return self._ops_dim_1(dim, self.device.ewise_sum)
    
    def min(self, dim):
        return self._ops_dim_1(dim, self.device.ewise_min)

    def max(self, dim):
        return self._ops_dim_1(dim, self.device.ewise_max)

    def mean(self, dim):
        return self._ops_dim_1(dim, self.device.ewise_mean, dtype=float_type(self._dtype))

    def argmax(self, dim):
        return self._ops_dim_1(dim, self.device.ewise_argmax, dtype=_int32)

    def argmin(self, dim):
        return self._ops_dim_1(dim, self.device.ewise_argmin, dtype=_int32)

    def std(self, dim):
        return self._ops_dim_1(dim, self.device.ewise_std, dtype=float_type(self._dtype))

    def var(self, dim):
        return self._ops_dim_1(dim, self.device.ewise_var, dtype=float_type(self._dtype))
    

    #[ ]TODO: cumsum、cumprob implement
    def cumsum(self, dim):
        raise NotImplementedError("cumsum operator is not implemented yet.")
        return self._ops_dim_1(dim, self.device.ewise_cumsum)

    def cumprob(self, dim):
        raise NotImplementedError("cumprob operator is not implemented yet.")
        return self._ops_dim_1(dim, self.device.ewise_cumprob)
    
    @property
    def dtype(self):
        # only support float32 for now
        return self._dtype
    
    def copy(self):
        out = NDArray.make(self.shape, self._handle.dev_id, device=self.device, dtype=self._dtype)
        self.device.compact(
            self._handle, out._handle, self.shape, self._stride, self._offset
        )
        self.wait()
        return out

    def view(self):
        out = NDArray.make(self.shape, self._handle.dev_id, self._stride, self.device, self._handle, self._offset, dtype=self._dtype)
        return out

    def wait(self):
        self.device.wait(self._handle)
        out = NDArray.make(self.shape, self._handle.dev_id, self._stride, self.device, self._handle, self._offset, dtype=self._dtype)
        return out

    
    def __str__(self):
        return self.numpy().__str__()
    
    
    def _completion(self, new_shape, begin_posi, new_stride=None):
        while(len(new_shape) > 0 and new_shape[-1] == 1):
            new_shape = new_shape[:-1]
            if new_stride is not None:
                new_stride = new_stride[:-1]
        new_shape.extend(list(self._shape[begin_posi:]))
        if new_stride is not None:
            new_stride.extend(self._stride[begin_posi:])
        if new_stride is not None:
            return new_shape, new_stride 
        else:
            return new_shape

    def _slice_process(self, sl, dim):
        start, stop, step = sl.start, sl.stop, sl.step
        if start is None:
            start = 0
        if start < 0:
            start = self.shape[dim] + start
        if stop is None:
            stop = self.shape[dim]
        if stop < 0:
            stop = self.shape[dim] + stop
        if step is None:
            step = 1
        
        assert stop > start, "Start must be less than stop"
        assert step > 0, "No support for  negative increments"
        return slice(start, stop, step)
    
    """ Matrix operation"""
    
    def T(self, dim=0):
        if dim >= 0:
            pre_dim_a, post_dim_a = prob(self._shape[:dim + 1]), prob(self._shape[dim + 1:])
        else:
            if len(self._shape) > 2:
                pre_dim_a, post_dim_a = prob(self._shape[:-2]), prob(self._shape[-2:])
            else:
                pre_dim_a, post_dim_a = 1, prob(self._shape)

        assert len(self._shape) - dim < 2, "The transpose tensor should be matrix and vector"
        if len(self._shape) == 1 or len(self._shape) - dim == 1:
            out = NDArray.make(self._shape + tuple(1), self._handle.dev_id, device=self.device, dtype=self._dtype)
            return out
        else:
            out = NDArray.make(self._shape[:-2] + tuple([self._shape[-1], self._shape[-2]]), self._handle.dev_id, device=self.device, dtype=float_type(self._dtype))
        
        self.device.matrix_T(self._handle, out._handle, pre_dim_a, self._shape[-1], post_dim_a, self._offset)
        return out

    def dot(self, other, dim_0 = 0, dim_1 = 0):
        
        if not isinstance(other, NDArray):
            other = NDArray(other)

        if dim_0 != 0:
            pre_dim_a, post_dim_a = prob(self._shape[:dim_0]), prob(self._shape[dim_0:])
        else:
            if len(self._shape) > 2:
                pre_dim_a, post_dim_a = prob(self._shape[:-2]), prob(self._shape[-2:])
            else:
                pre_dim_a, post_dim_a = 1, prob(self._shape)

        if dim_1 != 0:
            pre_dim_b, post_dim_b = prob(other._shape[:dim_0]), prob(other._shape[dim_0:])
        else:
            if len(other._shape) > 2:
                pre_dim_b, post_dim_b = prob(other._shape[:-2]), prob(other._shape[-2:])
            else:
                pre_dim_b, post_dim_b = 1, prob(other._shape)

        assert self._shape[-1] == other._shape[-2], "shapes {A} and shapes {B} not aligned: {DA}(dim -1) != {DB}(dim -2)".format(A=self._shape, B=other._shape, DA=self.shape[-1], DB=other.shape[-2])
        assert pre_dim_a == pre_dim_b, "Size of shapes {A} and shapes {B} not aligned: {DA} != {DB}".format(A=self._shape, B=other._shape, DA=pre_dim_a, DB=pre_dim_b)

        out = NDArray.make(self._shape[:-1] if len(self._shape) > 1 else tuple(1) + other._shape[-1:], self._handle.dev_id, device=self.device, dtype=float_type(_out_dtype(self._dtype, other._dtype)))
        
        self.device.matrix_dot(self._handle, other._handle, out._handle, pre_dim_a, self._shape[-1], other._shape[-1], post_dim_a, post_dim_b, self._offset, other._offset)
        return out

    #[ ] TODO: The matrix should be transpose first since the cublas is column-major
    def inv(self):
        assert len(self._shape) >= 2 and self._shape[-1] == self._shape[-2], "The matrix to be inversed must be square matrix"
        
        pre_dim_a, post_dim_a = prob(self._shape[:-2]), prob(self._shape[-2:])
        out = NDArray.make(self._shape, self._handle.dev_id, device=self.device, dtype=float_type(self._dtype))
        infos = NDArray.make(tuple(pre_dim_a), self._handle.dev_id, device=self.device, dtype=float_type(self._dtype))
        
        a_copy = self.copy()
        self.device.matrix_inv(a_copy._handle, out._handle, pre_dim_a, self._shape[-1], post_dim_a, self._offset, infos)
        #[ ]TODO: warning for the not succeed matrix
        return out
    
    #[ ]TODO: Support for the not square matrix
    def det(self):
        assert len(self._shape) >= 2 and self._shape[-1] == self._shape[-2], "The tensor to get det should be a matrix rather than a vector"
        
        pre_dim_a, post_dim_a = prob(self._shape[:-2]), prob(self._shape[-2:])
        out = NDArray.make(self._shape[:-2], self._handle.dev_id, device=self.device, dtype=float_type(self._dtype))
        infos = NDArray.make(tuple(pre_dim_a), self._handle.dev_id, device=self.device, dtype=float_type(self._dtype))

        a_copy = self.copy()
        self.device.matrix_det(a_copy._handle, out._handle, pre_dim_a, self._shape[-1], post_dim_a, self._offset, infos)
        return out
    
    def diagonal_sum(self):
        assert len(self._shape) >= 2 and self._shape[-1] == self._shape[-2], "The matrix to get the diagonal_sum must be square matrix"
        
        pre_dim_a, post_dim_a = prob(self._shape[:-2]), prob(self._shape[-2:])
        out = NDArray.make(self._shape[:-2], self._handle.dev_id, device=self.device, dtype=float_type(self._dtype))

        self.device.matrix_diagonal_sum(self._handle, out._handle, pre_dim_a, self._shape[-1], self._offset)
        return out

#[ ] TODO: Change to the boolean type(and NDArray should support for float32 int32 int64 type) 
def _all(array:NDArray):
    out = NDArray.make((1, ), array._handle.dev_id, device=array.device)
    post_dim = 1, prob(array._shape)
    array.device.ewise_sum(array, out, post_dim, array._offset)
    if out.numpy() == post_dim:
        return True
    else:
        return False
    
    #[ ] TODO: Change to the boolean type(and NDArray should support for float32 int32 int64 type) 
def _any(array:NDArray):
    out = NDArray.make((1, ), array._handle.dev_id, device=array.device)
    post_dim = 1, prob(array._shape)
    array.device.ewise_sum(array, out, post_dim, array._offset)
    if out.numpy() > 0:
        return True
    else:
        return False

def _where(bool_array, a_array, b_array):
    assert bool_array._shape == a_array._shape == b_array._shape, "The input arrays should keep the same shape: {BoA}/{AA}/{BA}".format(BoA=bool_array._shape, AA=a_array._shape, BA=b_array._shape)
    if not isinstance(bool_array, NDArray):
        bool_array = NDArray(bool_array)
    if not isinstance(a_array, NDArray):
        a_array = NDArray(a_array)
    if not isinstance(b_array, NDArray):
        b_array = NDArray(b_array)
    out = NDArray.make(bool_array._shape, bool_array._handle.dev_id)
    default_device().where(bool_array._handle, a_array._handle, b_array._handle, out._handle,
                           bool_array._offset, a_array._offset, b_array._offset)
    return out


def _zeros(shape, dtype, device_id=__default_devid__):
    out = NDArray.make(shape, device_id, dtype=dtype)
    out.device.ewise_assign(out._handle, 0, prob(shape), 0)
    return out

def _empty(shape, dtype, device_id=__default_devid__):
    out = NDArray.make(shape, device_id, dtype=dtype)
    return out

def _ones(shape, dtype, device_id=__default_devid__):
    out = NDArray.make(shape, device_id, dtype=dtype)
    out.device.ewise_assign(out._handle, 1, prob(shape), 0)
    return out

#[ ] TODO: Support auto type
def _full(shape, fill_value, dtype, device_id=__default_devid__):
    # assert isinstance(fill_value, int) or isinstance(fill_value, float), "full method only support built-in type"
    if dtype == None:
        if isinstance(fill_value, NDArray):
            out = NDArray.make(shape, device_id, dtype=fill_value.dtype)
        else:
            out = NDArray.make(shape, device_id, dtype=_supported_builtin_dtype(type(fill_value)))
    else:
        out = NDArray.make(shape, device_id, dtype=dtype)
    out.device.ewise_assign(out._handle, fill_value, prob(shape), 0)
    return out


def _uniform(low, high, shape, dtype, device_id=__default_devid__):
    raise NotImplementedError("it is not implemented now.")
    assert dtype in datatype_mapping, "The dtype '{D}' is not supported in current version".format(D=dtype)
    if dtype == None:
        out = NDArray.make(shape, device_id, dtype=_float64)
    else:
        out = NDArray.make(shape, device_id, dtype=dtype)
    if low < high: 
        out.device.ewise_uniform(out._handle, low, high, prob(shape))
    else:
        out.device.ewise_uniform(out._handle, high, low, prob(shape))
    return out

"""Image operations"""
def _filter(a_array, ROI, mask, channel, filter_kernel, anchor=None):
    #[ ] TODO: the datatype should be uint32 or uint64
    
    if not isinstance(a_array, NDArray):
        a_array = NDArray(a_array)
    if channel == 1:
        pre_dim, post_dim = prob(a_array.shape[:-2]), prob(a_array.shape[-2:])
        nstep = a_array.shape[-1]
    else:
        pre_dim, post_dim = prob(a_array.shape[:-3]), prob(a_array.shape[-3:])
        nstep = prob(a_array.shape[-2:])
    # print("shape: ", a_array.shape)
    # img_filter_shape = tuple((a_array.shape[-2] - k_shape[-2]) / stride + 1, (a_array.shape[-1] - k_shape[-1]) / stride + 1)
    # out = NDArray.make(shape=a_array.shape[:-2] + img_filter_shape, device=a_array.device)
    out = NDArray.make(shape=a_array.shape, device_id=a_array._handle.dev_id, device=a_array.device, dtype=a_array._dtype)
    if anchor is None:
        filter_kernel(a_array._handle, nstep, out._handle, nstep, ROI, mask, channel, pre_dim, post_dim)
    else:
        t = anchor < mask
        assert len(anchor) == len(mask) and len(anchor) <= 2 and all(t) if isinstance(t, Iterable) else t, "The anchor size should smaller than filter_kernel."
        filter_kernel(a_array._handle, nstep, out._handle, nstep, ROI, mask, anchor, channel, pre_dim, post_dim)
    

    return out

# def gaussian_filter(a_array, ROI, mask, channel):
#     return _filter(a_array, ROI, mask, channel, a_array.device.gaussian_filter)

def laplacian_filter(a_array, ROI, mask, channel):
    return _filter(a_array, ROI, mask, channel, a_array.device.laplacian_filter)

# def sobel_filter(a_array, ROI, horiz, channel):
#     return _filter(a_array, ROI, horiz, channel, a_array.device.sobel_filter)

# def box_filter(a_array, ROI, mask, anchor, channel):
#     return _filter(a_array, ROI, mask, channel, a_array.device.box_filter, anchor)
    
def median_filter(a_array, ROI, mask, anchor, channel):
    return _filter(a_array, ROI, mask, channel, a_array.device.median_filter, anchor)

def convolution(a_array, kernel, in_channel, padding=None, dilation=0, stride=1, constant=0):
    if padding is None:
        padding = (0, 0)
    elif isinstance(padding, int):
        padding = (0, padding)
    else:
        assert len(padding) == 2, "The padding should be a size-2 tuple"
    if not isinstance(a_array, NDArray):
        a_array = NDArray(a_array)
    if not isinstance(kernel, NDArray):
        kernel = NDArray(kernel)
    
    if in_channel > 1:
        assert in_channel == a_array.shape[-1], "The in_channel should keep same with input array dim, where in_channel='{A}' not equal to '{B}' in input array".format(A=in_channel, B=a_array.shape[-1])
        pre_dim, post_dim = prob(a_array.shape[:-3]), prob(a_array.shape[-3:])
        out = NDArray.make(shape=a_array.shape[:-3] + (int(padding[0] * 2 + a_array.shape[-3] - (kernel.shape[-2] * (dilation + 1) - 1)) / stride + 1, int(padding[1] * 2 + a_array.shape[-2] - (kernel.shape[-1] * (dilation + 1) - 1)) / stride + 1),
                            device_id=a_array._handle.dev_id, device=a_array.device, dtype=a_array.dtype)
        assert 0==1
    else:
        # print("a_array.shape: ", padding, kernel.shape, a_array.shape, (int((padding[0] * 2 + a_array.shape[-2] - (kernel.shape[-2] * (dilation + 1) - 1)) / stride + 1), int((padding[1] * 2 + a_array.shape[-1] - (kernel.shape[-1] * (dilation + 1) - 1)) / stride + 1)))
        pre_dim, post_dim = prob(a_array.shape[:-2]), prob(a_array.shape[-2:])
        out = NDArray.make(shape=a_array.shape[:-2] + (int((padding[0] * 2 + a_array.shape[-2] - (kernel.shape[-2] * (dilation + 1) - 1)) / stride + 1), int((padding[1] * 2 + a_array.shape[-1] - (kernel.shape[-1] * (dilation + 1) - 1)) / stride + 1)),
                            device_id=a_array._handle.dev_id, device=a_array.device, dtype=a_array.dtype)
        a_array.device.conv(a_array._handle, a_array.shape[-2], pre_dim, post_dim, out._handle, kernel._handle, kernel.shape[-2:], padding, stride, dilation, constant, a_array._offset, kernel._offset)

    return out

def box_filter(a_array, mask, in_channel, padding=None):
    if padding is None:
        padding = (0, 0)
    elif isinstance(padding, int):
        padding = (0, padding)
    else:
        assert len(padding) == 2, "The padding should be a size-2 tuple"
    assert len(mask) == 2, "The mask should be a size-2 tuple"
    kernel = _full(tuple(mask), 1. / (mask[0] * mask[1]), _float32)
    out = convolution(a_array, kernel, in_channel, padding)
    assert out.shape[-1] != 0, "{A}".format(A=out.shape)
    return out

def sobel_filter(a_array, in_channel, horiz=True, padding=None):
    if padding is None:
        padding = (0, 0)
    elif isinstance(padding, int):
        padding = (0, padding)
    else:
        assert len(padding) == 2, "The padding should be a size-2 tuple"
    if horiz:
        kernel = NDArray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], a_array.dtype)
    else:
        kernel = NDArray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], a_array.dtype)
    out = convolution(a_array, kernel, in_channel, padding)
    assert out.shape[-1] != 0, "{A}".format(A=out.shape)
    return out


def gaussian_filter(a_array, mask, in_channel, sigma=1., padding=None):
    if padding is None:
        padding = (0, 0)
    elif isinstance(padding, int):
        padding = (0, padding)
    else:
        assert len(padding) == 2, "The padding should be a size-2 tuple"
    assert len(mask) == 2, "The mask should be a size-2 tuple"
    kernel = NDArray(
        [[1./(2*math.pi*sigma) * math.exp(-((i - (mask[1] - 1) / 2)**2 + (j - (mask[0] - 1) / 2) ** 2) / (2 * sigma**2)) for i in range(mask[1])] for j in range(mask[0])],
        a_array.dtype
    )
    out = convolution(a_array, kernel, in_channel, padding)
    assert out.shape[-1] != 0, "{A}".format(A=out.shape)
    return out

def conv_2D(a_array, ROI, kernel, anchor, padding=None, ndivisor=1):
    stride = 1
    if padding is None:
        padding = (0, 0)
    else:
        assert len(padding) == 2, "The padding should be a size-2 tuple"
    if not isinstance(a_array, NDArray):
        a_array = NDArray(a_array)
    
    # if channel == 1:
    #     pre_dim = prob(a_array.shape[:-2])
    # else:
    #     pre_dim = prob(a_array.shape[:-3])
    # img_filter_shape = tuple((a_array.shape[-2] - k_shape[-2]) / stride + 1, (a_array.shape[-1] - k_shape[-1]) / stride + 1)
    # out = NDArray.make(shape=a_array.shape[:-2] + img_filter_shape, device=a_array.device)
    out = NDArray.make(shape=a_array.shape[:-3] + tuple((padding[1] * 2 + a_array.shape[-3] - kernel.shape[-2]) / stride + 1, (padding[0] * 2 + a_array.shape[-3] - kernel.shape[-1])) / stride + 1,
                        device_id=a_array._handle.dev_id, device=a_array.device, dtype=a_array.dtype)

    assert len(anchor) == 2 and all(anchor < kernel.shape[-2:]), "The anchor should be a size-2 tuple and the x-y posi should smaller than filter_kernel."
    a_array.device.convolution(a_array._handle, a_array.shape[-2], out._handle, out.shape[-2], ROI, kernel._handle, kernel.shape[-2:], anchor, padding, ndivisor, a_array.shape[-1])

    return out


if __name__ == "__main__":
    pass