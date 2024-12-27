import numpy as np

class dtype:
    def _startwith(self, strs):
        return self.__name__.startwith(strs)

"""support datatype"""
class _int64(dtype):
    _type=3
    pass
class _int32(dtype):
    _type=2
    pass
class _int16(dtype):
    _type=0
    pass
class _int8(dtype):
    _type=0
    pass
class _uint8(dtype):
    _type=0
    pass
class _uint16(dtype):
    _type=1
    pass
class _uint32(dtype):
    _type=2
    pass
class _uint64(dtype):
    _type=3
    pass
class _float32(dtype):
    _type=4
    pass
class _float64(dtype):
    _type=5
    pass
class _bool(dtype):
    _type=0
    pass
"""================"""
datatype_mapping = {
    _bool:np.bool_,
    _int8:np.int8,
    _int16:np.int16,
    _int32:np.int32,
    _int64:np.int64,
    _uint8:np.uint8,
    _uint16:np.uint16,
    _uint32:np.uint32,
    _uint64:np.uint64,
    _float32:np.float32,
    _float64:np.float64,
}

_dtype_strmap = {
    "_bool":_bool,
    "_int8":_int8,
    "_int16":_int16,
    "_int32":_int32,
    "_int64":_int64,
    "_uint8":_uint8,
    "_uint16":_uint16,
    "_uint32":_uint32,
    "_uint64":_uint64,
    "_float32":_float32,
    "_float64":_float64,
}

def _supported_dtype(dtype):
    if isinstance(dtype, np.dtype):
        if dtype==np.float32:
            return _float32
        elif dtype==np.float64:
            return _float64
        elif dtype==np.int64:
            return _int64
        elif dtype==np.int32:
            return _int32
        elif dtype==np.int16:
            return _int16
        elif dtype==np.int8:
            return _int8
        elif dtype==np.uint8:
            return _uint8
        elif dtype==np.uint16:
            return _uint16
        elif dtype==np.uint32:
            return _uint32
        elif dtype==np.uint64:
            return _uint64
        raise NotImplementedError("The datatype {DT} is not supported in the current version".format(DT=dtype))
    return dtype

def _out_dtype(dtype_1, dtype_2):
    dtype_name_1, dtype_name_2 = dtype_1.__name__, dtype_2.__name__
    if dtype_1 == dtype_2:
        return dtype_1
    if dtype_1.__name__[:2] == dtype_2.__name__[:2]:
        size = max(_sizeof(dtype_1), _sizeof(dtype_2))
        if size == 8 and dtype_name_1.startswith('_f'):
            return _float64
        if size == 4 and dtype_name_1.startswith('_f'):
            return _float32
        if size == 8 and dtype_name_1.startswith('_i'):
            return _int64
        if size == 4 and dtype_name_1.startswith('_i'):
            return _int32
        if size == 2 and dtype_name_1.startswith('_i'):
            return _int16
        if size == 8 and dtype_name_1.startswith('_u'):
            return _uint64
        if size == 4 and dtype_name_1.startswith('_u'):
            return _uint32
        if size == 2 and dtype_name_1.startswith('_u'):
            return _uint16
    else:
        size = max(_sizeof(dtype_1), _sizeof(dtype_2))
        if size == 8 and (dtype_name_1.startswith('_f') or dtype_name_2.startswith('_f')):
            return _float64
        if size == 8 and not (dtype_name_1.startswith('_f') or dtype_name_2.startswith('_f')):
            return _int64
        if size == 4 and (dtype_name_1.startswith('_f') or dtype_name_2.startswith('_f')):
            return _float64
        if size == 4 and not (dtype_name_1.startswith('_f') or dtype_name_2.startswith('_f')):
            return _int64
        if size == 2:
            return _int32
        if size == 1:
            return _int16

    raise NotImplementedError("The datatype {DT} is not supported in the current version".format(DT=dtype))

def _supported_builtin_dtype(dtype):
        if dtype==int:
            return _int64
        elif dtype==float:
            return _float64
        elif dtype==bool:
            return _bool
        else:
            raise ValueError("only support built-in type")
        raise NotImplementedError("The datatype {DT} is not supported in the current version".format(DT=dtype))


def _sizeof(dtype):
    if dtype == _bool or dtype == _uint8 or dtype == _int8:
        return 1
    elif dtype == _uint16 or dtype == _int16:
        return 2
    elif dtype == _uint32 or dtype == _int32 or dtype == _float32:
        return 4
    elif dtype == _uint64 or dtype == _int64 or dtype == _float64:
        return 8
    raise NotImplementedError("Unknown dtype: {A}".format(A=dtype))

def implicit_conversion(dtype_1, dtype_2):
    assert dtype_1._type != dtype_2._type or dtype_1 == dtype_2, "The implicit datatype conversion fail.."
    return dtype_2 if dtype_1._type < dtype_2._type else dtype_1

def float_type(dtype):
    # print(type(dtype), dtype, dtype.__dict__)
    if dtype.__name__.startswith('_f'):
        return dtype
    elif _sizeof(dtype) < 4:
        return _float32
    else:
        return _float64