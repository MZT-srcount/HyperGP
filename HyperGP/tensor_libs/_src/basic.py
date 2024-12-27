
from typing import Optional, List
from ...src.tensor_backend_selection import array_api, NDArray


MOD = "Async"

class _mask(int):
    def __init__(self, id):
        self.mask_id = id

"Mask Selects"
class Mask:
    NPP_MASK_SIZE_1_X_3 = _mask(0)
    NPP_MASK_SIZE_1_X_5 = _mask(1)
    NPP_MASK_SIZE_3_X_1 = _mask(2)
    NPP_MASK_SIZE_5_X_1 = _mask(3)
    NPP_MASK_SIZE_5_X_5 = _mask(4)
    NPP_MASK_SIZE_7_X_7 = _mask(5)
    NPP_MASK_SIZE_9_X_9 = _mask(6)
    NPP_MASK_SIZE_11_X_11 = _mask(7)
    NPP_MASK_SIZE_13_X_13 = _mask(8)
    NPP_MASK_SIZE_15_X_15 = _mask(9)

    def __getattr__(self, item):
        return item
    
    @staticmethod
    def anchor(mask_id):
        if mask_id == 0:
            return (0, 1)
        if mask_id == 1:
            return (0, 2)
        if mask_id == 2:
            return (1, 0)
        if mask_id == 3:
            return (2, 0)
        if mask_id == 4:
            return (2, 2)
        if mask_id == 5:
            return (3, 3)
        if mask_id == 6:
            return (4, 4)
        if mask_id == 7:
            return (5, 5)
        if mask_id == 8:
            return (6, 6)
        if mask_id == 9:
            return (7, 7)

    @staticmethod
    def mask(mask_id):
        if mask_id == 0:
            return (1, 3)
        if mask_id == 1:
            return (1, 5)
        if mask_id == 2:
            return (3, 1)
        if mask_id == 3:
            return (5, 1)
        if mask_id == 4:
            return (5, 5)
        if mask_id == 5:
            return (7, 7)
        if mask_id == 6:
            return (9, 9)
        if mask_id == 7:
            return (11, 11)
        if mask_id == 8:
            return (13, 13)
        if mask_id == 9:
            return (15, 15)
        
    @staticmethod
    def is_compliant(mask):
        assert isinstance(mask, _mask), "The type should be Mask, where: {TYPE}".format(TYPE=type(mask))

class _channel(int):
    def __init__(self, id):
        self.channel = id
"Channel Selects"
class Channel:
    C1R=_channel(1)
    C3R=_channel(3)
    C4R=_channel(4)

    def __getattr__(self, item):
        return item
    @staticmethod
    def is_compliant(channel):
        assert isinstance(channel, _channel), "The type should be Channel, where: {TYPE}".format(TYPE=type(channel))


#[ ]TODO: copy function to be implemented
class Op:
    def __call__(self):
        raise NotImplementedError

    def compute(self, *args):
        raise NotImplementedError
    
class Value:
    op: Optional[Op]
    input: List["Value"]
    cached_data: NDArray

    def _init(self, op: Optional[Op], inputs, cached_data=None):
        
        self.cached_data = cached_data
        self.op = op
        self.input = inputs

class TensorOp(Op):
    pass
    # def make(self, *args):
    #     tensor = Tensor.make_from_op(self, args)
    #     return tensor