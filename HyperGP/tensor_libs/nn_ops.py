from ._src.basic import TensorOp, Channel, Mask, MOD
from ._src._nn_ops import *
from .tensor_basic import Tensor


from ._src.basic import Channel as CHANNEL
from ._src.basic import Mask as MASK


def gauss_filter(x: Tensor, mask, in_channel: CHANNEL, sigma=1., padding=None):
    if not isinstance(x, Tensor):
        x = Tensor(x)
    if MOD=="IMM":
        Channel.is_compliant(in_channel)
        assert in_channel.channel == Tensor(x).shape[-1] or in_channel.channel==1, "input shape is not assistant with the channels"
        return Tensor(gaussian_filter(x.realize_cached_data, mask, in_channel.channel, sigma, padding))
    else:
        tensor = Tensor.make_from_op(EwiseFilterGauss(), [x, mask, in_channel, sigma, padding])
        if MOD == "Async":
            tensor.realize_cached_data
    return tensor


def lap_filter(x: Tensor, mask, channel: CHANNEL, ROI=None):
    if not isinstance(x, Tensor):
        x = Tensor(x)
    if MOD=="IMM":
        Mask.is_compliant(mask)
        Channel.is_compliant(channel)
        assert channel.channel == x.shape[-1] or channel.channel==1, "input shape is not assistant with the channels"
        if ROI is None:
            ROI = (x.shape[-2], x.shape[-1]) if channel.channel==1 else (x.shape[-3], x.shape[-2])
        return Tensor(laplacian_filter(x.realize_cached_data, ROI, mask.mask_id, channel.channel))
    else:
        tensor = Tensor.make_from_op(EwiseFilterLaplacian(), [x, mask, channel, ROI])
        if MOD == "Async":
            tensor.realize_cached_data
    return tensor

def sobel_filter(x: Tensor, in_channel: CHANNEL, horiz=True, padding=None):
    if not isinstance(x, Tensor):
        x = Tensor(x)
    if MOD=="IMM":
        Channel.is_compliant(in_channel)
        assert in_channel.channel == x.shape[-1] or in_channel.channel==1, "input shape is not assistant with the channels"
        return Tensor(sobel_filter(x.realize_cached_data, in_channel.channel, horiz, padding))
    else:
        tensor = Tensor.make_from_op(EwiseFilterSobel(), [x, in_channel, horiz, padding])
        if MOD == "Async":
            tensor.realize_cached_data
    return tensor
  
def mean_filter(x: Tensor, mask, in_channel: CHANNEL, padding=None):
    if not isinstance(x, Tensor):
        x = Tensor(x)
    if MOD=="IMM":
        Channel.is_compliant(in_channel)
        assert in_channel.channel == x.shape[-1] or in_channel.channel==1, "input shape is not assistant with the channels"
        return Tensor(box_filter(x.realize_cached_data, mask, in_channel.channel, padding))
    else:
        tensor = Tensor.make_from_op(EwiseFilterMean(), [x, mask, in_channel, padding])
        if MOD == "Async":
            tensor.realize_cached_data
    return tensor

def median_filter(x: Tensor, mask, channel: CHANNEL, ROI=None, anchor=None):
    if not isinstance(x, Tensor):
        x = Tensor(x)
    if MOD=="IMM":
        Mask.is_compliant(mask)
        Channel.is_compliant(channel)
        assert channel.channel == x.shape[-1] or channel.channel==1, "input shape is not assistant with the channels"
        if ROI is None:
            ROI = (x.shape[-2], x.shape[-1]) if channel.channel==1 else (x.shape[-3], x.shape[-2])
        if anchor is None:
            anchor = Mask.anchor(mask.mask_id)
        return Tensor(median_filter(x.realize_cached_data, ROI, Mask.mask(mask.mask_id), anchor, channel.channel))
    else:
        tensor = Tensor.make_from_op(EwiseFilterMedian(), [x, mask, channel, ROI, anchor])
        if MOD == "Async":
            tensor.realize_cached_data
    return tensor
    
def conv_2D(x: Tensor, kernel: Tensor, anchor=None, padding=None, ndivisor=1, ROI=None):
    if not isinstance(x, Tensor):
        x = Tensor(x)
    if MOD=="IMM":
        if anchor is None:
            anchor = (kernel.shape[0] / 2, kernel.shape[1] / 2)
        return Tensor(conv_2D(x.realize_cached_data, ROI, kernel, anchor, padding, ndivisor))
    else:
        tensor = Tensor.make_from_op(EwiseFilterCov2D(), [x, kernel, anchor, padding, ndivisor, ROI])
        if MOD == "Async":
            tensor.realize_cached_data
    return tensor