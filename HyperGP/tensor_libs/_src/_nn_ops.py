from ...src.ndarray import *
from .basic import TensorOp, Channel, Mask



# class EwiseFilterGauss(TensorOp):
#     def compute(self):
#         raise NotImplementedError("backp has not been implemented yet")
#     def __call__(self, a, mask, channel, ROI=None):
#         Mask.is_compliant(mask)
#         Channel.is_compliant(channel)
#         assert channel.channel == a.shape[-1] or channel.channel==1, "input shape is not assistant with the channels"
#         if ROI is None:
#             ROI = (a.shape[-2], a.shape[-1]) if channel.channel==1 else (a.shape[-3], a.shape[-2])
#         return Tensor(gaussian_filter(a.realize_cached_data, ROI, mask.mask_id, channel.channel))
    
class EwiseFilterGauss(TensorOp):
    def compute(self, a, mask, in_channel, sigma=1., padding=None):
        Channel.is_compliant(in_channel)
        assert in_channel.channel == a.shape[-1] or in_channel.channel==1, "input shape is not assistant with the channels"
        return gaussian_filter(a, mask, in_channel.channel, sigma, padding)
    

class EwiseFilterLaplacian(TensorOp):
    def compute(self, a, mask, channel, ROI=None):
        Mask.is_compliant(mask)
        Channel.is_compliant(channel)
        assert channel.channel == a.shape[-1] or channel.channel==1, "input shape is not assistant with the channels"
        if ROI is None:
            ROI = (a.shape[-2], a.shape[-1]) if channel.channel==1 else (a.shape[-3], a.shape[-2])
        return laplacian_filter(a, ROI, mask.mask_id, channel.channel)
        
# class EwiseFilterSobel(TensorOp):
#     def compute(self):
#         raise NotImplementedError("backp has not been implemented yet")
#     def __call__(self, a, channel, horiz=True, ROI=None):
#         Channel.is_compliant(channel)
#         assert channel.channel == a.shape[-1] or channel.channel==1, "input shape is not assistant with the channels"
#         if ROI is None:
#             ROI = (a.shape[-2], a.shape[-1]) if channel.channel==1 else (a.shape[-3], a.shape[-2])
#         return Tensor(sobel_filter(a.realize_cached_data, ROI, horiz, channel.channel))
    
class EwiseFilterSobel(TensorOp):
    def compute(self, a, in_channel, horiz=True, padding=None):
        Channel.is_compliant(in_channel)
        assert in_channel.channel == a.shape[-1] or in_channel.channel==1, "input shape is not assistant with the channels"
        return sobel_filter(a, in_channel.channel, horiz, padding)
    
# class EwiseFilterMean(TensorOp):
#     def compute(self):
#         raise NotImplementedError("backp has not been implemented yet")
#     def __call__(self, a, mask, channel, ROI=None, anchor=None):
#         Channel.is_compliant(channel)
#         assert channel.channel == a.shape[-1] or channel.channel==1, "input shape is not assistant with the channels"
#         if ROI is None:
#             ROI = (a.shape[-2], a.shape[-1]) if channel.channel==1 else (a.shape[-3], a.shape[-2])
#         Mask.is_compliant(mask)
#         if anchor is None:
#             anchor = Mask.anchor(mask.mask_id)
#         return Tensor(box_filter(a.realize_cached_data, ROI, Mask.mask(mask.mask_id), anchor, channel.channel))


class EwiseFilterMean(TensorOp):
    def compute(self, a, mask, in_channel, padding=None):
        Channel.is_compliant(in_channel)
        assert in_channel.channel == a.shape[-1] or in_channel.channel==1, "input shape is not assistant with the channels"
        return box_filter(a, mask, in_channel.channel, padding)
    
class EwiseFilterMedian(TensorOp):
    def compute(self, a, mask, channel, ROI=None, anchor=None):
        Mask.is_compliant(mask)
        Channel.is_compliant(channel)
        assert channel.channel == a.shape[-1] or channel.channel==1, "input shape is not assistant with the channels"
        if ROI is None:
            ROI = (a.shape[-2], a.shape[-1]) if channel.channel==1 else (a.shape[-3], a.shape[-2])
        if anchor is None:
            anchor = Mask.anchor(mask.mask_id)
        return median_filter(a, ROI, Mask.mask(mask.mask_id), anchor, channel.channel)
    
class EwiseFilterCov2D(TensorOp):
    def compute(self, a, kernel, anchor=None, padding=None, ndivisor=1, ROI=None):
        if anchor is None:
            anchor = (kernel.shape[0] / 2, kernel.shape[1] / 2)
        return conv_2D(a, ROI, kernel, anchor, padding, ndivisor)


