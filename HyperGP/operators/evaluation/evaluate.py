from ... import Tensor, tensor

class EvaluateMethod:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("No find '__call__' implement")

def rmse(output: Tensor, target: Tensor):
    return tensor.sqrt(tensor.sum((tensor.sub(output, target, dim_0=1)) ** 2, dim=1) / target.shape[0])

def mse(output: Tensor, target: Tensor):
    return (tensor.sub(output, target, dim_0=1) ** 2).sum(dim=1) / target.shape[0]

def mae(output: Tensor, target: Tensor):
    return (tensor.sub(output, target, dim_0=1) ** 2).sum(dim=1) / target.shape[0]

'''In GPU method, we use the proposed GPU-CPU mapping func?'''

def r2_s(output: Tensor, target: Tensor):
    avg_label = tensor.mean(target)
    sse = tensor.sum(tensor.sub(output, target) ** 2)
    sst = tensor.sum(tensor.sub(target, avg_label, dim=1) ** 2)
    return 1 - sse / sst

def r2(output: Tensor, target: Tensor):
    # avg_label = tensor.mean(output, dim=1)
    avg_label = tensor.mean(target)
    sse = tensor.sum(tensor.sub(output, target, dim=1) ** 2, dim=1)
    sst = tensor.sum(tensor.sub(target, avg_label, dim=1) ** 2)
    return tensor.sub(1, tensor.div(sse, sst, dim_0=1), dim_1=1)
