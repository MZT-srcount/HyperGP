from ...src.ops_dim import substract, add, multiply, divide, concatenate as concat
from ...src.ndarray import _where, _all, _any, _zeros, _ones, _full, _empty, _uniform
from ...src.ndarray import *
from ...src import float64
from .basic import TensorOp



class EWiseTDim(TensorOp):
    op = 'T'
    def compute(self, a, dim_0=0):
        return a.T(dim_0)

class EWiseInvDim(TensorOp):
    op = 'inv'
    def compute(self, a):
        return a.inv()

class EWiseDetDim(TensorOp):
    op = 'det'
    def compute(self, a):
        return a.det()

class EWiseDiagonalSum(TensorOp):
    op = 'det'
    def compute(self, a):
        return a.diagonal_sum()



class EWiseDotDim(TensorOp):
    op = 'dot'
    def compute(self, a, b, dim_0=0, dim_1=0):
        return a.dot(b, dim_0, dim_1)

class EWiseAdd(TensorOp):
    op = 'add'
    idx = 0
    def compute(self, a, b, dim_0=0, dim_1=0):
        if dim_0 == 0 and dim_1 == 0:
            return a + b
        else:
            return add(a, b, dim_0, dim_1)

class EWiseMul(TensorOp):
    op = 'mul'
    idx = 2
    def compute(self, a, b, dim_0=0, dim_1=0):
        if dim_0 == 0 and dim_1 == 0:
            return a * b
        else:
            return multiply(a, b, dim_0, dim_1)

class EWiseSub(TensorOp):
    op = 'sub'
    idx = 1
    def compute(self, a, b, dim_0=0, dim_1=0):
        if dim_0 == 0 and dim_1 == 0:
            return a - b
        else:
            return substract(a, b, dim_0, dim_1)

class EWiseDiv(TensorOp):
    op = 'div'
    idx = 3
    def compute(self, a, b, dim_0=0, dim_1=0):
        if dim_0 == 0 and dim_1 == 0:
            return a / b
        else:
            return divide(a, b, dim_0, dim_1)

class EWiseAssign_R2L(TensorOp):
    op = 'assigin_r2l'
    idx = 11
    def compute(self, a, b):
        a[:] = b
        return a
  
class EWisePass(TensorOp):
    op = 'assigin_1'
    idx = 11
    def compute(self, a):
        return a
          
class EWiseLt(TensorOp):
    def compute(self, a, b):
        return a < b
        
class EWiseLe(TensorOp):
    def compute(self, a, b):
        return a <= b
        
class EWiseGt(TensorOp):
    def compute(self, a, b):
        return a > b
        
class EWiseGe(TensorOp):
    def compute(self, a, b):
        return a >= b
    
class EWiseEq(TensorOp):
    def compute(self, a, b):
        return a == b
    
class EWiseNeq(TensorOp):
    def compute(self, a, b):
        return a != b

class EWiseWhere(TensorOp):
    def compute(self, bool_array, true_array, false_array):
        return _where(bool_array, true_array, false_array)

class EWiseAny(TensorOp):
    def compute(self, a):
        return _any(a)
    
class EWiseAll(TensorOp):
    def compute(self, a):
        return _all(a)
    
class EWiseSin(TensorOp):
    idx = 4
    def compute(self, b):
        return b.sin()

class EWiseCos(TensorOp):
    idx = 5
    def compute(self, a):
        return a.cos()

class EWiseTan(TensorOp):
    idx = 6
    def compute(self, a):
        return a.tan()

class EWiseSqrt(TensorOp):
    idx = 7
    def compute(self, a):
        return a.sqrt()

class EWiseLoge(TensorOp):
    idx = 7
    def compute(self, a):
        return a.loge()


class EWiseLog10(TensorOp):
    def compute(self, a):
        return a.log10()


class EWiseLog2(TensorOp):
    def compute(self, a):
        return a.log2()

class EWiseArcSin(TensorOp):
    idx = 8
    def compute(self, a):
        return a.arcsin()


class EWiseArcCos(TensorOp):
    idx = 9
    def compute(self, a):
        return a.arccos()
    

class EWiseArcTan(TensorOp):
    idx = 10
    def compute(self, a):
        return a.arctan()

class EWiseSign(TensorOp):
    def compute(self, a):
        return a.sign()

class EWiseExp(TensorOp):
    idx = 10
    def compute(self, a):
        return a.exp()
    
class EWiseAbs(TensorOp):
    def compute(self, a):
        return a.abs()

class EWiseNeg(TensorOp):
    def compute(self, a):
        return a.__neg__()
    
class EWiseCeil(TensorOp):
    def compute(self, a):
        return a.ceil()

class EWiseFloor(TensorOp):
    def compute(self, a):
        return a.floor()


class EWiseSum(TensorOp):
    def compute(self, a, dim=0):
        return a.sum(dim)
    
class EWiseMin(TensorOp):
    def compute(self, a, dim=0):
        return a.min(dim)
    
class EWiseMax(TensorOp):
    def compute(self, a, dim=0):
        return a.max(dim)
    
class EWiseMean(TensorOp):
    def compute(self, a, dim=0):
        return a.mean(dim)
    
class EWiseArgmax(TensorOp):
    def compute(self, a, dim=0):
        return a.argmax(dim)
    
class EWiseArgmin(TensorOp):
    def compute(self, a, dim=0):
        return a.argmin(dim)
    
class EWiseStd(TensorOp):
    def compute(self, a, dim=0):
        return a.std(dim)
    
class EWiseVar(TensorOp):
    def compute(self, a, dim=0):
        return a.var(dim)
    
class EwiseSubDim(TensorOp):
    def compute(self, a, b, dim):
        return a.substract(b, dim)

"""
init:
arange
zeros
ones
full
eye?

vector ops:
abs
neg
pow
exp
ceil、floor
sign
min Dim
max Dim
mean Dim
argmax Dim
argmin Dim
std Dim
var Dim
cumsum Dim
cumprob Dim
sort?
unique?
shuffle?
mod?
choice?

clamp: limit the element a range.

judgement:
where
<
>
==
any
all

matrix ops:
.dot
.T
.trace
.linalg.det行列式
.linalg.inv逆矩阵
.linalg.solve解矩阵
.rref化简矩阵

image ops:
filter(gaussian, laplacian, sobel)
feature extraction(SIFT, uLBP, HOG, Hist, DIF)


"""


class ScalarAdd(TensorOp):
    def compute(self, a, b):
        return a + b
    

class ScalarMul(TensorOp):
    op = 'mul'
    def compute(self, a, b):
        return a * b

class ScalarSub(TensorOp):
    op = 'sub'
    def compute(self, a, b):
        return a - b

class ScalarDiv(TensorOp):
    op = 'div'
    def compute(self, a, b):
        return a / b




if __name__ == "__main__":
    print("============================TEST:ops.py============================")
    test_l = [[1, 2], [3, 4]]
    test_l.reverse()
    print(test_l)