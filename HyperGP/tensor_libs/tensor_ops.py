from ..src.ops_dim import substract, add, multiply, divide, concatenate as concat
from ..src.ndarray import _where, _all, _any, _zeros, _ones, _full, _empty, _uniform
from ..src.ops_dim import add as _add
from ..src import float64
from ._src._tensor_ops import *
from ._src.basic import MOD

from .tensor_basic import Tensor

def add(x: Tensor, y: Tensor, dim_0: int=0, dim_1: int=0) -> Tensor:
	"""Elementwise addition: :math:`x + y`.

	Args:
		x, y(Tensor or array_like) : The arrays to be added.
			   If x1.shape != x2.shape and x1.shape != 0 and x2.shape != 0, then dim_0 or dim_1 should be set to do broadcast operation. 
		dim_0: The dim of x to do broadcast.
		 		x.shape[dim_0:] should be equal to y.shape[dim_1:]
		dim_1: The dim of y to do broadcast.
		 		x.shape[dim_0:] should be equal to y.shape[dim_1:]
	
	Returns:
		a new 'Tensor' is returned
	
	Examples:

		import modules

		>>> import numpy as np
		>>> from HyperGP import Tensor
		>>> import time

		array initialization

		>>> x1 = np.random.uniform(-1, 1, size=(500, 100000))
		>>> x2 = np.random.uniform(-1, 1, size=(500, 100000))
		>>> x1_t, x2_t = Tensor(x1), Tensor(x2)

		runtime test
		
		>>> st = time.time()
		>>> ar = [x1 + x2 for i in range(10)]
		>>> print("numpy runtime: ", time.time() - st)
		numpy runtime:  0.17456567287445068

		>>> st = time.time()
		>>> ar = [x1_t + x2_t for i in range(10)]
		>>> print("numpy runtime: ", time.time() - st)
		HyperGP runtime:  0.00162813663482666

		broadcast operation

		>>> ar = [x1 + x2 for i in range(10)]
		>>> ar = [HyperGP.add(x1_t, x2_t, dim_0=1, dim_1=1) for i in range(10)]
		>>> for x in ar: 
		... 	x.wait()
		numpy runtime:  0.17173876762390136
		HyperGP runtime:  0.001430368423461914
		



	Note:
		xxxxxxx

	"""
	if not isinstance(x, Tensor):
		x = Tensor(x)
	if not isinstance(y, Tensor):
		y = Tensor(y)
	if MOD == "IMM":
		return Tensor(_add(x.cached_data, y.cached_data, dim_0, dim_1))
	else:
		tensor = Tensor.make_from_op(EWiseAdd(), [x, y, dim_0, dim_1])
		if MOD == "Async":
			tensor.realize_cached_data
	return tensor


def sub(x: Tensor, y: Tensor, dim_0=0, dim_1=0):
	"""Elementwise subtraction: :math:`x - y`.

	Args:
		x, y(Tensor or array_like) : The arrays to perform subtraction.
			   If x1.shape != x2.shape and x1.shape != 0 and x2.shape != 0, then dim_0 or dim_1 should be set to do broadcast operation. 
		dim_0: The dim of x to do broadcast.
		 		x.shape[dim_0:] should be equal to y.shape[dim_1:]
		dim_1: The dim of y to do broadcast.
		 		x.shape[dim_0:] should be equal to y.shape[dim_1:]
	
	Returns:
		a new 'Tensor' is returned
	
	Examples:
		xxxxx

		>>> xxxx
		xxxx
	
	Note:
		xxxxxxx

	"""
	if not isinstance(x, Tensor):
		x = Tensor(x)
	if not isinstance(y, Tensor):
		y = Tensor(y)
	if MOD == "IMM":
		return Tensor(substract(x.cached_data, y.cached_data, dim_0, dim_1))
	else:
		tensor = Tensor.make_from_op(EWiseSub(), [x, y, dim_0, dim_1])
		if MOD == "Async":
			tensor.realize_cached_data
	return tensor


def mul(x: Tensor, y: Tensor, dim_0=0, dim_1=0):
	"""Elementwise multiply: :math:`x * y`.

	Args:
		x, y(Tensor or array_like) : The arrays to perform multiplication.
			   If x1.shape != x2.shape and x1.shape != 0 and x2.shape != 0, then dim_0 or dim_1 should be set to do broadcast operation. 
		dim_0: The dim of x to do broadcast.
		 		x.shape[dim_0:] should be equal to y.shape[dim_1:]
		dim_1: The dim of y to do broadcast.
		 		x.shape[dim_0:] should be equal to y.shape[dim_1:]
	
	Returns:
		a new 'Tensor' is returned
	
	Examples:
		xxxxx

		>>> xxxx
		xxxx
	
	Note:
		xxxxxxx

	"""
	if not isinstance(x, Tensor):
		x = Tensor(x)
	if not isinstance(y, Tensor):
		y = Tensor(y)
	if MOD == "IMM":
		return Tensor(multiply(x.cached_data, y.cached_data, dim_0, dim_1))
	else:
		tensor = Tensor.make_from_op(EWiseMul(), [x, y, dim_0, dim_1])
		if MOD == "Async":
			tensor.realize_cached_data
	return tensor

def div(x: Tensor, y: Tensor, dim_0=0, dim_1=0):
	"""Elementwise division: :math:`x / y`.

	Args:
		x, y(Tensor or array_like) : The arrays to perform division.
			   If x1.shape != x2.shape and x1.shape != 0 and x2.shape != 0, then dim_0 or dim_1 should be set to do broadcast operation. 
		dim_0: The dim of x to do broadcast.
		 		x.shape[dim_0:] should be equal to y.shape[dim_1:]
		dim_1: The dim of y to do broadcast.
		 		x.shape[dim_0:] should be equal to y.shape[dim_1:]
	
	Returns:
		a new 'Tensor' is returned
	
	Examples:
		xxxxx

		>>> xxxx
		xxxx
	
	Note:
		xxxxxxx

	"""
	if not isinstance(x, Tensor):
		x = Tensor(x)
	if not isinstance(y, Tensor):
		y = Tensor(y)
	if MOD == "IMM":
		return Tensor(divide(x.cached_data, y.cached_data, dim_0, dim_1))
	else:
		tensor = Tensor.make_from_op(EWiseDiv(), [x, y, dim_0, dim_1])
		if MOD == "Async":
			tensor.realize_cached_data
	return tensor



def pow(x: Tensor, y: Tensor, dim_0=0, dim_1=0):
	raise NotImplementedError("Not Implement in current version.")
	if MOD == "IMM":
		return Tensor(divide(Tensor(x).cached_data, y.cached_data, dim_0, dim_1))
	else:
		tensor = Tensor.make_from_op(EWiseDiv(), [Tensor(x), y, dim_0, dim_1])
		if MOD == "Async":
			tensor.realize_cached_data
	return tensor


def sum(x: Tensor, dim=0):
	"""Sum of array elements along with the corresponding dim.

	Args:
		x(Tensor or array_like) : Elements to sum.
		dim: The dim of x which a 'sum' is performed.
	
	Returns:
		a new 'Tensor' is returned
	
	Examples:
		xxxxx

		>>> xxxx
		xxxx
	
	Note:
		xxxxxxx

	"""
	if not isinstance(x, Tensor):
		x = Tensor(x)
	if MOD == "IMM":
		tensor = Tensor(x.cached_data.sum(dim))
	else:
		tensor = Tensor.make_from_op(EWiseSum(), [x, dim])
		if MOD == "Async":
			tensor.realize_cached_data
	return tensor

def min(x: Tensor, dim=0):
	"""Min value of the array elements along with the corresponding dim.

	Args:
		x(Tensor or array_like) : Elements to search for min values.
		dim: The dim of x which a 'min' is performed.
	
	Returns:
		a new 'Tensor' is returned
	
	Examples:
		xxxxx

		>>> xxxx
		xxxx
	
	Note:
		xxxxxxx

	"""
	if not isinstance(x, Tensor):
		x = Tensor(x)
	if MOD == "IMM":
		tensor = Tensor(x.cached_data.min(dim))
	else:
		tensor = Tensor.make_from_op(EWiseMin(), [x, dim])
		if MOD == "Async":
			tensor.realize_cached_data
	return tensor

def max(x: Tensor, dim=0):
	"""Max value of the array elements along with the corresponding dim.

	Args:
		x(Tensor or array_like) : Elements to search for maximum values.
		dim: The dim of x which a 'max' is performed.
	
	Returns:
		a new 'Tensor' is returned
	
	Examples:
		xxxxx

		>>> xxxx
		xxxx
	
	Note:
		xxxxxxx

	"""
	if not isinstance(x, Tensor):
		x = Tensor(x)
	if MOD == "IMM":
		tensor = Tensor(x.cached_data.max(dim))
	else:
		tensor = Tensor.make_from_op(EWiseMax(), [x, dim])
		if MOD == "Async":
			tensor.realize_cached_data
	return tensor

def argmin(x: Tensor, dim=0):
	"""Return the indices of the minimum of an array along the dim.

	Args:
		x(Tensor or array_like) : Elements to search for minimum values.
		dim: The dim of x which a 'min' is performed.
	
	Returns:
		ret(Tensor): the indices of the minimum of an array along the dim.
	
	Examples:
		xxxxx

		>>> xxxx
		xxxx
	
	Note:
		xxxxxxx

	"""
		
	if not isinstance(x, Tensor):
		x = Tensor(x)
	if MOD == "IMM":
		tensor = Tensor(x.cached_data.argmin(dim))
	else:
		tensor = Tensor.make_from_op(EWiseArgmin(), [x, dim])
		if MOD == "Async":
			tensor.realize_cached_data
	return tensor

def argmax(x: Tensor, dim=0):
	"""Return the indices of the maximum of an array along the dim.

	Args:
		x(Tensor or array_like) : Elements to search for maximum values.
		dim: The dim of x which a 'max' is performed.
	
	Returns:
		ret(Tensor): the indices of the maximum of an array along the dim.
	
	Examples:
		xxxxx

		>>> xxxx
		xxxx
	
	Note:
		xxxxxxx

	"""
	if not isinstance(x, Tensor):
		x = Tensor(x)
	if MOD == "IMM":
		tensor = Tensor(x.cached_data.argmax(dim))
	else:
		tensor = Tensor.make_from_op(EWiseArgmax(), [x, dim])
		if MOD == "Async":
			tensor.realize_cached_data
	return tensor

def mean(x: Tensor, dim=0):
	"""Mean value of the array elements along with the corresponding dim.

	Args:
		x(Tensor or array_like) : Elements to compute mean values.
		dim: The dim of x which a 'max' is performed.
	
	Returns:
		ret(Tensor): the mean value of an array along the dim.
	
	Examples:
		xxxxx

		>>> xxxx
		xxxx
	
	Note:
		xxxxxxx

	"""
	if not isinstance(x, Tensor):
		x = Tensor(x)
	if MOD == "IMM":
		tensor = Tensor(x.cached_data.mean(dim))
	else:
		tensor = Tensor.make_from_op(EWiseMean(), [x, dim])
		if MOD == "Async":
			tensor.realize_cached_data
	return tensor

def std(x: Tensor, dim=0):
	"""The standard deviation of the array elements along with the corresponding dim.

	Args:
		x(Tensor or array_like) : Elements to compute the standard deviation.
		dim: The dim of x which a 'std' is performed.
	
	Returns:
		ret(Tensor): the standard deviation of an array along the dim.
	
	Examples:
		xxxxx

		>>> xxxx
		xxxx
	
	Note:
		xxxxxxx

	"""
	if not isinstance(x, Tensor):
		x = Tensor(x)
	if MOD == "IMM":
		tensor = Tensor(x.cached_data.std(dim))
	else:
		tensor = Tensor.make_from_op(EWiseStd(), [x, dim])
		if MOD == "Async":
			tensor.realize_cached_data
	return tensor

def var(x: Tensor, dim=0):
	"""The variance of the array elements along with the corresponding dim.

	Args:
		x(Tensor or array_like) : Elements to compute the variance.
		dim: The dim of x which a 'var' is performed.
	
	Returns:
		ret(Tensor): the variance of an array along the dim.
	
	Examples:
		xxxxx

		>>> xxxx
		xxxx
	
	Note:
		xxxxxxx

	"""
	if not isinstance(x, Tensor):
		x = Tensor(x)
	if MOD == "IMM":
		tensor = Tensor(x.cached_data.var(dim))
	else:
		tensor = Tensor.make_from_op(EWiseVar(), [x, dim])
		if MOD == "Async":
			tensor.realize_cached_data
	return tensor

def sqrt(x: Tensor):
	"""Compute the non-negative square-root of the array elements, elementwise.

	Args:
		x(Tensor or array_like) : Elements to compute the non-negative square-root.
	
	Returns:
		ret(Tensor): the elementwise non-negative square-root of an array.
	
	Examples:
		xxxxx

		>>> xxxx
		xxxx
	
	Note:
		xxxxxxx

	"""
	if not isinstance(x, Tensor):
		x = Tensor(x)
	if MOD == "IMM":
		tensor = Tensor(x.cached_data.sqrt())
	else:
		tensor = Tensor.make_from_op(EWiseSqrt(), [x])
		if MOD == "Async":
			tensor.realize_cached_data
	return tensor

def abs(x: Tensor):
	"""Compute the absolute values of an array, elementwise.

	Args:
		x(Tensor or array_like) : Elements to compute the absolute values.
	
	Returns:
		ret(Tensor): the elementwise absolute values of an array.
	
	Examples:
		xxxxx

		>>> xxxx
		xxxx
	
	Note:
		xxxxxxx

	"""
	if not isinstance(x, Tensor):
		x = Tensor(x)
	if MOD == "IMM":
		tensor = Tensor(x.cached_data.abs())
	else:
		tensor = Tensor.make_from_op(EWiseAbs(), [x])
		if MOD == "Async":
			tensor.realize_cached_data
	return tensor

def loge(x: Tensor):
	"""Perform the natural logarithm of an array, elementwise.

	The natural logarithm is logarithm in base e, so that `log(exp(x)) = x`

	Args:
		x(Tensor or array_like) : Elements to perform the natural logarithm.
	
	Returns:
		ret(Tensor): the natural logarithm of an array.
	
	Examples:
		xxxxx

		>>> xxxx
		xxxx
	
	Note:
		xxxxxxx

	"""
	if not isinstance(x, Tensor):
		x = Tensor(x)
	if MOD == "IMM":
		tensor = Tensor(x.cached_data.loge())
	else:
		tensor = Tensor.make_from_op(EWiseLoge(), [x])
		if MOD == "Async":
			tensor.realize_cached_data
	return tensor

def log10(x: Tensor):
	"""Perform the base-10 logarithm of an array, elementwise.

	Args:
		x(Tensor or array_like) : Elements to perform the base-10 logarithm.
	
	Returns:
		ret(Tensor): the base-10 logarithm of an array.
	
	Examples:
		xxxxx

		>>> xxxx
		xxxx
	
	Note:
		xxxxxxx

	"""
	if not isinstance(x, Tensor):
		x = Tensor(x)
	if MOD == "IMM":
		tensor = Tensor(x.cached_data.log10())
	else:
		tensor = Tensor.make_from_op(EWiseLog10(), [x])
		if MOD == "Async":
			tensor.realize_cached_data
	return tensor

def log2(x: Tensor):
	"""Perform the base-2 logarithm of an array, elementwise.

	Args:
		x(Tensor or array_like) : Elements to perform the base-2 logarithm.
	
	Returns:
		ret(Tensor): the base-2 logarithm of an array.
	
	Examples:
		xxxxx

		>>> xxxx
		xxxx
	
	Note:
		xxxxxxx

	"""
	if not isinstance(x, Tensor):
		x = Tensor(x)
	if MOD == "IMM":
		tensor = Tensor(x.cached_data.log2())
	else:
		tensor = Tensor.make_from_op(EWiseLog2(), [x])
		if MOD == "Async":
			tensor.realize_cached_data
	return tensor

def sin(x: Tensor):
	"""Perform sine of an array, elementwise.

	Args:
		x(Tensor or array_like) : Elements to perform the sine.
	
	Returns:
		ret(Tensor): the sin values of an array.
	
	Examples:
		xxxxx

		>>> xxxx
		xxxx
	
	Note:
		xxxxxxx

	"""

	if not isinstance(x, Tensor):
		x = Tensor(x)
	if MOD == "IMM":
		tensor = Tensor(x.cached_data.sin())
	else:
		tensor = Tensor.make_from_op(EWiseSin(), [x])
		if MOD == "Async":
			tensor.realize_cached_data
	return tensor

def cos(x: Tensor):
	"""Perform cosine of an array, elementwise.

	Args:
		x(Tensor or array_like) : Elements to perform cosine.
	
	Returns:
		ret(Tensor): the cosine values of an array.
	
	Examples:
		xxxxx

		>>> xxxx
		xxxx
	
	Note:
		xxxxxxx

	"""
	if not isinstance(x, Tensor):
		x = Tensor(x)
	if MOD == "IMM":
		tensor = Tensor(x.cached_data.cos())
	else:
		tensor = Tensor.make_from_op(EWiseCos(), [x])
		if MOD == "Async":
			tensor.realize_cached_data
	return tensor

def tan(x: Tensor):
	"""Compute tangent of an array, elementwise.

	Args:
		x(Tensor or array_like) : Elements to compute tangent.
	
	Returns:
		ret(Tensor): the tangent values of an array.
	
	Examples:
		xxxxx

		>>> xxxx
		xxxx
	
	Note:
		xxxxxxx

	"""
	if not isinstance(x, Tensor):
		x = Tensor(x)
	if MOD == "IMM":
		tensor = Tensor(x.cached_data.tan())
	else:
		tensor = Tensor.make_from_op(EWiseTan(), [x])
		if MOD == "Async":
			tensor.realize_cached_data
	return tensor

def arcsin(x: Tensor):
	"""Inverse sine of an array, elementwise.

	So that if `y = sin(x)` then `x = arcsin(y)`

	Args:
		x(Tensor or array_like) : Elements to inverse sine.
	
	Returns:
		ret(Tensor): the inverse sine values of an array.
	
	Examples:
		xxxxx

		>>> xxxx
		xxxx
	
	Note:
		return nan if element not in [-1, 1]

	"""
	if not isinstance(x, Tensor):
		x = Tensor(x)
	if MOD == "IMM":
		tensor = Tensor(x.cached_data.arcsin())
	else:
		tensor = Tensor.make_from_op(EWiseArcSin(), [x])
		if MOD == "Async":
			tensor.realize_cached_data
	return tensor

def arccos(x: Tensor):
	"""Inverse cosine of an array, elementwise.

	So that if `y = cos(x)` then `x = arccos(y)`

	Args:
		x(Tensor or array_like) : Elements to inverse cosine.
	
	Returns:
		ret(Tensor): the inverse cosine values of an array.
	
	Examples:
		xxxxx

		>>> xxxx
		xxxx
	
	Note:
		return nan if element not in [-1, 1]

	"""
	if not isinstance(x, Tensor):
		x = Tensor(x)
	if MOD == "IMM":
		tensor = Tensor(x.cached_data.arccos())
	else:
		tensor = Tensor.make_from_op(EWiseArcCos(), [x])
		if MOD == "Async":
			tensor.realize_cached_data
	return tensor

def arctan(x: Tensor):
	"""Inverse tan of an array, elementwise.

	So that if `y = tan(x)` then `x = arctan(y)`

	Args:
		x(Tensor or array_like) : Elements to calculate the inverse tan.
	
	Returns:
		ret(Tensor): the inverse tan values of an array.
	
	Examples:
		xxxxx

		>>> xxxx
		xxxx
	
	Note:
		xxxxx

	"""
	if not isinstance(x, Tensor):
		x = Tensor(x)
	if MOD == "IMM":
		tensor = Tensor(x.cached_data.arctan())
	else:
		tensor = Tensor.make_from_op(EWiseArcTan(), [x])
		if MOD == "Async":
			tensor.realize_cached_data
	return tensor

def exp(x: Tensor):
	"""Calculate the exponent of an array, elementwise.

	Args:
		x(Tensor or array_like) : Elements to calculate the exponent.
	
	Returns:
		ret(Tensor): the exponent values of an array.
	
	Examples:
		xxxxx

		>>> xxxx
		xxxx
	
	Note:
		xxxxx

	"""
	if not isinstance(x, Tensor):
		x = Tensor(x)
	if MOD == "IMM":
		tensor = Tensor(x.cached_data.exp())
	else:
		tensor = Tensor.make_from_op(EWiseExp(), [x])
		if MOD == "Async":
			tensor.realize_cached_data
	return tensor

def ceil(x: Tensor):
	"""Ceiling all elements of an array, elementwise.

	Args:
		x(Tensor or array_like) : Elements to be ceiled.
	
	Returns:
		ret(Tensor): the elements after ceiling.
	
	Examples:
		xxxxx

		>>> xxxx
		xxxx
	
	Note:
		xxxxx

	"""
	if not isinstance(x, Tensor):
		x = Tensor(x)
	if MOD == "IMM":
		tensor = Tensor(x.cached_data.ceil())
	else:
		tensor = Tensor.make_from_op(EWiseCeil(), [x])
		if MOD == "Async":
			tensor.realize_cached_data
	return tensor

def floor(x: Tensor):
	"""flooring all elements of an array, elementwise.

	Args:
		x(Tensor or array_like) : Elements to be floored.
	
	Returns:
		ret(Tensor): the elements after flooring.
	
	Examples:
		xxxxx

		>>> xxxx
		xxxx
	
	Note:
		xxxxx

	"""
	if not isinstance(x, Tensor):
		x = Tensor(x)
	if MOD == "IMM":
		tensor = Tensor(x.cached_data.floor())
	else:
		tensor = Tensor.make_from_op(EWiseFloor(), [x])
		if MOD == "Async":
			tensor.realize_cached_data
	return tensor

def sign(x: Tensor):
	"""Get each sign of an array, elementwise.

	Args:
		x(Tensor or array_like) : Elements to get sign.
	
	Returns:
		ret(Tensor): the sign of each element.
	
	Examples:
		xxxxx

		>>> xxxx
		xxxx
	
	Note:
		xxxxx

	"""
	if not isinstance(x, Tensor):
		x = Tensor(x)
	if MOD == "IMM":
		tensor = Tensor(x.cached_data.sign())
	else:
		tensor = Tensor.make_from_op(EWiseSign(), [x])
		if MOD == "Async":
			tensor.realize_cached_data
	return tensor

def concatenate(arrays, dim=0, device=None):
	"""Concatenate all of the arrays along with the dim.

	Args:
		arrays (tuple) : The arrays waiting to concatenate.
		dim(int) : The dim to perform concatenation
	
	Returns:
		ret(Tensor): A new array after concatenating.
	
	Examples:
		xxxxx

		>>> xxxx
		xxxx
	
	Note:
		xxxxx

	"""
	assert isinstance(arrays, tuple), "The input arrays should be organized as tuple"
	cdds = tuple(Tensor(array).cached_data for array in arrays)
	return Tensor(concat(cdds, dim, device))


def where(condition, true_array, false_array):
	"""Return elements choosen from true_array or false_array depending on the condition

	Args:
		condition (Tensor or array_like) : Choose the element in true_array if the corresponding value in condition is true else choose the element in false_array
		true_array, false_array (Tensor or array_like) : Values from which to choose
	
	Returns:
		ret(Tensor): A new array in which elements from true_array(correponding index in condition is true) and false_array(correponding index in bool_array is false).
	
	Examples:
		xxxxx

		>>> xxxx
		xxxx
	
	Note:
		xxxxx

	"""
	
	if not isinstance(condition, Tensor):
		condition = Tensor(condition)
	if not isinstance(true_array):
		true_array = Tensor(true_array)
	if not isinstance(false_array):
		false_array = Tensor(false_array)
	if MOD == "IMM":
		tensor = Tensor(_where(bool_array=condition.cached_data, true_array=true_array.cached_data, false_array=false_array.cached_data))
	else:
		tensor = Tensor.make_from_op(EWiseWhere(), [condition, true_array, false_array])
		if MOD == "Async":
			tensor.realize_cached_data
	return tensor


def all(x:Tensor):
	"""Test whether all the elements in an array is true

	Args:
		x (Tensor or array_like, boolean) : The array to be tested
	
	Returns:
		ret(boolean)
	
	Examples:
		xxxxx

		>>> xxxx
		xxxx
	
	Note:
		xxxxx

	"""
	if not isinstance(x, Tensor):
		x = Tensor(x)
	if MOD == "IMM":
		tensor = Tensor(_all(x.cached_data))
	else:
		tensor = Tensor.make_from_op(EWiseAll(), [x])
		if MOD == "Async":
			tensor.realize_cached_data
	return tensor.cached_data


def any(x:Tensor):
	"""Test whether there is any element in an array is true

	Args:
		x (Tensor or array_like, boolean) : The array to be tested
	
	Returns:
		ret(boolean)
	
	Examples:
		xxxxx

		>>> xxxx
		xxxx
	
	Note:
		xxxxx

	"""
	if not isinstance(x, Tensor):
		x = Tensor(x)
	if MOD == "IMM":
		tensor = Tensor(_any(x.cached_data))
	else:
		tensor = Tensor.make_from_op(EWiseAny(), [x])
		if MOD == "Async":
			tensor.realize_cached_data
	return tensor.cached_data


def zeros(shape: tuple, dtype=float64, device_id=query_device()):
	"""Generate a new array with given shape and dtype, filled with zero

	Args:
		shape : the shape of the new 'Tensor' array
		dtype : The desired dtype of the new 'Tensor' array. Default is HyperGP.float64
		device_id(int) : The id of a device in which a new array to be loaded. Default is the first workable device id.
	
	Returns:
		ret(boolean)
	
	Examples:
		xxxxx

		>>> xxxx
		xxxx
	
	Note:
		xxxxx

	"""
	tensor = Tensor(_zeros(shape, dtype=dtype, device_id=device_id))
	return tensor

def empty(shape: tuple, dtype=float64, device_id=query_device()):
	"""Generate a new array with given shape and dtype, without filling values

	Args:
		shape : the shape of the new 'Tensor' array
		dtype : The desired dtype of the new 'Tensor' array. Default is HyperGP.float64
		device_id(int) : The id of a device in which a new array to be loaded. Default is the first workable device id.
	
	Returns:
		ret(boolean)
	
	Examples:
		xxxxx

		>>> xxxx
		xxxx
	
	Note:
		xxxxx

	"""
	tensor = Tensor(_empty(shape, dtype=dtype, device_id=device_id))
	return tensor

def ones(shape: tuple, dtype=float64, device_id=query_device()):
	"""Generate a new array with given shape and dtype, filled with one

	Args:
		shape : the shape of the new 'Tensor' array
		dtype : The desired dtype of the new 'Tensor' array. Default is HyperGP.float64
		device_id(int) : The id of a device in which a new array to be loaded. Default is the first workable device id.
	
	Returns:
		ret(boolean)
	
	Examples:
		xxxxx

		>>> xxxx
		xxxx
	
	Note:
		xxxxx

	"""
	tensor = Tensor(_ones(shape, dtype=dtype, device_id=device_id))
	return tensor

def full(shape: tuple, fill_value: float, dtype=None, device_id=query_device()):
	"""Generate a new array with given shape and dtype, filled with given 'fill_value'

	Args:
		shape : the shape of the new 'Tensor' array
		fill_value(float) : The value to be filled in the new array
		dtype : The desired dtype of the new 'Tensor' array. Default is HyperGP.float64
		device_id(int) : The id of a device in which a new array to be loaded. Default is the first workable device id.
	
	Returns:
		ret(boolean)
	
	Examples:
		xxxxx

		>>> xxxx
		xxxx
	
	Note:
		xxxxx

	"""
	tensor = Tensor(_full(shape, fill_value, dtype=dtype, device_id=device_id))
	return tensor

def uniform(low=0, high=1, shape=(1,), dtype=float64, device_id=query_device()):
	tensor = Tensor(_uniform(low, high, shape, dtype, device_id))
	return tensor

################# matrix operations #########################

def T(x: Tensor, dim=0):
	if not isinstance(x, Tensor):
		x = Tensor(x)
	if MOD == "IMM":
		tensor = Tensor(x.cached_data.T(dim))
	else:
		tensor = Tensor.make_from_op(EWiseTDim(), [x, dim])
		if MOD == "Async":
			tensor.realize_cached_data
	return tensor

def dot(x: Tensor, y: Tensor, dim_0=0, dim_1=0):
	if not isinstance(x, Tensor):
		x = Tensor(x)
	if not isinstance(y, Tensor):
		y = Tensor(y)
	if MOD == "IMM":
		return Tensor(x.cached_data.dot(y.cached_data, dim_0, dim_1))
	else:
		tensor = Tensor.make_from_op(EWiseDotDim(), [x, y, dim_0, dim_1])
		if MOD == "Async":
			tensor.realize_cached_data
	return tensor

def inv(x: Tensor):
	if not isinstance(x, Tensor):
		x = Tensor(x)
	if MOD == "IMM":
		tensor = Tensor(x.cached_data.inv())
	else:
		tensor = Tensor.make_from_op(EWiseInvDim(), [x])
		if MOD == "Async":
			tensor.realize_cached_data
	return tensor

def det(x: Tensor):
	if not isinstance(x, Tensor):
		x = Tensor(x)
	if MOD == "IMM":
		tensor = Tensor(x.cached_data.det())
	else:
		tensor = Tensor.make_from_op(EWiseDetDim(), [x])
		if MOD == "Async":
			tensor.realize_cached_data
	return tensor

def diagonal_sum(x: Tensor):
	if not isinstance(x, Tensor):
		x = Tensor(x)
	if MOD == "IMM":
		tensor = Tensor(x.cached_data.diagonal_sum())
	else:
		tensor = Tensor.make_from_op(EWiseDiagonalSum(), [x])
		if MOD == "Async":
			tensor.realize_cached_data
	return tensor

def assign2(x: Tensor, y: Tensor):
	if not isinstance(x, Tensor):
		x = Tensor(x)
	if not isinstance(y, Tensor):
		y = Tensor(y)
	if MOD == "IMM":
		x.cached_data[:] = y.cached_data
		return x
	else:
		tensor = Tensor.make_from_op(EWiseAssign_R2L(), [x, y])
		if MOD == "Async":
			tensor.realize_cached_data
	return tensor

def assign1(x: Tensor):
	if not isinstance(x, Tensor):
		x = Tensor(x)
	if MOD == "IMM":
		return x
	else:
		tensor = Tensor.make_from_op(EWisePass(), [x])
		if MOD == "Async":
			tensor.realize_cached_data
	return tensor


assign1.idx = -1
add.idx = 0
sub.idx = 1
mul.idx = 2
div.idx = 3
sin.idx = 4
cos.idx = 5
loge.idx = 6
exp.idx = 7
assign2.idx = 11

if __name__ == "__main__":
	print("============================TEST:ops.py============================")
	test_l = [[1, 2], [3, 4]]
	test_l.reverse()
	print(test_l)