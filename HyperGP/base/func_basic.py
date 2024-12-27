import random
from ..tensor_libs._src.basic import TensorOp
import copy
from .base_struct import States

class Func:
    def __init__(self, name: str, arity, states=None, **kwargs):
        self.name = name
        self.arity = arity
        self.kwargs = kwargs
        if states is not None:
            for key, value in states.items():
                setattr(self, key, value)

    def __call__(self, *args, **kwargs):
        assert hasattr(self, 'func'), 'Func object is not callable'
        # print('here......self.kwargs: ', self.kwargs)
        return self.func(*args, **self.kwargs, **kwargs)

    def __str__(self):
        return self.name

class Constant:
    idx=-1
    def __init__(self, val, **kwargs):
        self.val = val
        self.arity = 0
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        return str(self.val)


class Terminal:
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.arity = 0
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def type(self):
        if hasattr(self, 'ephemeral'):
            if self.ephemeral:
                return 'Constant'
            else:
                return 'Actual'
        else:
            return 'Input'

    def __str__(self):
        return self.name

'''TODO[]: currently, the func in primitiveSet is shared by all nodes using this primitiveSet'''
class BasePrimitiveSet:
    """
    primitive_set: [(name, func, arity), ..]
    terminal_set: [(name, type), ..]
    """

    def __init__(self, input_arity, primitive_set=None, prefix='x'):
        """
        Args:
            input_arity(int): the number of terminals.
            primitive_set(list): a list with the format [(name, func, arity, states), ..]. 
                for each prim, a tuple with `name`, `function`, `arity` is needed. \
                    To make the framework more flexible, it is also supported to register the states with `States` module for each prim, just add it behind the three element in each tuple.\
                    The detailed examples will be provided below.
            prefix(str): determine the name used for input terminals. The pre-defined str is 'x', then the terminals will be printed like 'x1', 'x2', ... . 

        
        Examples:

            Initialize the PrimitiveSet
            
            >>> pset = PrimitiveSet(
            >>> input_arity=1,
            >>> primitive_set=[
            >>>     ('add', HyperGP.tensor.add, 2),
            >>>     ('sub', HyperGP.tensor.sub, 2),
            >>>     ('mul', HyperGP.tensor.mul, 2),
            >>>     ('div', HyperGP.tensor.div, 2),
            >>> ])

            For each prim, we can also register additional states.\
            For example, when we apply HyperGP to image classification:
            
            >>> pset = PrimitiveSet(
            >>> input_arity=1,
            >>> primitive_set=[
                ("gau_filter", HyperGP.gauss_filter, 3, States(type="filter"))
            >>> ])
            >>> print(pset.genFunc("gau_filter").type)
            filter

        """
        self.mod = True
        self.prefix = prefix
        self.used_primitive_set, self.func_count = {}, 0
        self.used_terminal_set, self.terminal_count = {}, 0
        self.__primitive_set, self.__terminal_set = [], []
        self.arguments, self.context = [], {}
        if primitive_set is not None:
            self.__registerPrimitive(primitive_set)

        terminal_set = []
        for i in range(input_arity):
            terminal_set.append(self.prefix + str(i))
        self.__registerTerminal(terminal_set)

    def __registerPrimitive(self, primitive_set, **kwargs):
        for primitive in primitive_set:
            if not isinstance(primitive[1], TensorOp) or not hasattr(primitive[1], "idx"):
                self.mod = False
                break
        for primitive in primitive_set:
            if self.mod:
                states = States(idx=primitive[1].idx, func=primitive[1])
            else:
                states = States(idx=self.func_count, func=primitive[1])
            if len(primitive) >= 4 and isinstance(primitive[3], States):
                states.update(primitive[3])
            else:
                assert len(primitive) < 4, "The param '4' in the input should be states, {PARAM} is find".format(PARAM=primitive[3])
            
            if self.mod:
                self.used_primitive_set[primitive[0]] = \
                    Func(name=primitive[0], arity=primitive[2], states=states, **kwargs)
            else:
                self.used_primitive_set[primitive[0]] = \
                    Func(name=primitive[0], arity=primitive[2], states=states, **kwargs)
            self.__primitive_set.append(primitive[0])
            self.context[primitive[0]] = primitive[1]
            self.func_count += 1
        if self.mod:
            self.func_count = max([primitive[1].idx for primitive in primitive_set] + [self.func_count]) + 1

    def registerPrimitive(self, name, func, arity, states:States=None, **kwargs):
        """
        We can also register function after the ``PrimitiveSet`` module has been initialized.

        Args:
            name(str): a sign of the function, which will be shown when print the function. It is also used to search the register function, with ``genFunc`` function.
            func(function-like): the function want to register.
            arity(int): the arity of the register function.
            states(HyperGP.States): the states want to register in the function, using ``HyperGP.States`` module.
            kwargs: The input kwargs for each prim will be used as default parameters whenever the prim is called.\

        Examples:

            >>> pset = PrimitiveSet(
            >>>     input_arity=1
            >>> )
            >>> 
            >>> param_types_1 = ["img", "mask", "channel"]
            >>> param_types_2 = ["img", "channel"]
            >>> param_types_3 = ["img", "w_h", "region"]
            >>> 
            >>> pset.registerPrimitive("gau_filter", HyperGP.gauss_filter, 3, states=States(type="filter", param=param_types_1), padding=(1, 1))
            >>> pset.registerPrimitive("sobel_filter", HyperGP.sobel_filter, 2, states=States(type="filter", param=param_types_2), padding=(1, 1))
            >>> pset.registerPrimitive("mean_filter", HyperGP.mean_filter, 3, states=States(param=param_types_1, type="filter"), padding=(1, 1))
            >>> pset.registerPrimitive("mean", s_mean, 1, states=States(param=["img"], type="norm"))
            >>> pset.registerPrimitive("region_detect", region_detect, 3, states=States(param=param_types_3, type="region"))
                
        """
        # self.used_func_set.append(Func(idx=self.func_count, name=primitive[0], arity=primitive[2], **kwargs))
        
        if isinstance(func, TensorOp) and self.mod and hasattr(func, "idx"):
            states_basic = States(idx=func.idx, func=func)
        else:
            states_basic = States(idx=self.func_count, func=func)

        if states is not None:
            states.update(states_basic)
        else:
            states = states_basic
        if isinstance(func, TensorOp) and self.mod and hasattr(func, "idx"):
            self.used_primitive_set[name] = Func(name=name, arity=arity, states=states, **kwargs)
            self.func_count = max(func.idx + 1, self.func_count)
        else:
            self.used_primitive_set[name] = Func(name=name, arity=arity, states=states, **kwargs)
            self.mod = False
            self.func_count += 1
        self.__primitive_set.append(name)
        self.context[name] = func

    def __registerTerminal(self, terminal_set, **kwargs):
        for i, terminal in enumerate(terminal_set):
            # self.used_terminal_set.append([terminal[0], Terminal(idx=self.terminal_count, type=terminal[1], **kwargs)])
            # self.terminal_index[terminal] = self.terminal_count
            self.used_terminal_set[terminal] = Terminal(name=terminal, idx=self.terminal_count, **kwargs)
            self.__terminal_set.append(terminal)
            self.arguments.append(terminal)
            self.terminal_count += 1

    """terminal_set: [(name, func), ..]"""
    def registerEphemeralTerminal(self, name, func, ephemeral=True, **kwargs):
        """
        Used to generate the terminal with functions

        Args:
            name(str): a sign of the terminal. It is used to search the register terminal, with ``genTerminal`` function.
            function(function-like): The function called when get it from the ``PrimitiveSet`` module

        Examples:

            >>> pset = PrimitiveSet(
            >>>     input_arity=2
            >>> )
            >>> def constants():
            ... return random.uniform(0, 1)
            >>> pset.registerEphemeralTerminal("y", constants)
            >>> term = pset.genTerminal("y")
            >>> print(term, type(term))
            0.9997361496151884 <class 'HyperGP.base.func_basic.Constant'>
        """
        if not ephemeral and name in self.context.keys():
            raise ValueError('terminals are required to have a unique name')
        if not ephemeral:
            self.context[name] = func
        self.used_terminal_set[name] = Terminal(name=name, func=func, ephemeral=ephemeral, idx=-1, **kwargs)
        self.__terminal_set.append(name)
        self.terminal_count += 1

    def registerTerminal(self, name=None, **kwargs):
        """
        Except for the auto generated terminals, we can also register terminals after the ``PrimitiveSet`` module has been initialized.

        Args:
            name(str): a sign of the terminal, which will be shown when print it. It is also used to search the register terminal, with ``genTerminal`` function.
            kwargs: will be registered to the attrs of the terminal.

        Examples:

            >>> pset = PrimitiveSet(
            >>>     input_arity=2
            >>> )
            >>> 
            >>> pset.registerTerminal("y")
            >>> print(pset.terminalSet)
            ['x0', 'x1', 'y']
            >>> pset.registerTerminal()
            >>> print(pset.terminalSet)
            ['x0', 'x1', 'y', 'x2']
        """

        if name == None:
            name = self.prefix + str(self.terminal_count)
        self.used_terminal_set[name] = Terminal(name=name, idx=self.terminal_count, **kwargs)
        self.__terminal_set.append(name)
        self.arguments.append(name)
        self.terminal_count += 1
    
    
    @property
    def primitiveSet(self):
        """
        Get a name list of the register functions

        Returns:
            The name list of the register functions

        Examples:
            >>> pset = PrimitiveSet(
            >>> input_arity=1,
            >>> primitive_set=[
            >>>     ('add', HyperGP.tensor.add, 2),
            >>>     ('sub', HyperGP.tensor.sub, 2),
            >>> ])
            >>> print(pset.primitiveSet)
            ['add', 'sub']

        """
        return self.__primitive_set

    @property
    def terminalSet(self):
        """
        Get a name list of the terminals

        Returns:
            The name list of the register terminals

        Examples:
            >>> pset = PrimitiveSet(
            >>> input_arity=5,
            >>> primitive_set=[
            >>>     ('add', HyperGP.tensor.add, 2),
            >>>     ('sub', HyperGP.tensor.sub, 2),
            >>> ])
            >>> print(pset.terminalSet)
            ['x0', 'x1', 'x2', 'x3', 'x4']
            
        """
        return self.__terminal_set

    def select(self, **kwargs):
        raise NotImplementedError("The implementation of function 'select' not provided")

    def max_arity(self):
        """
        Statistics the max arity of the registered function

        Returns:
            A new ``PrimitiveSet``
        """

        return max(list(map(lambda x: x.arity, self.used_primitive_set)))

    def genFunc(self, name):
         
        """
        Search the register function with its name

        Args:
            name(str): the function name want to search

        Returns:
            The callable register function

        Examples:
            >>> pset = PrimitiveSet(
            >>> input_arity=5,
            >>> primitive_set=[
            >>>     ('add', HyperGP.tensor.add, 2),
            >>>     ('sub', HyperGP.tensor.sub, 2),
            >>> ])
            >>> prim = pset.genFunc('add')
            >>> print(prim, type(prim))
            add, <class 'HyperGP.base.func_basic.Func'>
            
        """

        # return Func(name=name, arity=self.used_primitive_set[name].arity)
        return self.used_primitive_set[name]

    def genTerminal(self, name):
         
        """
        Search the register terminal with its name

        Args:
            name(str): the terminal name want to search

        Returns:
            The register terminal. 

        Note:
            If it is a callable ephemeral constant, then the generated term is a ``Constant`` module with the return value of the function.
        
        Examples:
            >>> pset = PrimitiveSet(
            >>> input_arity=5,
            >>> primitive_set=[
            >>>     ('add', HyperGP.tensor.add, 2),
            >>>     ('sub', HyperGP.tensor.sub, 2),
            >>> ])
            >>> term = pset.genTerminal('x0')
            >>> print(term, type(term))
            x0, <class 'HyperGP.base.func_basic.Terminal'>
            
        """

        if self.used_terminal_set[name].type == 'Constant':
            return Constant(self.used_terminal_set[name].func())
        else:
            # return Terminal(name)
            return self.used_terminal_set[name]
    
    def copy(self):
        """
        A deep copy of the primitive set

        Returns:
            A new ``PrimitiveSet``
        """
        return copy.deepcopy(self)
    
if __name__ == '__main__':
    c = Terminal('100')
    print(c.__dict__)