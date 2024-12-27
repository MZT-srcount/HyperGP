from HyperGP.base.func_basic import BasePrimitiveSet
import random

class PrimitiveSet(BasePrimitiveSet):
    """
    ``PrimitiveSet`` module is used to collects the primitives and terminals used in GP evolution
    
    To use the PrimitiveSet, we should firstly import it from HyperGP and initilize it:
    
    >>> from HyperGP import PrimitiveSet

    """
    def select(self):
        """
        randomly select from the primitive set
        
        Returns:
            a ``Func``, ``Terminal`` or ``Constant`` module

        Examples:

            >>> prim = pset.select()
            >>> print(prim, type(prim))
            div, <class 'HyperGP.base.func_basic.Func'>

        """
        primitive_set = self.primitiveSet + self.terminalSet
        choice = random.randint(0, len(primitive_set) - 1)
        if choice < len(self.used_primitive_set):
            return self.genFunc(self.primitiveSet[choice])
        else:
            return self.genTerminal(self.terminalSet[choice - len(self.primitiveSet)])

    def selectFunc(self):
        """ 
        randomly select a function from the primitive set
        
        Returns:
            a ``Func`` module

        Examples:

            >>> prim = pset.selectFunc()
            >>> print(prim, type(prim))
            add, <class 'HyperGP.base.func_basic.Func'>

        """

        return self.genFunc(self.primitiveSet[random.randint(0, len(self.primitiveSet) - 1)])

    def selectTerminal(self):
        
        """ 
        randomly select a terminal from the primitive set
        
        Returns:
            a ``Terminal`` or ``Constant`` module

        Examples:

            >>> term = pset.selectTerminal()
            >>> print(term, type(term))
            x0, <class 'HyperGP.base.func_basic.Terminal'>

        """

        return self.genTerminal(self.terminalSet[random.randint(0, len(self.used_terminal_set) - 1)])
