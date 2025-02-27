from HyperGP.libs.states import VarStates, ProgBuildStates
from HyperGP.libs.utils import HalfAndHalf
from HyperGP.libs.regression.tree import TreeNode
import random

class MutMethod:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("No find '__call__' implement")

class RandTrMut(MutMethod):
    """Perform the subtree mutation operation on the program.

       Subtree mutation selects a random subtree from the embedded program to
       be replaced. A donor subtree is generated at random and this is
       inserted into the original parent to form an offspring. This
       implementation uses the "headless chicken" method where the donor
       subtree is grown using the initialization methods and a subtree of it
       is selected to be donated to the parent.

    """
    def __call__(self, prog, cond: ProgBuildStates, prob=1, node_states=None, method=HalfAndHalf, **kwargs):
    
        """
        Call RandTrMut method.

        Args:
            prog: The individual
            cond: The 'cond' will be pass to the 'method' to generate the random subtree
            prob: The probability to perform the subtree mutation.
            method: The method called to generate subtree, in which the fixed parameter types:(cond, node_states) will be passed to in this call.

        Returns:
            A new prog
        """
        if random.uniform(0, 1) < prob:
            subtr_1 = prog.slice(random.randint(0, len(prog) - 1))
            subtr_2 = method()(cond, node_states)
            prog[subtr_1] = subtr_2
        return prog


class RandHoistMut(MutMethod):
    
    """Perform the hoist mutation operation on the program.

       Hoist mutation selects a random subtree from the embedded program, replacing it by a random subtree of the selected mutation subtree.

    """

    def __call__(self, prog, prob=1):
        """
        Call RandHoistMut method.

        Args:
            prog: The individual
            prob: The probability to perform the subtree mutation.
            
        Returns:
            A new prog
        """
        if random.uniform(0, 1) < prob:
            rd_1 = random.randint(0, len(prog) - 1)
            subtr_1 = prog.slice(rd_1)
            subtr_2 = prog.slice(random.randint(rd_1, rd_1 + subtr_1 - 1))
            prog[subtr_1] = prog[subtr_2]

        return prog


class RandPointMut(MutMethod):
    
    """Perform the point mutation operation on the program.

       Point mutation selects a random node from the embedded program, replacing it by a random node from pset with the same arity.

    """
    def __call__(self, prog, cond: ProgBuildStates, prob=1):
        """
        Call RandHoistMut method.

        Args:
            prog: The individual
            cond: Used to generate new node to make point mutation. The `primitiveSet` module should be in it.
            prob: The probability to perform the subtree mutation.
            
        Returns:
            A new prog
        """

        if random.uniform(0, 1) < prob:
            pset = cond.pset
            rd_posi = random.randint(0, len(prog) - 1)
            arity = prog.get_encode(rd_posi)[1]
            if arity == 0:
                prog[rd_posi] = cond.pset.selectTerminal()
            else:
                func_set = pset.primitiveSet
                cdds = [func for func in func_set if pset.genFunc(func).arity == arity]
                prog[rd_posi] = pset.genFunc(cdds[random.randint(0, len(cdds) - 1)])
        return prog


