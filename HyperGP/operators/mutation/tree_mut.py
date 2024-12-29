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
    def __call__(self, prog, cond: ProgBuildStates, node_states=None, method=HalfAndHalf, **kwargs):
        subtr_1 = prog.slice(random.randint(0, len(prog) - 1))
        subtr_2 = method()(cond, node_states)
        prog[subtr_1] = subtr_2
        return prog


class RandHoistMut(MutMethod):
    def __call__(self, prog, cond: ProgBuildStates):

        rd_1 = random.randint(0, len(prog) - 1)
        subtr_1 = prog.slice(rd_1)
        subtr_2 = prog.slice(random.randint(rd_1, rd_1 + subtr_1 - 1))
        prog[subtr_1] = prog[subtr_2]

        return prog


class RandPointMut(MutMethod):
    def __call__(self, prog, cond: ProgBuildStates, node_states=None):
        rd_1 = random.randint(0, len(prog) - 1)
        rd_node = prog[rd_1]
        arity = rd_node.arity
        cdds = []
        if arity == 0:
            cdd = cond.pset.selectTerminal(random)
        else:
            func_set = cond.pset.primitiveSet
            for i in range(len(func_set)):
                if func_set[i].arity == arity:
                    cdds.append(i)
            cdd = cdds[random.randint(0, len(cdds) - 1)]
            cdd = cond.pset.selectFunc(cdd)
        new_node = TreeNode(cdd, node_states)
        prog[rd_1] = new_node
        return prog


