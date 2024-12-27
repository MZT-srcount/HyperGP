from HyperGP.library.states import ProgBuildStates
from HyperGP.library.regression.tree import TreeNode
import random

class ProgBuildMethod:
    def __call__(self, cond: ProgBuildStates, node_states=None):
        raise NotImplementedError("The '__call__' function details should be provided")


class HalfAndHalf(ProgBuildMethod):

    def __call__(self, cond: ProgBuildStates, node_states=None):
        pset = cond.pset
        rd_state = random
        depth_rg = cond.depth_rg

        max_depth = rd_state.randint(depth_rg[0], depth_rg[1])
        if max_depth == 1:
            return [pset.selectTerminal()]

        if depth_rg[0] > 1:
            nodeval = pset.selectFunc()
        else:
            nodeval = pset.select()

        root = nodeval
        stack = [(root, 0)]
        fstack = []

        while stack:
            node, cur_depth = stack.pop()
            fstack.append(node)
            for i in range(node.arity):
                if cur_depth < depth_rg[0]:
                    nodeval = pset.selectFunc()
                elif cur_depth < max_depth:
                    nodeval = pset.select()
                else:
                    nodeval = pset.selectTerminal()
                stack.append((nodeval, cur_depth + 1))
        return fstack


class Full(ProgBuildMethod):
    from HyperGP.library.primitive_set import PrimitiveSet

    def __call__(self, cond: ProgBuildStates, node_states=None):
        pset = cond.pset
        rd_state = random
        depth_rg = cond.depth_rg

        if depth_rg[1] == 1:
            return pset.selectTerminal(rd_state)

        root = pset.selectFunc()
        stack = [(root, 0)]
        fstack = []


        while stack:
            node, cur_depth = stack.pop()
            fstack.append(node)
            for i in range(node.arity):
                if cur_depth < depth_rg[1]:
                    nodeval = pset.selectFunc()
                else:
                    nodeval = pset.selectTerminal()
                stack.append((nodeval, cur_depth + 1))

        return fstack



"""=========================TEST========================"""
if __name__ == '__main__':
    print(type(HalfAndHalf))