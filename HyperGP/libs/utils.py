from .states import ProgBuildStates
import random, numpy as np
from numba import jit

class ProgBuildMethod:
    def __call__(self, cond: ProgBuildStates, node_states=None):
        raise NotImplementedError("The '__call__' function details should be provided")


class HalfAndHalf(ProgBuildMethod):

    def __call__(self, cond: ProgBuildStates, node_states=None):
        pset = cond.pset
        depth_rg = cond.depth_rg

        max_depth = random.randint(depth_rg[0], depth_rg[1])
        if max_depth == 1:
            return [pset.selectTerminal()]
        

        root = pset.selectFunc() if depth_rg[0] > 1 else pset.select()
        max_arity = max([pset.genFunc(f_str).arity for f_str in pset.primitiveSet])
        stack = [None] * (max_arity ** (max_depth + 1) - 1)
        stack[0] = (root, 0)
        # print('max_arity', max_arity, max_arity ** max_depth, max_depth, depth_rg[0])
        fstack = [None] * (max_arity ** (max_depth + 1) - 1)
        stack_idx, fstack_idx = 1, 0
        border_1, border_2 = depth_rg[0] - 1, max_depth - 1
        while stack_idx > 0:
            stack_idx -= 1
            node, cur_depth = stack[stack_idx]
            # print('fstack_idx', fstack_idx, cur_depth, stack_idx)
            fstack[fstack_idx] = node
            fstack_idx += 1
            arity = node.arity
            stack[stack_idx:stack_idx + arity] = [(pset.selectFunc() if cur_depth < border_1 else (pset.select() if cur_depth < border_2 else pset.selectTerminal()), cur_depth + 1) for _ in range(arity)]
            stack_idx += arity
            # for _ in range(arity):
            #     if cur_depth < border_1:
            #         nodeval = pset.selectFunc()
            #     elif cur_depth < border_2:
            #         nodeval = pset.select()
            #     else:
            #         nodeval = pset.selectTerminal()
            #     stack[stack_idx] = (nodeval, cur_depth + 1)
            #     stack_idx += 1
        return fstack[:fstack_idx]

class GrowDefault(ProgBuildMethod):
    
    @staticmethod
    @jit(nopython=True)
    def gen_list(pset_arities, depth_rg_0, depth_rg_1, f_len, term_len):
        max_arity = max(pset_arities)
        
        max_depth = random.randint(depth_rg_0, depth_rg_1)
        if max_depth == 1:
            return np.array([random.randint(f_len, f_len + term_len - 1)], dtype=np.int32)

        fstack = np.empty(shape=(max_arity ** (max_depth + 1) - 1), dtype=np.int32)
        stack_idx, fstack_idx = 1, 0
        border_1, border_2 = depth_rg_0 - 1, max_depth - 1
        stack = np.empty(shape=((max_arity ** (max_depth + 1) - 1) * 2), dtype=np.int32)
        stack[0], stack[1] = random.randint(0, f_len - 1) if depth_rg_0 > 1 else random.randint(0, f_len + term_len - 1), 0
        while stack_idx > 0:
            stack_idx -= 1
            node, cur_depth = stack[stack_idx * 2], stack[stack_idx * 2 + 1]
            # print('fstack_idx', fstack_idx, cur_depth, stack_idx)
            fstack[fstack_idx] = node
            fstack_idx += 1
            arity = pset_arities[node]
            stack[stack_idx * 2:(stack_idx + arity) * 2] = [x for _ in range(arity) for x in [random.randint(0, f_len - 1) if cur_depth < border_1 else (random.randint(0, f_len + term_len - 1) if cur_depth < border_2 else random.randint(f_len, f_len + term_len - 1)), cur_depth + 1]]
            stack_idx += arity
        return fstack[:fstack_idx]


    def __call__(self, cond: ProgBuildStates, node_states=None):
        pset = cond.pset
        depth_rg = cond.depth_rg

        f_len, term_len = len(pset.primitiveSet), len(pset.terminalSet)
        pset_arities = [pset.genFunc(f_str).arity for f_str in pset.primitiveSet] + [pset.genTerminal(f_str).arity for f_str in pset.terminalSet]
        pset_names = pset.primitiveSet + pset.terminalSet
        pset_defs = [pset.genFunc(f_str) for f_str in pset.primitiveSet] + [pset.genTerminal(f_str) for f_str in pset.terminalSet]

        fstack = self.gen_list(pset_arities, depth_rg[0], depth_rg[1], f_len, term_len)
        # assert 0==1
        return [pset_defs[f] for f in fstack]



class Full(ProgBuildMethod):
    from HyperGP.libs.primitive_set import PrimitiveSet

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