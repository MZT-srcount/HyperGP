
from HyperGP.library.representation.TGP import TGPIndv
import random

class CrossoverMethod:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("No find '__call__' implement")

class RandTrCrv(CrossoverMethod):
    def __call__(self, prog_1, prog_2, rd_state=None):
        if rd_state is None:
            rd_state = random
        node_list_1 = prog_1.list()
        node_list_2 = prog_2.list()

        subtr_1 = prog_1.slice(rd_state.randint(0, len(node_list_1) - 1))
        subtr_2 = prog_2.slice(rd_state.randint(0, len(node_list_2) - 1))

        prog_1[subtr_1], prog_2[subtr_2] = prog_2[subtr_2], prog_1[subtr_1]
        return prog_1, prog_2
