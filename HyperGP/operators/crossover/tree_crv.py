
from HyperGP.libs.representation.TGP import TGPIndv
import random, copy

class CrossoverMethod:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("No find '__call__' implement")

class RandTrCrv(CrossoverMethod):
    
    """Perform the crossover operation on the two choiced programs.

       In this method, A random subtree of one program is selected to replace a random subtree of the other program.

    """

    def __call__(self, prog_1, prog_2, prob=1):
        """
        Call RandTrCrv method.

        Args:
            prog_1: The first program
            prog_2: The second program
            prob: The probability to perform the crossover between prog_1 and prog_2.
            
        Returns:
            A new prog
        """
        if random.uniform(0, 1) < prob:
            if prog_1 == prog_2:
                prog_2 = prog_1.copy()

            subtr_1 = prog_1.slice(random.randint(0, len(prog_1) - 1))
            subtr_2 = prog_2.slice(random.randint(0, len(prog_2) - 1))
            encode_1 = prog_1.get_encode(subtr_1)
            encode_2 = prog_2.get_encode(subtr_2)
            prog_1.set_encode(subtr_1, encode_2)
            prog_2.set_encode(subtr_2, encode_1)
            # prog_1[subtr_1], prog_2[subtr_2] = prog_2[subtr_2], prog_1[subtr_1]
        return prog_1, prog_2
