
class MutMethod:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("No find '__call__' implement")
    
class SLGEPMut(MutMethod):
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