from HyperGP.base.tree_basic import Node


class TreeNode(Node):

    def __init__(self, nodeval, states=None, **kwargs):
        # self.nodeval = nodeval  # 该节点的值，Func则对应Func类；特征则对应int;常量则对应float

        if states is not None:
            if 'module_states' not in states and 'states' not in states:
                super().__init__(nodeval, states=states, **kwargs)
            else:
                super().__init__(nodeval, **states, **kwargs)
        else:
            super().__init__(nodeval, **kwargs)



def buildNode(nodeval, node_states=None):
    if node_states is not None:
        if 'module_states' not in node_states:
            states = {'states': node_states['module_states']}
        else:
            states = node_states
        return TreeNode(nodeval, **states)