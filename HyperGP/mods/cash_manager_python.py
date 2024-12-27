import warnings

import numpy as np
from HyperGP.base.func_basic import Constant


class ListNode:
    def __init__(self, val):
        self.node = val
        self.next = None
        self.pre = None

    def delete(self):
        if self.next is not None:
            self.next.pre = self.pre
        if self.pre is not None:
            self.pre.next = self.next

class LinkList:
    def __init__(self):
        self.list_head = ListNode(-1)
        self.list_tail = ListNode(-1)
        self.list_head.next = self.list_tail
        self.list_tail.pre = self.list_head

    def insert(self, item, posi):
        if posi == 0:
            return self.push_front(item)

        if not isinstance(item, ListNode):
            node = ListNode(item)
        else:
            node = item

        c_node, c_posi = self.list_head, 0

        while c_node != self.list_tail:

            if posi == c_posi:
                node.next = c_node.next
                c_node.next.pre = node
                c_node.next = node
                node.pre = c_node
                return node
            c_node = c_node.next
            c_posi += 1

        return self.push_back(item)
    def push_back(self, item):
        if not isinstance(item, ListNode):
            node = ListNode(item)
        else:
            node = item

        node_pre = self.list_tail.pre
        self.list_tail.pre = node
        node.next. node.pre = self.list_tail, node_pre
        node_pre.next = node
        return node

    def push_front(self, item):
        if not isinstance(item, ListNode):
            node = ListNode(item)
        else:
            node = item
        node_next = self.list_head.next
        self.list_head.next = node
        node.next, node.pre = node_next, self.list_head
        node_next.pre = node
        return node

    def __getitem__(self, posi):
        pass

    def delete(self, posi):
        delete_node = None
        if posi < 0:
            init_node = self.list_tail.pre
            time = - posi - 1
            if init_node == self.list_head:
                warnings.WarningMessage('The input posi is out of range, where cash size is %d, but %d idx is given' % (0, posi))
                return None
            for t in range(time):
                if init_node != self.list_head:
                    init_node = init_node.pre
                else:
                    warnings.WarningMessage('The input posi is out of range, where cash size is %d, but %d idx is given' % (t, posi))
                    return None
            delete_node = init_node
            init_node.delete()
            return delete_node
        else:
            time = posi
            init_node = self.list_head.next
            if init_node == self.list_tail:
                warnings.WarningMessage('The input posi is out of range, where cash size is %d, but %d idx is given' % (0, posi))
                return None
            for t in range(time):
                if init_node != self.list_tail:
                    init_node = init_node.next
                else:
                    warnings.WarningMessage('The input posi is out of range, where cash size is %d, but %d idx is given' % (t, posi))
                    return None
            delete_node = init_node
            init_node.delete()
            return delete_node



    def __len__(self):
        c_node = self.list_head.next
        len = 0
        while c_node != self.list_tail:
            c_node = c_node.next
            len += 1
        return len

    def list(self):
        node_list = []
        c_node = self.list_head.next
        while c_node != self.list_tail:
            node_list.append(c_node)
            c_node = c_node.next
        return node_list

class CashManager:

    def __init__(self, limit=1e5):
        # super.__init__()
        self.cash = {}#{str: linknode}
        self.cash_list = LinkList()
        self.limit_size = limit

    def add(self):
        pass

    def __getitem__(self, item):
        return self.cash[item].node[1]

    def __setitem__(self, key, value):
        if key not in self.cash:
            link_node = self.cash_list.push_front((key, value))
            if len(self.cash_list) > self.limit_size:
                delete_node = self.cash_list.delete(-1)
                self.cash.pop(delete_node.node[0])
            self.cash[key] = link_node
        else:
            self.cash[key].node = (key, value)

    def getSemantic(self, item):

        if item in self.cash:
            linknode:ListNode = self.cash[item]
            linknode.delete()
            self.cash_list.push_front(linknode)
        return self.cash[item].node[1]

    # def set(self, subtrees: list, semantics: list):
    #     """set cash with semantic"""
    #     if len(semantics) != len(subtrees):
    #         raise ValueError("subtrees size '%d' not equal to semantics size '%d' " %(len(subtrees), len(semantics)))
    #     for i, subtr in enumerate(subtrees):
    #         if subtr not in self.cash:
    #             self[subtr] = semantics[i]


    def set(self, **kwargs):
        """set cash with semantic"""
        for key, value in kwargs.items():
            if key not in self.cash:
                self[key] = value

    # def set(self, subtree, inputs):
    #     """compile the node and run it to get semantic"""
    #     pass

    # """select some node to compute the semantics to store in cash"""
    # def select(self, ind):
    #     pass

    def update(self, pop):
        pass

    """
    generate a new ind list, with a cash list that replace the origin node.
    to gen with just one time traversal, use two list, one is the ind, the other one is generated for the record node.
    """
    def getCash(self, ind):
        record_sign, record_num = np.zeros(len(ind)), 0
        cash_sign, cash_num = np.zeros(len(ind)), 0
        cash_set = {}
        sym_set = {}
        ind_after_cash = []
        idx_after_cash = []
        symset_after_cash = {}

        c_childs = ind.list(childs=True)[0]

        '''To get cash list and the node with record point'''
        for idx in range(len(ind) - 1, -1, -1):
            node = ind[idx]
            if 'cash_record' in node.states and node.states['cash_record']:#[]TODO, need to reaffirm
                record_sign[idx] = 1
                record_num += 1
            if node.arity != 0:
                childs = [ind[c_idx] for c_idx in c_childs[idx]]
                sym = str(node.nodeval) + '('
                for child in childs:
                    sym += sym_set[child] + ', '
                sym = sym[:-2] + ')'
                sym_set[node] = sym
                if sym in self.cash:
                    cash_sign[idx] = 1
                    # cash_list.append(idx)
                    cash_num += 1
            else:
                if isinstance(node.nodeval, Constant):
                    sym_set[node] = str(node.nodeval)
                elif hasattr(node.nodeval, 'idx'):
                    sym_set[node] = str(node.nodeval)
                # else:
                #     record_sign[idx] = 1
                #     record_num += 1

        def scan(begin, record_sign, ind_list):
            end = begin + 1
            total = ind_list[begin].arity
            while total > 0:
                if record_sign[end] == 1:
                    return False, begin + 1
                # if cash_sign[end] == 1:
                #     cash_list.append(end)
                    # cash_arity.append(total - 1)
                total += ind_list[end].arity - 1
                # suc_list.append(ind[end])
                end += 1
            return True, end

        if cash_num > 0:
            idx = 0
            while idx < len(ind):
                if cash_sign[idx] == 1:
                    suc, new_idx = scan(idx, record_sign, ind)
                    if suc:
                        sym = sym_set[ind[idx]]
                        symset_after_cash[ind[idx]] = sym_set[ind[idx]]
                        cash_set[sym] = self.cash[sym].node[1]
                        idx = new_idx
                    else:
                        ind_after_cash.append(ind[idx])
                        idx_after_cash.append(idx)
                        idx += 1
                else:
                    ind_after_cash.append(ind[idx])
                    idx_after_cash.append(idx)
                    idx += 1
        else:
            ind_after_cash = ind
            cash_set = cash_set
            idx_after_cash = list(range(len(ind)))

        return ind_after_cash, cash_set, idx_after_cash, symset_after_cash



        # '''scan the subtree, whether there is a record'''
        # suc = True
        # end = idx + 1
        # total = node.arity
        # while total > 0:
        #     if record_list[end] == 1:
        #         record_list[
        #             idx] = 1  # sign that the ancestor of this subtree can not use cash, to avoid repeatly traversal
        #         suc = False
        #         break
        #     total += ind[end].arity - 1
        #     end += 1
        # if suc:
        #     cash_list[idx] = 1



        # return cash_set


if __name__ == '__main__':
    a, b = 1, [1, 2, 3]
    print(b, a, b)