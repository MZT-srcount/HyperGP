import time
from HyperGP.base.base_struct import States, BaseStruct
import multiprocessing
from .mod_base import ModBase
import dill
import _pickle as cPickle
import statistics
from multiprocessing.sharedctypes import RawArray as SArray
import ctypes
import random

# sharedmemory = None

def init_pool(array, seeds):
    # global sharedmemory
    # sharedmemory = array
    random.seed(seeds)
    
class CallBack:
    def __call__(self, res):
        et = time.time()
        self.run_time = (et, res[0])
        self.res = res[1]

def _run(funcs, main_core, states_list_dumps):
    st = time.time()
    res_list = []
    if not main_core:
        states_list = cPickle.loads(states_list_dumps)
        funcs_list = funcs
        
        kwargs = states_list[-1]
        for i, paras in enumerate(states_list[:-1]):
            res_list.append(funcs_list[i](**paras, **kwargs))
    else:
        kwargs = states_list_dumps[-1]
        res_list = list(map(lambda idx: funcs[idx](**states_list_dumps[idx], **kwargs), range(len(states_list_dumps[:-1]))))
    time_cost = time.time() - st
    # print('res_list[0]: ', len(res_list[0]))
    
    return (time_cost, res_list)

class MultiProcess(ModBase):

    def __init__(self):
        self.funcs = {}

    def _popSet(self, pop):
        print("process, popSet")
        self.core_count = multiprocessing.cpu_count() if multiprocessing.cpu_count() < 10 else 10
        # pop.process_manager = self
        '''initialize the process pool and keep it in the class, to avoid the frequent intialization'''
        # global sharedmemory
        # sharedmemory = SArray('c', 1024 ** 3)
        self.process_pool = multiprocessing.Pool(processes=self.core_count, initializer=init_pool, initargs=(None, random.random()))
        pop.gmoduleRegister(parallel=self.__call__)

    def __register(self, func, cpu_count):

        if not isinstance(func, list) and func not in self.funcs:
            self.funcs[func] = States()
            self.funcs[func].count = cpu_count
        elif isinstance(func, list):
            for f in func:
                if f not in self.funcs:
                    self.funcs[f] = States()
                    self.funcs[f].count = cpu_count


    """
    states:     [States x n], which is devided in process, used to store the independent vars #[[x n], [x n], ...]
    kwargs:     will be copied to each process. For better acceleration, it is suggested to pass only the nessary parameters for every process
    mask:       independent with mask: list, corresponding to the states
    """

    def __call__(self, func, states, **kwargs):
        st_0 = time.time()
        '''record the function, for better acceleration'''
        self.__register(func, self.core_count)

        # '''translate the **kwargs to a dict for multiprocessing transform'''
        # kwargs_dict = dict(**kwargs)

        # divd_states = []
        # if mask is None:
        #     mask = list(range(0, len(states)))
        #     divd_states = states
        # else:
        #     for item in mask:
        #         divd_states.append(states[item])
        st_1 = time.time()
        progs, callbacks = [], []
        run_times_record = []
        if isinstance(func, list):
            core_count = statistics.mean([self.funcs[f].count for f in func])
        else:
            core_count = self.funcs[func].count
        avg_count = int(len(states) / (core_count)) if len(states) > core_count else len(states)

        st = time.time()
        if not isinstance(func, list):
            func = [func] * len(states)
        elif len(func) != len(states):
            raise ValueError("The size of func list '%d' should equal to mask len '%d'" % (len(func), len(states)))

        # global sharedmemory
        len_items = []
        dicts_list = []
        funcs_list = []
        count_fp = len(states) - avg_count
        # assert len(states) == len(progs), '{0}, {1}'.format(len(states), len(progs))
        
        states_maincore = states[:len(states) % avg_count]
        funcs_maincore = func[:len(states) % avg_count]
        
        st = time.time()
        
        for i in range(len(states) % avg_count, len(states), avg_count):
            run_times_record.append(time.time())
            func_for_p = func[i:i + avg_count]
            dict_for_p = states[i:i + avg_count] + [kwargs]
            
            dicts_list.append(cPickle.dumps(dict_for_p))
            funcs_list.append(func_for_p)
        
        for i in range(len(dicts_list)):
            callbacks.append(CallBack())
            progs.append(
                self.process_pool.apply_async(_run, args=(funcs_list[i], False, dicts_list[i]),
                                              callback=callbacks[-1])
            )
        res_f = _run(funcs_maincore, True, states_maincore + [kwargs])
        
        res = res_f[1]
        transfer_cost, run_cost = 0, 0
        for i in range(len(dicts_list)):
            progs[i].wait()
            run_cost += callbacks[i].run_time[1]
            transfer_cost += callbacks[i].run_time[0] - run_times_record[i] - callbacks[i].run_time[1]
            res.extend(callbacks[i].res)
        # print('t2: ', time.time() - st_1)
        # print('run_cost, transfer_cost, core_count', run_cost / len(dicts_list), transfer_cost / len(dicts_list), self.funcs[func[0]].count)
        
        self.funcs[func[0]].count *= int(run_cost / transfer_cost)
        if self.funcs[func[0]].count > self.core_count:
            self.funcs[func[0]].count = self.core_count
        elif self.funcs[func[0]].count == 0:
            self.funcs[func[0]].count = 1
        
        # print('t3: ', time.time() - st_1, time.time() - st_0) 
        
        return res
