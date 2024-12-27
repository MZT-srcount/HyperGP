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

sharedmemory = None

def init_pool(array, seeds):
    global sharedmemory
    sharedmemory = array
    random.seed(seeds)
    
class CallBack:
    def __call__(self, res):
        et = time.time()
        self.run_time = (et, res[0])
        self.res = res[1]

def _run(func, main_core, init_len_s = None, len_items = None, states=None, **kwargs):
    st = time.time()
    global sharedmemory
    if init_len_s is not None:
        dict = {}
        init_len = 0
        for i, len_items in enumerate(len_items):
            try:
                key, item = cPickle.loads(sharedmemory[init_len_s + init_len:init_len_s + len_items])
            except:
                key, item = dill.loads(sharedmemory[init_len_s + init_len:init_len_s + len_items])
            dict[key] = item
            init_len = len_items
        func = cPickle.loads(func)
        res = func(**dict)
    else:
        res = list(map(lambda idx: func[idx](**states[idx], **kwargs), range(len(states))))
    time_cost = time.time() - st
    return (time_cost, res)

class MultiProcess(ModBase):

    def __init__(self):
        self.funcs = {}

    def _popSet(self, pop):
        core_count = multiprocessing.cpu_count()
        # pop.process_manager = self
        '''initialize the process pool and keep it in the class, to avoid the frequent intialization'''
        global sharedmemory
        sharedmemory = SArray('c', 1024 ** 3)
        self.process_pool = multiprocessing.Pool(processes=core_count, initializer=init_pool, initargs=(sharedmemory, random.random()))
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

        '''record the function, for better acceleration'''
        self.__register(func, multiprocessing.cpu_count())

        # '''translate the **kwargs to a dict for multiprocessing transform'''
        # kwargs_dict = dict(**kwargs)

        # divd_states = []
        # if mask is None:
        #     mask = list(range(0, len(states)))
        #     divd_states = states
        # else:
        #     for item in mask:
        #         divd_states.append(states[item])

        progs, callbacks = [], []
        run_times_record = []
        if isinstance(func, list):
            core_count = statistics.mean([self.funcs[f].count for f in func])
        else:
            core_count = self.funcs[func].count
        avg_count = int(len(states) / core_count)

        st = time.time()
        if not isinstance(func, list):
            func = [func] * len(states)
        elif len(func) != len(states):
            raise ValueError("The size of func list '%d' should equal to mask len '%d'" % (len(func), len(states)))

        # for i, item in enumerate(states):
        #     dict_for_p = {**item, **kwargs}
        #     dict_list = []
        #     dict_for_p_ = {}
        #     for key, item in dict_for_p.items():
        #         dict_for_p_[key] = cPickle.dumps(item)
        # print(time.time() - st, '---------------0--')
        global sharedmemory
        len_items = []
        dicts_list = []
        count_fp = len(states) - avg_count
        # assert len(states) == len(progs), '{0}, {1}'.format(len(states), len(progs))
        for i, item in enumerate(states):
            if len(states) - i <= avg_count:
                continue
            run_times_record.append(time.time())
            func_for_p = cPickle.dumps(func[i])
            dict_for_p = {**item, **kwargs}
            dict_list = []
            len_items.append([])
            for key, item in dict_for_p.items():
                # dict_for_p[key] = cPickle.dumps(item)
                # print(key, item)
                try:
                    dict_list.append(cPickle.dumps((key,item)))
                except:
                    dict_list.append(dill.dumps((key,item)))
                if len(len_items[i]) == 0:
                    len_items[i].append(len(dict_list[-1]))
                else:
                    len_items[i].append(len(dict_list[-1]) + len_items[i][-1])
            dicts_list.append(dict_list)

        # print(avg_count, len(states), count_fp)
        init_len_s = 0
        for i, item in enumerate(states):
            callbacks.append(CallBack())
            if len(states) - i <= avg_count:
                continue
            init_len = 0
            for idx, lens in enumerate(len_items[i]):
                sharedmemory[init_len_s + init_len:init_len_s + lens] = dicts_list[i][idx]
                init_len = lens
            # p = multiprocessing.Process(target=_run, args=(func_for_p, False), kwargs=dict_for_p)
            progs.append(
                self.process_pool.apply_async(_run, args=(func_for_p, False, init_len_s, len_items[i]),
                                              callback=callbacks[-1])
                # p
            )
            init_len_s += len_items[i][-1]
            # p.start()

        # print(time.time() - st, '---------------2--')
        '''main process also need to work, a qualified capitalist must squeeze the surplus labor as much as possible ^o^ ^o^ '''
        states_maincore = []
        funcs_maincore = []
        for i in range(avg_count):
            funcs_maincore.append(func[len(states) - avg_count + i])
            states_maincore.append(states[len(states) - avg_count + i])
        res_f = _run(funcs_maincore, True, states=states_maincore, **kwargs)

        res = []
        transfer_cost, run_cost = 0, 0
        prog_size = len(progs)
        for i in range(prog_size):
            progs[i].wait()
            run_cost += callbacks[i].run_time[1]
            transfer_cost += callbacks[i].run_time[0] - run_times_record[i] - callbacks[i].run_time[1]
            res.append(callbacks[i].res)
        res.extend(res_f[1])
        et = time.time() - st
        print('run_cost, transfer_cost, core_count', run_cost / core_count, transfer_cost / core_count, core_count)
        
        return res
