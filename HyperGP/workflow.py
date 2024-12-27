import multiprocessing.process
from .base.base_struct import BaseStruct
from HyperGP.mods import AvailableMods, __Mods
import types
import inspect
import itertools
import random
from HyperGP.base.base_struct import States
from HyperGP.library.states import WorkflowStates
from HyperGP.src import device as set_device, query_device
import multiprocessing
import copy
from HyperGP.src import check_gpu

class GpOptimizer(BaseStruct, __Mods):
    """
    To use ``GpOptimizer`` module, we should first import it:
    
    Examples:
        >>> from HyperGP import GpOptimizer
    """
    available_mods = AvailableMods()
    
    def __init__(self, states=None, module_states=None, parallel=True, gpu=True, cash=False, **kwargs):
        """
        Initialize the optimizer
        
        Args:
            states(HyperGP.States):
            module_states(HyperGP.States):
            parallel(boolean): 
            gpu(boolean):
            kwargs:
        
        Returns:
            a new ``GPOptimizer`` module
        
        Examples:
            >>> optimizer = GpOptimizer()
        """
        self.gpu=gpu
        self.cash = cash
        self.monitors = []
        self.components = {} #for easier to deepcopy
        self.status = {**kwargs, **{
            'gpu': self.gpu,
            'cash': self.cash,
            'parallel': parallel
        }}
        self.workflowstates = WorkflowStates()
    
        self.proc, self.queue = None, None
        super().__init__(states, module_states, **kwargs)
        if parallel:
            self.enable('parallel')
    
    def status_init(self, **kwargs):
        """
        Register the needed states in the evolution. 
        
        The states will be stored in the ``workflowstates`` attribute, then we can get it with `str-like` object when we use ``iter_component`` method, \
        or just get it with attribute operator.
        
        Examples:
            
            >>> pop_size = 1000
            >>> pset = HyperGP.PrimitiveSet(input_arity=1,  primitive_set=[('add', HyperGP.tensor.add, 2),('sub', HyperGP.tensor.sub, 2)])
            >>> pop = HyperGP.Population(parallel=False)
            >>> pop.initPop(pop_size=pop_size, prog_paras=ProgBuildStates(pset=pset, depth_rg=[2, 3], len_limit=10000))
            >>> pop.stateRegister(cprogs = pop.states['progs'].copy)
            >>> input = np.random.uniform(0, 10, size=(1, 10000))
            >>> optimizer.status_init(
            ...     p_list=pop.states['cprogs'].indivs,
                    fit_list = pop.states['cprogs'].fitness,
                    input=Tensor(input),
                    pset=pset,
                    output=None
            >>> )
            >>> print(optimizer.input)
            xxxxxxxxxxxxxxxx

        """
        for key, value in kwargs.items():
            self.workflowstates[key] = value

    def __getattr__(self, item):
        # assert item in self.workflowstates
        # return self.workflowstates[item]
        if item not in self.__dict__:
            assert item in self.workflowstates, "{ITEM} not in workflowstates and __dict__".format(ITEM=item)
            return self.workflowstates[item]
        return self.__dict__[item]

    def _mask_check(self, masks):
        mask_lens = set([len(mask) if isinstance(mask, list) else 1 for mask in masks])
        assert len(mask_lens) <= 2, "The length of each element in mask should keep same"
        return max(mask_lens)

    def _mask_index(self, l, mask):
        return [l[m] for m in mask]

    def monitor(self, tool, track_object, save_path):
        """
        Register the components want to be iteratively executed.

        Args:
            tool: The monitor to be called
            track_object (str-like or list): States to be monitored
            save_path: File path to save the results
        
        Examples:
            xxxxxxxxxxx
            )
        """

        self.monitors.append([tool, track_object, save_path])

    def iter_component(self, *args):
        """
        Register the components want to be iteratively executed.

        Args:
            args: The ``HyperGP.states.ParaStates`` module is needed to register each component.
        
        Examples:
            >>> optimizer.iter_component(
            >>>     ParaStates(func=HyperGP.RandTrCrv(), source=["p_list", "p_list"], to=["p_list", "p_list"],
            ...            mask=set_prmask(pop_size)),
            >>>     ParaStates(func=shuffle, source=["p_list"], to=["p_list"],
            ...             mask=[1]),
            >>>     ParaStates(func=HyperGP.RandTrMut(), source=["p_list", ProgBuildStates(pset=pset, depth_rg=[2, 3], len_limit=100), True], to=["p_list"],
            ...             mask=[set_armask(pop_size), 1, 1]),
            >>>     ParaStates(func=HyperGP.executor, source=["p_list", "input", "pset"], to=["output", None],
            ...             mask=[1, 1, 1]),
            >>>     ParaStates(func=HyperGP.evaluation, source=["output", "output_tensor"], to=["fit_list"],
            ...             mask=[1, 1])
            )
        """
        self.components['from_list'] = []
        self.components['to_list'] = []
        self.components['mask_list'] = []
        self.components['pdefine_list'] = []
        self.components['param_list'] = []
        self.components['unit_size_list'] = []
        self.components['func_list'] = []
        self.components['mfunc_list'] = []

        for func in args:
            max_len = self._mask_check(func["mask"])
            self.components['unit_size_list'].append(max_len)
            signature = inspect.signature(func["func"])
            params = signature.parameters
            self.components['from_list'].append(func["source"])
            self.components['func_list'].append(func["func"])
            self.components['to_list'].append(func["to"])
            self.components['mask_list'].append(func["mask"])
            self.components['pdefine_list'].append(func["parallel"])
            self.components['param_list'].append(list(params.keys()))
            self.components['mfunc_list'].append(func["func_mask"])
    
    def enable(self, mod, **kwargs):
        if getattr(self, mod):
            self.__setattr__(mod, self.available_mods.__getattribute__(mod)())
            self.__getattribute__(mod)._popSet(self, **kwargs)

    def _run_independent(self, iter, device=[0]):
        
        for i in range(iter):
            print('iteration: %d'%i)
            for j, (func, from_l, to_l, mask_l) in enumerate(zip(self.components['func_list'], self.components['from_list'], self.components['to_list'], self.components['mask_list'])):
                for k in range(len(from_l)):
                    if isinstance(from_l[k], str):
                        from_l[k] = self.workflowstates[from_l[k]]
                states = [States(**{self.components['param_list'][j][k]:source[mask_l[k][z]] for k, source in enumerate(from_l) if isinstance(mask_l[k], list)}) for z in range(self.components['unit_size_list'][j])]
                states_kwargs = {self.components['param_list'][j][k]:from_l[k] for k, mask in enumerate(mask_l) if isinstance(mask, int)}
                if len(states) == 0:
                    states = [States(**states_kwargs)]
                    states_kwargs = {}
                funcs = [func] * self.components['unit_size_list'][j] if not isinstance(func, list) else [self._mask_index(func, mask) for mask in self.components['mfunc_list'][j]]
                rets = self.__parallel(funcs, states, device, self.components['pdefine_list'][j], kwargs=states_kwargs)
                if len(to_l) == 0:
                    continue
                for k, key in enumerate(to_l):
                    if key == None or isinstance(key, str):
                        continue
                    key.clear()
                if len(to_l) > 1:
                    for k, key in enumerate(to_l):
                        if key == None:
                            continue
                        result = []
                        for res in rets:
                            if isinstance(res[k], list):
                                result.extend(res[k])
                            else:
                                result.append(res[k])

                        if isinstance(key, str):
                            if len(result) > 1:
                                self.workflowstates[key] = result
                            else:
                                self.workflowstates[key] = result[0]
                        else:
                            key.extend(result)
                else:
                    result = []
                    for res in rets:
                        if isinstance(res, list):
                            result.extend(res)
                        else:
                            result.append(res)
                    if isinstance(to_l[0], str):
                        self.workflowstates[to_l[0]] = result
                    else:
                        to_l[0].extend(result)
            
            for monitor in self.monitors:
                track_object = monitor[1]
                if isinstance(monitor[1], str):
                    track_object = self.workflowstates[monitor[1]]
                    monitor[0](track_object, save_path=monitor[2])
                if isinstance(monitor[1], list):
                    track_object = [self.workflowstates[key] for key in monitor[1]]
                    monitor[0](*track_object, save_path=monitor[2])


    def __run_parallel(self, iter, device=[0]):
        
        context = multiprocessing.get_context("spawn")
        manager_queue = context.Manager().Queue()
        proc = context.Process(target=_state_transform, args=(iter, manager_queue, self._package, self.status, device))
        self.proc = proc
        self.queue = manager_queue

    def run(self, iter, device=[0], async_parallel=False):
        """
        Run the optimizer with iteration time and device
        
        Args:
            iter(int): The iteration time
            device(list): GPU index list used in the optimizer
            async_parallel(boolean): Whether asynchronously execute the optimizer. If it is True, the method will immediately return, then a ``wait`` method is needed to wait the evolution process finish.
        
        Examples:

            >>> optimizer.run(10)

            or run it asynchronously:
            >>> optimizer.run(10, async_parallel=True)
            >>> optimizer.wait()

        """
        if async_parallel:
            self.__run_parallel(iter, device)
        else:
            self._run_independent(iter, device)

    def detach(self):
        """
        Copy the optimizer. It can be used to avoid repeatly register the same states or components.

        Returns:
            A new ``GpOptimizer`` with independent same states.

        Examples:
            >>> optimizers = []
            >>> for i in range(5):
            ...    optimizers.append(optimizer.detach())

            >>> for optimizer in optimizers:
            ...    optimizer.run(100, async_parallel=True)
            >>> for optimizer in optimizers:
            ...    optimizer.wait()
        """
        print(self.status)
        new_workflow = GpOptimizer(**self.status)
        new_workflow._update(copy.deepcopy(self._package, {}))
        return new_workflow

    def wait(self):
        """
        Wait the optimizer finish when an asynchronous run is called.
        """
        if self.proc is not None:
            self.proc.join()
            self._update(self.queue.get())

    @property
    def _package(self):
        return (self.states, self.module_states, self.workflowstates, self.components, self.monitors)

    def _update(self, package):
        self.states = package[0]
        self.module_states = package[1]
        self.workflowstates = package[2]
        self.components = package[3]
        self.monitors = package[4]

    def __parallel(self, method, states, gparallel, parallel=True, kwargs={}):
        if isinstance(method, list) and len(method) != len(states):
            raise ValueError('The method size %d not equal to the cond size %d' % (len(method), len(states)))
        default_devid = query_device()
        if 'parallel' in self.gmodule_states and parallel:
            ret_cond = self.parallel(method, states, **kwargs)
        else:
            ret_cond = []
            if isinstance(states, States):
                ret_cond.append(method(**states, **kwargs))
            elif isinstance(states, list):
                assert len(states) == len(method)
                for i, state in enumerate(states):
                    set_device(gparallel[random.randint(0, len(gparallel) - 1)])
                    ret_cond.append(method(**state, **kwargs)
                                if not isinstance(method, list)
                                else method[i](**state, **kwargs))
            else:
                assert 0==1
        set_device(default_devid)
        return ret_cond
                


def _state_transform(iter, ret_queue, package, status, device=[0]):
    new_optimizer = GpOptimizer(**status)
    new_optimizer._update(package)
    new_optimizer._run_independent(iter, device)
    ret_queue.put(new_optimizer._package)