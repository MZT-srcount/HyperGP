# HyperGP.executor

```python
def executor(progs, input, pset, cash=None)
```
设计用于程序个体的批量执行


#### 参数

- **progs**: [HyperGP.Individual]  
输入需要执行的多个个体
- **input**: HyperGP.Tensor  
输入需要拟合的数据
- **pset**: HyperGP.PrimitiveSet  
输入符号集
- **cash**: CashManager  
是否采用代际缓存技术：保存当代个体的部分节点的计算结果从而减少下代的执行计算量

#### 示例
```python
from HyperGP import executor, PrimitiveSet, Population

# Generae the primitive set
pset = PrimitiveSet(
    input_arity=1,
    primitive_set=[
        ('add', HyperGP.add, 2),
        ('sub', HyperGP.sub, 2),
        ('mul', HyperGP.mul, 2),
        ('div', HyperGP.div, 2),
])

# Generate the input data
input_array = np.random.uniform(0, 10, size=(1, 10000))

# Initialize the population
pop = Population()
pstates = ProgBuildStates(pset=pset, depth_rg=[2, 6], len_limit=100)
pop.initPop(pop_size=pop_size, prog_paras=pstates)

# Execute the individuals
output, _ = executor(pop.states['progs'].indivs, input_array, pset)



```