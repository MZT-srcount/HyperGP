# HyperGP.PrimitiveSet

```python
class PrimitiveSet
```

**********************

## 初始化

```python
def __init__(input_arity, primitive_set, prefix='x')
```

#### 参数
- input_arity: int
- primitive_set: 
- prefix: str

#### 示例


**********************

## 注册函数

```python
def registerPrimitive(name, func, arity, states, **kwargs)
```

#### 参数

#### 示例


```python
registerEphemeralTerminal(name, func, ephemeral=True, **kwargs)
```

#### 参数

#### 示例

```python
registerTerminal(name, **kwargs)
```

#### 参数

#### 示例


**********************

## 状态获取函数

```python
@property
def primitiveSet()
```
#### 示例

```python
@property
def terminalSet()
```
#### 示例

```python
def genFunc(name)
```
#### 参数

#### 示例

```python
def genTerminal(name)
```

#### 参数

#### 示例

```python
def copy()
```


**************************************
### 辅助函数

```python
def select()
```

```python
def selectFunc()
```

```python
def selectTerminal()
```