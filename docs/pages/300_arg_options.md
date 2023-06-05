# Argument Options

Use `yapx.arg` to modify properties of an argument.

For reference, here is an example this does *not* use `yapx.arg`:

```python
>>> import yapx
...
>>> def demo(
...     x: int = 1,
...     y: int = 2,
...     z: int = 3,
... ):
...     print(
...         f"x={x} {type(x)}",
...         f"y={y} {type(y)}",
...         f"z={z} {type(z)}",
...         sep="\n",
...     )
...
>>> yapx.run(
...     demo,
...     _print_help=True,
... )
usage: __main__.py [-h] -x X -y Y -z Z

optional arguments:
  -h, --help            show this help message and exit

Arguments:
  -x X                  > Type: int, Default: 1
  -y Y                  > Type: int, Default: 2
  -z Z                  > Type: int, Default: 3

>>> yapx.run(demo)
x=1 <class 'int'>
y=2 <class 'int'>
z=3 <class 'int'>
```

And here is the same example, but using `yapx.arg` to modify each argument to accept an environment variable and provide help text.


```python
>>> import yapx
...
>>> def demo(
...     x: int = yapx.arg(default=1, env='X_VALUE', help='The value of X.'),
...     y: int = yapx.arg(default=2, env='Y_VALUE', help='The value of Y.'),
...     z: int = yapx.arg(default=3, env='Z_VALUE', help='The value of Z.'),
... ):
...     print(
...         f"x={x} {type(x)}",
...         f"y={y} {type(y)}",
...         f"z={z} {type(z)}",
...         sep="\n",
...     )
...
>>> yapx.run(
...     demo,
...     _print_help=True,
... )
usage: __main__.py [-h] -x X -y Y -z Z

optional arguments:
  -h, --help            show this help message and exit

Arguments:
  -x X                  The value of X. > Type: int, Default: 1, Env:
                        ['X_VALUE']
  -y Y                  The value of Y. > Type: int, Default: 2, Env:
                        ['Y_VALUE']
  -z Z                  The value of Z. > Type: int, Default: 3, Env:
                        ['Z_VALUE']

>>> import os
>>> os.environ['X_VALUE'] = '123'
>>> os.environ['Y_VALUE'] = '456'
>>> os.environ['Z_VALUE'] = '789'
>>> yapx.run(demo)
x=123 <class 'int'>
y=456 <class 'int'>
z=789 <class 'int'>
```

This method of specifying `yapx.arg` as a default value introduces a couple subtle issues, however:

1. if the function is called outside of yapx, the `yapx.arg` object will be returned as-is, unprocessed.
2. your IDE will complain about a type mismatch.

This code illustrates issue #1:

```python
>>> import yapx
...
>>> def demo(
...     x: int = yapx.arg(default=1),
...     y: int = yapx.arg(default=2),
...     z: int = yapx.arg(default=3),
... ):
...     print(
...         f"x={x} {type(x)}",
...         f"y={y} {type(y)}",
...         f"z={z} {type(z)}",
...         sep="\n",
...     )
...
### called with yapx
>>> yapx.run(demo, _args=['-x', '1', '-y', '2'])
x=1 <class 'int'>
y=2 <class 'int'>
z=3 <class 'int'>

### called directly
>>> demo(x=1, y=2)
x=1 <class 'int'>
y=2 <class 'int'>
z=Field(name=None,type=None,default=3, ...) <class 'dataclasses.Field'>
>>> demo(x=1
```

## Annotated

A better way is to use the `Annotated` type-hint. It solves this problem and allows arguments to be configured differently depending  on whether the function is called directly or from the command-line.

```python
>>> import yapx
>>> from yapx.types import Annotated
...
>>> def demo(
...     x: Annotated[int, yapx.arg(default=1, help='The value of X.')],
...     y: Annotated[int, yapx.arg(default=2, help='The value of Y.')],
...     z: Annotated[int, yapx.arg(default=3, help='The value of Z.')],
... ):
...     print(
...         f"x={x} {type(x)}",
...         f"y={y} {type(y)}",
...         f"z={z} {type(z)}",
...         sep="\n",
...     )

### called with yapx
>>> yapx.run(demo, _args=['-x', '1', '-y', '2'])
x=1 <class 'int'>
y=2 <class 'int'>
z=3 <class 'int'>

### called directly
>>> demo(x=1, y=2)
TypeError: demo() missing 1 required positional argument: 'z'
```

Using the `Annotated` syntax, an argument can assume a different default value depending on whether it was called directly:

```python
>>> import yapx
>>> from yapx.types import Annotated
...
>>> def demo(
...     x: Annotated[int, yapx.arg(default=1, help='The value of X.')] = 7,
...     y: Annotated[int, yapx.arg(default=2, help='The value of Y.')] = 8,
...     z: Annotated[int, yapx.arg(default=3, help='The value of Z.')] = 9,
...     called_from_cli: Annotated[bool, yapx.arg(default=True)] = False,
... ):
...     print(
...         f"x={x} {type(x)}",
...         f"y={y} {type(y)}",
...         f"z={z} {type(z)}",
...         f"Called from CLI: {called_from_cli}",
...         sep="\n",
...     )

### called with yapx
>>> yapx.run(demo)
x=1 <class 'int'>
y=2 <class 'int'>
z=3 <class 'int'>
Called from CLI: True

### called directly
>>> demo()
x=7 <class 'int'>
y=8 <class 'int'>
z=9 <class 'int'>
Called from CLI: False
```

It is useful to determine whether a function is invoked directly or from the command-line. To help make the distinction, yapx recognizes a "magic" variable `_called_from_cli` and sets it to `True` when present.


```python
>>> import yapx
...
>>> def demo(
...     _called_from_cli: bool = False,
... ) -> bool:
...     print(f"Called from CLI: {_called_from_cli}")

### called with yapx
>>> yapx.run(demo)
Called from CLI: True

### called directly
>>> demo()
Called from CLI: False
```
