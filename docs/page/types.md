# Type Annotations

yapx makes direct use of Python type-annotations to offer type-validation and other conveniences.

yapx will infer argument types when necessary:

```python hl_lines="3 19-22 27-31" linenums="0"
>>> import yapx
...
>>> def demo(a, b = 1, c = 3.14, d = False):
...     print(
...         f"a={a} {type(a)}",
...         f"b={b} {type(b)}",
...         f"c={c} {type(c)}",
...         f"d={d} {type(d)}",
...         sep="\n",
...     )
...
>>> yapx.run(
...     demo,
...     _print_help=True,
... )
usage: __main__.py [-h] -a A [-b B] [-c C] [-d]
...
Arguments:
  -a A                  > Type: str, Required
  -b B                  > Type: int, Default: 1
  -c C                  > Type: float, Default: 3.14
  -d                    > Type: bool, Default: False

>>> yapx.run(demo, _args=[
...     '-a', 'hello', '-b', '6', '-c', '2.718', '-d'
... ])
a=hello <class 'str'>
b=6 <class 'int'>
c=2.718 <class 'float'>
d=True <class 'bool'>
```

The use of type-annotations (aka "type-hints") allows `yapx` to accept and validate more complex arguments types such as `list`, `set`, and even `dict`.

```python hl_lines="6-13 31-42 54-61" linenums="0"
>>> import yapx
>>> from pathlib import Path
>>> from typing import Dict, List, Sequence, Set
...
>>> def demo(
...     a: str,
...     b: int,
...     c: float,
...     d: bool,
...     a_seq: Sequence[str],
...     b_list: List[int],
...     c_set: Set[float],
...     d_dict: Dict[str, bool],
... ):
...     print(
...         f"a={a} {type(a)}",
...         f"b={b} {type(b)}",
...         f"c={c} {type(c)}",
...         f"d={d} {type(d)}",
...         f"a_seq={a_seq} {type(a_seq)}",
...         f"b_list={b_list} {type(b_list)}",
...         f"c_set={c_set} {type(c_set)}",
...         f"d_dict={d_dict} {type(d_dict)}",
...         sep="\n",
...     )
...
>>> yapx.run(demo, _print_help=True)
usage: __main__.py [-h] -a A [-b B] [-c C] [-d]
...
Arguments:
  -a A                  > Type: str, Required
  -b B                  > Type: int, Required
  -c C                  > Type: float, Required
  -d                    > Type: bool, Default: None
  --a-seq A_SEQ [A_SEQ ...]
                        > Type: Sequence[str], Required
  --b-list B_LIST [B_LIST ...]
                        > Type: List[int], Required
  --c-set C_SET [C_SET ...]
                        > Type: Set[float], Required
  --d-dict D_DICT [D_DICT ...]
                        > Type: Dict[str, bool], Required

>>> yapx.run(demo, _args=[
...     '-a', 'hello',
...     '-b', '6',
...     '-c', '2.718',
...     '-d',
...     '--a-seq', 'hello', 'goodbye',
...     '--b-list', '1', '2', '3',
...     '--c-set', '3.14', '3.14', '2.718',
...     '--d-dict', 'hello=1', 'goodbye=0', 'stop=n', 'go=y',
... ])
a=hello <class 'str'>
b=6 <class 'int'>
c=2.718 <class 'float'>
d=True <class 'bool'>
a_seq=['hello', 'goodbye'] <class 'list'>
b_list=[1, 2, 3] <class 'list'>
c_set={2.718, 3.14} <class 'set'>
d_dict={'hello': True, 'goodbye': False, 'stop': False, 'go': True} <class 'dict'>
```

### Type Containers

Complex argument types (`List`, `Set`, `Dict`, etc.) can be provided with commas: `--arg 1, 2, 3`, or without them: `--arg 1 2 3`

`yapx` treats the types `Sequence`, `List`, `Tuple` essentially the same way. As such, `yapx` has limited support for `Tuple` types, and requires that they contain a single type and be unbounded in length, e.g., `Tuple[int ...]`.

Be careful of using complex types in an app that also has subcommands or positional arguments. Take this example:

```sh
awesome-app --list one two three four some-command
```

The argument parser believes that `some-command` is just another item in the list.

### Dict Args

As shown in the above example, when an argument is defined as a `Dict`, the argument values are expected to be a list of key-value pairs, separated by an equal sign, e.g.:

`--arg foo=bar this=that` is parsed to `{'foo': 'bar', 'this': 'that'}`

If is a key is present with no value, it assumes the value `None`, e.g.:

`--arg foo bar` is parsed to `{'foo': None, 'bar': None}`

This is actually a clever way to obtain an *ordered* set, since the `Dict` will inherently deduplicate keys *and* retain order.

### Choices

There are two ways to limit acceptable values for a given parameter.

Using the `Literal` annotation:

```python title="Choices using Literal" linenums="1" hl_lines="2 5 16-17 20"
>>> import yapx
>>> from yapx.types import Literal
...
>>> def demo(
...     value: Literal['one', 'two', 'three'],
... ):
...     print(value)
...
>>> yapx.run(demo, _print_help=True)
usage: __main__.py [-h] --value {one,two,three}

optional arguments:
  -h, --help            show this help message and exit

Arguments:
  --value {one,two,three}
                        > Type: Literal['one', 'two', 'three'], Required

>>> yapx.run(demo, _args=['--value', 'five'])
ArgumentError: argument --value: invalid choice: 'five' (choose from 'one', 'two', 'three')

>>> yapx.run(demo, _args=['--value', 'two'])
two
```

Or, using an `Enum` annotation:

```python title="Choices using Enum" linenums="1" hl_lines="2 5-8 11 23-24 27 30"
>>> import yapx
>>> from enum import Enum, auto
>>> from typing import List
...
>>> class ValidChoices(Enum):
...     one = auto()
...     two = auto()
...     three = auto()
...
>>> def demo(
...     value: List[ValidChoices]
... ):
...     print(value)
...
>>> yapx.run(demo, _print_help=True)
usage: __main__.py [-h] --value
               {one,two,three} [{one,two,three} ...]

optional arguments:
  -h, --help            show this help message and exit

Arguments:
  --value {one,two,three} [{one,two,three} ...]
                        > Type: ValidChoices], Required

>>> yapx.run(demo, _args=['--value', 'five'])
ArgumentError: argument --value: invalid choice: 'five' (choose from 'one', 'two', 'three')

>>> yapx.run(demo, _args=['--value', 'one', 'two'])
[<ValidChoices.one: 1>, <ValidChoices.two: 2>]
```

> Note: the use of `from __future__ import annotations` can cause issues with yapx parsing type-hints when using non-native types like custom `Enum`.

Comparing the two, `Literal` is quicker to implement, but you cannot have a list of them; i.e., `List[Literal[...]]` is not supported, but a list of enum `List[ValidChoices]` works well. Enums also offer more precise comparisons.

### Pydantic

yapx performs type-casting and validation of some basic, built-in Python types with no dependencies outside of the standary library. But if the `pydantic` library is present, yapx will rely on it to support even more types. Install it using: `pip install 'yapx[pydantic]'`

```python
>>> import yapx
>>> from typing import Pattern
...
>>> def is_match(text: str, pattern: Pattern) -> bool:
...     return bool(pattern.fullmatch(text))
...
>>> yapx.run(is_match, _args=['--text', '123', '--pattern', '\\d+'])
# with pydantic:
True
# without pydantic:
UnsupportedTypeError: Unsupported type: typing.Pattern
"pip install 'yapx[pydantic]'" to support more types.
```
