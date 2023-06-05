# Related Projects

## [argparse](https://docs.python.org/3/library/argparse.html)

yapx simply adds some capabilities to Python's `argparse`. `yapx.ArgumentParser` inherits from `argparse.ArgumentParser` and can serve as a drop-in replacement.

## [argparse-dataclass](https://github.com/mivade/argparse_dataclass)

So much of the inspiration for this project -- and a good bit of the initial code -- came from this `argparse_dataclass` module. This project would not exist without it :pray:

## [typer](https://github.com/tiangolo/typer)

typer and yapx serve a very similar purpose of using type annotations to build CLI applications in Python. Here are some differences:

- yapx is built on `argparse`; typer is built on `click`.
- typer is more mature -- better documented! -- and is built by the author of FastAPI, so you know it's good.
- ...

## [click](https://github.com/pallets/click)

A trusted tool for building CLIs.


## [invoke](https://github.com/pyinvoke/invoke)

Another trusted Python library for the command-line, with an emphasis on task execution, à la `make`.

## [myke](https://github.com/fresh2dev/myke)

`myke` is a superset of the yapx project with an emphasis on task execution, à la `make`. yapx is the library and myke is an instantiation of it.

Read more about myke @ https://www.Fresh2.dev/code/r/myke/i
