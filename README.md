# yapx

> Yet another argparse extension for building CLI tools.

| Links         |                        |
|---------------|------------------------|
| Code Repo     | https://www.github.com/fresh2dev/yapx           |
| Mirror Repo   | https://www.Fresh2.dev/code/r/yapx        |
| Documentation | https://www.Fresh2.dev/code/r/yapx/i           |
| Changelog     | https://www.Fresh2.dev/code/r/yapx/i/changelog |
| License       | https://www.Fresh2.dev/code/r/yapx/i/license   |
| Funding       | https://www.Fresh2.dev/funding        |

[![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/fresh2dev/yapx?color=blue&style=for-the-badge)](https://www.github.com/fresh2dev/yapx/releases)
[![GitHub Release Date](https://img.shields.io/github/release-date/fresh2dev/yapx?color=blue&style=for-the-badge)](https://www.github.com/fresh2dev/yapx/releases)
[![License](https://img.shields.io/github/license/fresh2dev/yapx?color=blue&style=for-the-badge)](https://www.Fresh2.dev/code/r/yapx/i/license)
[![GitHub issues](https://img.shields.io/github/issues-raw/fresh2dev/yapx?color=blue&style=for-the-badge)](https://www.github.com/fresh2dev/yapx/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr-raw/fresh2dev/yapx?color=blue&style=for-the-badge)](https://www.github.com/fresh2dev/yapx/pulls)
[![GitHub Repo stars](https://img.shields.io/github/stars/fresh2dev/yapx?color=blue&style=for-the-badge)](https://star-history.com/#fresh2dev/yapx&Date)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/yapx?color=blue&style=for-the-badge)](https://pypi.org/project/yapx)
[![Docs Website](https://img.shields.io/website?down_message=unavailable&label=docs&style=for-the-badge&up_color=blue&up_message=available&url=https://www.Fresh2.dev/code/r/yapx/i)](https://www.Fresh2.dev/code/r/yapx/i)
[![Coverage Website](https://img.shields.io/website?down_message=unavailable&label=coverage&style=for-the-badge&up_color=blue&up_message=available&url=https://www.Fresh2.dev/code/r/yapx/i/tests/coverage)](https://www.Fresh2.dev/code/r/yapx/i/tests/coverage)
[![Funding](https://img.shields.io/badge/funding-%24%24%24-blue?style=for-the-badge)](https://www.Fresh2.dev/funding)

*Brought to you by...*

<a href="https://www.fresh2.dev"><img src="https://img.fresh2.dev/fresh2dev.svg" style="filter: invert(50%);"></img></a>

---

`yapx` is *yet another argparse extension* that helps you build Python CLI applications with ease.

`yapx` makes use of type annotations build a capable command-line utility using Python's native `argparse` library. yapx inherits from -- and can serve as a drop-in replacement to -- the `argparse.ArgumentParser` to provide benefits including:

- adding arguments derived from existing functions or dataclases.
- support use of environment variables as argument values.
- performing argument validation, with or without [Pydantic](...).
- generation of shell-completion scripts with [ shtab](...).
- CLI to TUI support with [Trogon](...).

> Note: `yapx` is in *beta* status. Please report ideas and issues [here](...).

## Install

```sh
pip install yapx
```

By default, `yapx` has no 3rd-party dependencies, but they can be added to unlock additional functionality. For example:

When [Pydantic](TODO) is present, `yapx` can support more types.

```sh
pip install 'yapx[pydantic]'
```

When [shtab](TODO) is present, `yapx` can output shell-completion scripts.

```sh
pip install 'yapx[shtab]'
```

When [Trogon](TODO) is present, `yapx` can present a terminal user interface (TUI) of the application.

```sh
pip install 'yapx[tui]'
```

Install all of these with:

```sh
pip install 'yapx[pydantic,shtab,tui]'
# or, equivalently:
pip install 'yapx[extras]'
```

## Use

Here's a simple example using typical argparse semantics:

```python
>>> import yapx
...
### Define some function.
>>> def add_nums(x: int, y: int):
...     return x + y
...
### Create a parser
>>> parser = yapx.ArgumentParser()
### Add function arguments to parser.
>>> parser.add_arguments(add_nums)
### Parse arguments.
>>> parsed = parser.parse_args(['-x', '1', '-y', '2'])
...
### Call function with parsed args.
>>> add_nums(parsed.x, parsed.y)
3
```

This capability -- and much more --  is simplified even further with the `yapx.run` command:

```python
>>> import yapx
...
>>> def add_nums(x: int, y: int):
...     return x + y
...
>>> yapx.run(add_nums, _args=['-x', '1', '-y', '2'])
3
```

Read more @ https://www.Fresh2.dev/code/r/yapx/i
