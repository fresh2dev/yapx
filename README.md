# yapx

> Build awesome Python CLIs with ease.

| Links         |                                              |
|---------------|----------------------------------------------|
| Code Repo     | https://www.github.com/fresh2dev/yapx        |
| Mirror Repo   | https://www.f2dv.com/code/r/yapx             |
| Documentation | https://www.f2dv.com/code/r/yapx/i           |
| Changelog     | https://www.f2dv.com/code/r/yapx/i/changelog |
| License       | https://www.f2dv.com/code/r/yapx/i/license   |
| Funding       | https://www.f2dv.com/funding                 |

[![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/fresh2dev/yapx?color=blue&style=for-the-badge)](https://www.github.com/fresh2dev/yapx/releases)
[![GitHub Release Date](https://img.shields.io/github/release-date/fresh2dev/yapx?color=blue&style=for-the-badge)](https://www.github.com/fresh2dev/yapx/releases)
[![License](https://img.shields.io/github/license/fresh2dev/yapx?color=blue&style=for-the-badge)](https://www.f2dv.com/code/r/yapx/i/license)
[![GitHub issues](https://img.shields.io/github/issues-raw/fresh2dev/yapx?color=blue&style=for-the-badge)](https://www.github.com/fresh2dev/yapx/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr-raw/fresh2dev/yapx?color=blue&style=for-the-badge)](https://www.github.com/fresh2dev/yapx/pulls)
[![GitHub Repo stars](https://img.shields.io/github/stars/fresh2dev/yapx?color=blue&style=for-the-badge)](https://star-history.com/#fresh2dev/yapx&Date)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/yapx?color=blue&style=for-the-badge)](https://pypi.org/project/yapx)
[![Docker Pulls](https://img.shields.io/docker/pulls/fresh2dev/yapx?color=blue&style=for-the-badge)](https://hub.docker.com/r/fresh2dev/yapx)
[![Docs Website](https://img.shields.io/website?down_message=unavailable&label=docs&style=for-the-badge&up_color=blue&up_message=available&url=https://www.f2dv.com/code/r/yapx/i)](https://www.f2dv.com/code/r/yapx/i)
[![Coverage Website](https://img.shields.io/website?down_message=unavailable&label=coverage&style=for-the-badge&up_color=blue&up_message=available&url=https://www.f2dv.com/code/r/yapx/i/tests/coverage)](https://www.f2dv.com/code/r/yapx/i/tests/coverage)
[![Funding](https://img.shields.io/badge/funding-%24%24%24-blue?style=for-the-badge)](https://www.f2dv.com/funding)

---

`yapx` is *yeah, another argparse extension* that helps you build awesome Python CLI applications with ease.

When building a Python CLI, you typically define your functions (with their arguments, types, and defaults), *then* separately define command-line arguments (along with types, defaults, etc.). The goal of yapx is to combine these steps into one. It does this by making use of type annotations (aka "type-hints") to parse your functions and build a CLI using Python's native `argparse` library.

yapx features:

- add arguments derived from existing functions or dataclases.
- use of environment variables as argument values.
- perform argument type-casting and validation.
- generation of shell-completion scripts with [shtab](https://github.com/iterative/shtab).
- CLI to TUI support with [Trogon](https://github.com/Textualize/trogon).

> Note: `yapx` is in *beta* status. Please report ideas and issues [here](https://github.com/fresh2dev/yapx/issues).

## Install

By default, `yapx` has no 3rd-party dependencies, but they can be added to unlock additional functionality.

```sh
pip install 'yapx[extras]'
```

What's in 'yapx[extras]'?

- `yapx[shtab]` --> support for export shell-completion scripts.
- `yapx[pydantic]` --> support for more types.
- `yapx[tui]` --> display the CLI as a TUI (terminal user interface).


## Use

Creating a CLI is as simple creating a Python file like so...

```python title="say-hello.py"
import yapx

def say_hello(name):
    print(f"Hello {name}.")

if __name__ == "__main__":
    yapx.run(say_hello)
```

...and invoking it from the command-line.

```sh
$ python say-hello.py --name World
Hello world.
```

While `yapx.run` provides the most featureful experience, yapx also supports familiar argparse semantics:

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

## Read More

Read more about yapx @ https://www.f2dv.com/code/r/yapx/i

See all of my projects @ https://www.f2dv.com/code/r

*Brought to you by...*

<a href="https://www.fresh2.dev"><img src="https://img.fresh2.dev/fresh2dev.svg" style="filter: invert(50%);"></img></a>
