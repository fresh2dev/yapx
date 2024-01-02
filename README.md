<h1 align="center">Yapx</h1>
<p align="center"><em>The next generation of Python's Argparse.</em></p>
<h2 align="center">
<a href="https://www.f2dv.com/r/yapx/" target="_blank">Documentation</a>
| <a href="https://www.f2dv.com/s/yapx/" target="_blank">Slide Deck</a>
| <a href="https://www.github.com/fresh2dev/yapx/" target="_blank">Git Repo</a>
</h2>

*Yapx* is "Yeah, Another Argparse eXtension", a Python library for creating tools with type-safe command-line interfaces (CLIs) -- and even textual user interfaces (TUIs) -- by analyzing type-hints of functions and dataclasses.

> Yeah, Another?

I intended to publish this package as simply `apx`, but PyPi didn't allow it, so I tacked on the *y* to make \*`yapx`\*. The nomenclature "*Yet Another*" seems demeaning considering what this package is capable of.


So, *<b>Yeah</b>, Another Argparse eXtension* :punch:


Think about the repetitive steps involved in creating a command-line application.

1. Define function(s).
2. Define the command-line interface; i.e., build the `ArgumentParser`.
3. Parse arguments.
4. Call the appropriate function(s).

For example:

```python
from argparse import ArgumentParser

# 1. Define function(s).
def say_hello(name: str = 'World'):
    print(f"Hello {name}")

# 2. Define the command-line interface.
parser = ArgumentParser()
parser.add_argument("--name", default="World")

# 3. Parse arguments.
parsed_args = parser.parse_args()

# 4. Call the appropriate function(s).
say_hello(name=parsed_args.name)
```

Yapx combines these steps into one:

```python
import yapx

def say_hello(name: str = 'World'):
    print(f"Hello {name}")

yapx.run(say_hello)
```

Yapx is a superset of Python's native Argparse `ArgumentParser`, so you can make use of the high-level abstractions or do the low-level work you're familiar with. Either way, Yapx provides benefits including:

- :lock: Type-casting and validation, with or without [*Pydantic*](https://docs.pydantic.dev).
- :question: Automatic addition of "*helpful*" arguments, including `--help`, `--help-all`, `--version`, and most impressively `--tui`.
- :heart: Prettier help and error messages.
- :zap: Command-line autocompletion scripts.

Yapx is among several modern Python CLI frameworks, including [*Typer*](https://github.com/tiangolo/typer) and [*Fire*](https://github.com/google/python-fire). Distinguishing characteristics of Yapx include:

- :package: No 3rd-party dependencies required (but can be opted into)
- :lock: Type-safe argument validation
- :spider_web: Infinitely-nested commands
- :tv: Display your CLI as a TUI
- :question: Handling of unknown arguments using `*args` and `**kwargs`
- :bulb: Most intuitive
- :sweat: Least documented

I'd appreciate a more analytical comparison between Yapx and the array of Python CLI frameworks, but that's too ambitious of a goal right now :grimacing:


## Install

```
pip install 'yapx[extras]'
```

Or, to install without 3rd-party dependencies:

```
pip install yapx
```

## Use

<a href="https://www.f2dv.com/s/yapx/" target="_blank">
    <img src="https://img.fresh2.dev/slides_placeholder.png"></img>
</a>

---

[![License](https://img.shields.io/github/license/fresh2dev/yapx?color=blue&style=for-the-badge)](https://www.f2dv.com/r/yapx/license/)
[![GitHub tag (with filter)](https://img.shields.io/github/v/tag/fresh2dev/yapx?filter=!*%5Ba-z%5D*&style=for-the-badge&label=Release&color=blue)](https://www.f2dv.com/r/yapx/changelog/)
[![GitHub last commit (branch)](https://img.shields.io/github/last-commit/fresh2dev/yapx/main?style=for-the-badge&label=updated&color=blue)](https://www.f2dv.com/r/yapx/changelog/)
[![GitHub Repo stars](https://img.shields.io/github/stars/fresh2dev/yapx?color=blue&style=for-the-badge)](https://star-history.com/#fresh2dev/yapx&Date)
<!-- [![Funding](https://img.shields.io/badge/funding-%24%24%24-blue?style=for-the-badge)](https://www.f2dv.com/fund/) -->
<!-- [![GitHub issues](https://img.shields.io/github/issues-raw/fresh2dev/yapx?color=blue&style=for-the-badge)](https://www.github.com/fresh2dev/yapx/issues/) -->
<!-- [![GitHub pull requests](https://img.shields.io/github/issues-pr-raw/fresh2dev/yapx?color=blue&style=for-the-badge)](https://www.github.com/fresh2dev/yapx/pulls/) -->
<!-- [![PyPI - Downloads](https://img.shields.io/pypi/dm/yapx?color=blue&style=for-the-badge)](https://pypi.org/project/yapx/) -->
<!-- [![Docker Pulls](https://img.shields.io/docker/pulls/fresh2dev/yapx?color=blue&style=for-the-badge)](https://hub.docker.com/r/fresh2dev/yapx/) -->
