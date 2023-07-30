# yapx

> The next generation of Python's Argparse.

| Links         |                                       |
|---------------|---------------------------------------|
| Code Repo     | https://www.github.com/fresh2dev/yapx |
| Documentation | https://www.f2dv.com/r/yapx           |
| Changelog     | https://www.f2dv.com/r/yapx/changelog |
| License       | https://www.f2dv.com/r/yapx/license   |
| Funding       | https://www.f2dv.com/fund             |

[![GitHub Repo stars](https://img.shields.io/github/stars/fresh2dev/yapx?color=blue&style=for-the-badge)](https://star-history.com/#fresh2dev/yapx&Date)
[![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/fresh2dev/yapx?color=blue&style=for-the-badge)](https://www.f2dv.com/r/yapx/changelog)
[![GitHub Release Date](https://img.shields.io/github/release-date/fresh2dev/yapx?color=blue&style=for-the-badge)](https://www.f2dv.com/r/yapx/changelog)
[![License](https://img.shields.io/github/license/fresh2dev/yapx?color=blue&style=for-the-badge)](https://www.f2dv.com/r/yapx/license)
## Overview

Yapx is "yeah, another argparse extension" for building Python CLIs with the utmost simplicity.

It works by reading type hints of Python functions and dataclasses, and uses them to construct an argparse `ArgumentParser`.

Yapx features:

- Support for subcommands
- Support for lists in various forms: (positional) `1 2 3`, (multiple) `-x 1 -x 2 -x 3`, (multi-value) `-x 1 2 3`, (comma-separated), `-x 1, 2, 3`, (string) `-x '[1, 2, 3]'`
- Support for dictionaries / key-value mappings such as `--values one=1 two=2 three=3`
- Support for optional booleans: `--flag` / `--no-flag`
- Support for feature flags: `--dev` / `--test` / `--prod`
- Support for counting parameters: `-vvv`
- Automatic "helpful" arguments: `--help`, `--help-all`, `--version`, etc.
- Support for setting values from environment variables

## Install

Yapx has no 3rd-party dependencies out-of-the-box:

```sh
pip install yapx
```

Extras are available to unlock additional functionality:

- `yapx[pydantic]`: enables support for additional types
- `yapx[shtab]`: enables shell-completion
- `yapx[rich]`: enables prettier help and error messages
- `yapx[tui]`: enables experimental TUI support
- `yapx[extras]`: enables each of the above

## Use

See notebooks of examples here:

https://www.f2dv.com/r/yapx/page/overview

Read more about yapx @ https://www.f2dv.com/r/yapx

## Support

If this project delivers value to you, please [provide feedback](https://www.github.com/fresh2dev/yapx/issues), code contributions, and/or [funding](https://www.f2dv.com/fund).

See all of my projects @ https://www.f2dv.com/projects

*Brought to you by...*

<a href="https://www.f2dv.com"><img src="https://img.fresh2.dev/fresh2dev.svg" style="filter: invert(50%);"></img></a>
