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
[![Funding](https://img.shields.io/badge/funding-%24%24%24-blue?style=for-the-badge)](https://www.f2dv.com/fund)

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
- `yapx[extras]`: enables each of the above
- `trogon-yapx`: enables experimental TUI support

## Use

See notebooks of examples here:

https://www.f2dv.com/code/r/yapx/i/page/overview

Read more about yapx @ https://www.f2dv.com/code/r/yapx/i

## Support

If this project delivers value to you, please [provide feedback](https://www.github.com/fresh2dev/yapx/issues), code contributions, and/or [funding](https://www.f2dv.com/fund).

See all of my projects @ https://www.f2dv.com/code/r

*Brought to you by...*

<a href="https://www.f2dv.com"><img src="https://img.fresh2.dev/fresh2dev.svg" style="filter: invert(50%);"></img></a>
