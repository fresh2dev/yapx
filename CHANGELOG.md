# Changelog

## [0.2.2] - 2023-07-28

### :clap: Features

- Override add_subparsers defaults [54b88a5]

## [0.2.1] - 2023-07-24

### :point_right: Changes

- Use argparse-tui fork of trogon [ce7917c]
- refactor: BooleanOptionalAction [4910820]
- refactor: Small optimizations [e32e101]

## 0.2.0 - 2023-07-18

- better error handling
- optional integration with rich, rich-argparse for prettier output
- support for automatic bool-negation parameters, e.g.: `--flag / --no-flag`
- support for "counting" arguments: `value: Annotated[int, yapx.arg(nargs=0)]`
- support for "feature" arguments: `value: Annotated[str, yapx.arg(nargs=0)]`
- Sequence-types (list, set, dict, etc.) use nargs=1 by default.

Previously, this was the way to populate a list: `--value 1 2 3`

Now, this is the way: `--value 1 --value 2 --value 3`

The prior behavior can still be obtained using: `value: Annotated[List[int], yapx.arg(nargs='*')]`

### :clap: Features

- Appended sequences, counting args, feature args, and bool arg negation. [09f315e]
- Improved error handling, prettier output with rich [29d85b3]
- Add `--tui` flag to help [abb5780]
- Add tui as argument or subcommand [839ff27]
- Add `patch_run` [4f07cd0]
- Expose yapx.build_parser, yapx.Namespace, yapx.is_... [79f8d03]
- Command name is optional; is now derived from func/model name [2f00be7]
- Attempt to get prog version [13edbbb]
- Support Pydantic v2 [7b25bc9]

### :fist: Fixes

- Casting of Dict value types [4743696]
- Raise ValueError when string not in Enum names [8c2dee5]
- Don't include shtab parameter in parsed args [b8cd492]
- Correct value of counting parameters with no default [a324f35]
- Only add `--version` flag if prog_version is not empty [15c34c1]
- Convert default values of boolean arguments [543c58f]
- Make all helpful flags optional [16a74dc]

### :point_right: Changes

- *Breaking:* `build_parser` and rework `run` [bbc7505]
- Inherit appropriate argparse actions [12e3bce]
- Remove tui extra for now [750a6a2]

SLOC Analysis:

| Directory   |   Empty |   Comment |   Docstring |   Code |   Statements |
|-------------|---------|-----------|-------------|--------|--------------|
| src/yapx    |     366 |        45 |         322 |   1604 |          909 |
| tests       |     332 |       186 |         241 |   1383 |          699 |


## 0.1.0 - 2023-06-04

- Initial release :rocket:

SLOC Analysis:

| Directory   |   Empty |   Comment |   Docstring |   Code |   Statements |
|-------------|---------|-----------|-------------|--------|--------------|
| src/yapx    |     338 |        46 |         363 |   1490 |          806 |
| tests       |     337 |       173 |         215 |   1331 |          725 |
