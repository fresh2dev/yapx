# Changelog

## 0.5.1 - 2023-12-10

### :clap: Features

- Allow stdin input for any type

### :point_right: Changes

- *Breaking:* When casting from `str`, return `None` if blank

SLOC Analysis:

| Directory   |   Empty |   Comment |   Docstring |   Code |   Statements |
|-------------|---------|-----------|-------------|--------|--------------|
| src/yapx    |     466 |        60 |         596 |   2008 |         1120 |
| tests       |     392 |       199 |         275 |   1636 |          820 |

## 0.5.0 - 2023-12-07

### :clap: Features

- `yapx.arg(stdin=...)` to get arg value from stdin stream.

### :fist: Fixes

- *Breaking:* Do not append to default sequence value (addresses https://bugs.python.org/issue16399)
- Show default values of more types in `--help`

### :point_right: Changes

- *Breaking:* no longer remove commas occurring between sequence parameter values

SLOC Analysis:

| Directory   |   Empty |   Comment |   Docstring |   Code |   Statements |
|-------------|---------|-----------|-------------|--------|--------------|
| src/yapx    |     461 |        60 |         591 |   1989 |         1105 |
| tests       |     385 |       199 |         264 |   1592 |          806 |

## 0.4.2 - 2023-10-30

### :clap: Features

- `build_parser_from_spec/file` [9e51ae8]

SLOC Analysis:

| Directory   |   Empty |   Comment |   Docstring |   Code |   Statements |
|-------------|---------|-----------|-------------|--------|--------------|
| src/yapx    |     455 |        57 |         510 |   1987 |         1100 |
| tests       |     382 |       199 |         264 |   1580 |          799 |

## 0.4.1 - 2023-08-29

### :clap: Features

- Suppress `--help-all` flag with `add_help_all=False` [048ad79]

## 0.4.0 - 2023-08-26

### :clap: Features

- *Breaking:* Support infinitely nested subcommands [b8ad0c4]

### :fist: Fixes

- Don't add field to dataclass when annotated with `Optional[yapx.Context]` [00db7bd]
- Allow arbitrary types when using pydantic v2 [3105bac]

### :point_right: Changes

- Simplify help messages and metavars [66d81d4]

SLOC Analysis:

| Directory   |   Empty |   Comment |   Docstring |   Code |   Statements |
|-------------|---------|-----------|-------------|--------|--------------|
| src/yapx    |     434 |        57 |         504 |   1860 |         1030 |
| tests       |     364 |       188 |         252 |   1524 |          753 |

## 0.3.0 - 2023-08-17

### :clap: Features

- Add vanity arg methods; counting_arg, feature_arg, unbounded_arg [135c34c]
- *Breaking:* `yapx.Context` used instead of `_relay_value` [a93341a]

### :fist: Fixes

- Refactor to ensure `parse_args` is called only once [a401a4f]
- Dict annotation returning empty list when empty arg given [47dcd49]
- Restore ability to suppress tui option [223ba03]

### :point_right: Changes

- *Breaking:* Positional args to `yapx.arg(...)` are now considered flags. [8a5540f]
- *Breaking:* `*args` and `**kwargs` are populated by unknown args [1338dcd]
- Only show defaults in help text if not null or empty [bc60880]

SLOC Analysis:

| Directory   |   Empty |   Comment |   Docstring |   Code |   Statements |
|-------------|---------|-----------|-------------|--------|--------------|
| src/yapx    |     404 |        51 |         509 |   1804 |          984 |
| tests       |     358 |       184 |         251 |   1460 |          738 |

---


## 0.2.4 - 2023-07-29

### :clap: Features

- Add `--tui` parameter to every subparser [1b0eef8]

SLOC Analysis:

| Directory   |   Empty |   Comment |   Docstring |   Code |   Statements |
|-------------|---------|-----------|-------------|--------|--------------|
| src/yapx    |     375 |        45 |         341 |   1654 |          935 |
| tests       |     332 |       186 |         241 |   1385 |          700 |

---

## 0.2.3 - 2023-07-28

### :fist: Fixes

- Fix parsing of positional arguments with sequence types [231aaee]

---

## 0.2.2 - 2023-07-28

### :clap: Features

- Override add_subparsers defaults [54b88a5]

---

## 0.2.1 - 2023-07-24

### :point_right: Changes

- Use argparse-tui fork of trogon [ce7917c]
- refactor: BooleanOptionalAction [4910820]
- refactor: Small optimizations [e32e101]

---

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

---

## 0.1.0 - 2023-06-04

- Initial release :rocket:

SLOC Analysis:

| Directory   |   Empty |   Comment |   Docstring |   Code |   Statements |
|-------------|---------|-----------|-------------|--------|--------------|
| src/yapx    |     338 |        46 |         363 |   1490 |          806 |
| tests       |     337 |       173 |         215 |   1331 |          725 |

<!-- generated by git-cliff -->
