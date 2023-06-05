# CLI Commands

yapx imposes this simple structure for CLI apps:

```sh
___________ _________________ _________ _____________

awesome-cli --log-level debug say-hello --name Donald
___________ _________________ _________ _____________
  CLI App       Root-args      Command   Command-args
```

The above CLI app can be implemented like so:

```python
>>> import yapx
...
>>> def setup(log_level = 'info'):
...     print(f"Log level: {log_level}")
...
>>> def say_hello(name):
...     print(f"Hello {name}!")
...
>>> def say_goodbye(name):
...     print(f"Goodbye {name}!")
...
>>> yapx.run(setup, say_hello, say_goodbye, _print_help=True)

*******************************************************
usage: __main__.py [-h]
               [--log-level LOG_LEVEL]
               {say-hello,say-goodbye} ...

positional arguments:
  {say-hello,say-goodbye}
    say-hello
    say-goodbye

...
*******************************************************
>>> say-hello
usage: __main__.py say-hello [-h] --name NAME
...
*******************************************************
>>> say-goodbye
usage: __main__.py say-goodbye [-h] --name NAME
...
*******************************************************

>>> yapx.run(
... setup,
... say_hello,
... say_goodbye,
... _args=['say-hello', '--name', 'Donald']
)
Log level: info
Hello Donald!

>>> yapx.run(
... setup,
... say_hello,
... say_goodbye,
... _args=['say-goodbye', '--name', 'Donald']
)
Log level: info
Goodbye Donald!
```

The first function passed to `yapx.run(...)` is the *root-command*. Each subsequent command is a *sub-command*, e.g.:

```python
>>> yapx.run(root_cmd, subcmd_1, subcmd_2, ...)

usage: __main__.py [-h] {subcmd-1,subcmd-2} ...
```

The root-command is *always executed*, and the appropriate sub-command is executed based on the given arguments.

The function name is automatically coerced into a command-name. You can specify an alternate name by passing the functions as keyword-args to `yapx.run`:

```python
>>> yapx.run(
...     root_cmd,
...     say_hello=subcmd_1,
...     say_goodbye=subcmd_2,
... )
### or, equivalently:
>>> yapx.run(root_cmd, **{
...     'say-hello': subcmd_1,
...     'say-goodbye': subcmd_2,
... })

usage: __main__.py [-h] {say-hello,say-goodbye} ...
```

You can omit the root-command by passing `None` as the 1st argument, or by using `yapx.run_commands(...)` instead, e.g.:

```python
>>> yapx.run(None, say_hello, say_goodbye)
### or, equivalently:
>>> yapx.run_commands(say_hello, say_goodbye)

usage: __main__.py [-h] {say-hello,say_goodbye} ...
```

## Relay Value

The return-value of the root-command can be passed to a sub-command using the magic variable `_relay_value`. Here's an example:

```python title="_relay_value Example" hl_lines="8-9 11-12"
>>> import yapx
...
>>> def setup(name, upper=False):
...     if upper:
...         name = name.upper()
...     return name
...
>>> def say_hello(_relay_value):
...     print(f"Hello {_relay_value}!")
...
>>> def say_goodbye(_relay_value):
...     print(f"Goodbye {_relay_value}!")
...
>>> yapx.run(
    setup,
    say_hello,
    say_goodbye,
    _args=["--name", "Donald", "--upper", "say-hello"]
)
Hello DONALD!
```

## Yield to Setup, Teardown

Use `yield` within the root-command, to run teardown logic *after* any sub-commands are executed. This can be used in conjunction with `_relay_value` to offer a *setup-and-teardown* workflow for things like database connections.

```python title="Setup, Teardown Example using yield" hl_lines="4-9"
>>> import yapx
...
>>> def setup(db, table):
...     # setup
...     conn = sql.connect()
...     # yield
...     yield conn, sql.sanitize(db), sql.sanitize(table)
...     # teardown
...     conn.close()
...
>>> def add_rows(_relay_value):
...     conn, db, table = _relay_value
...     conn.query(
...         f'INSERT INTO {table} ...',
...         use_db=db
...     )
...
>>> def remove_rows(_relay_value):
...     conn, db, table = _relay_value
...     conn.query(
...         f'DELETE FROM {table} ...',
...         use_db=db
...     )
...
>>> yapx.run(setup, add_rows, remove_rows, _args=[
...     "--db", "AdventureWorks",
...     "--table", "customers",
...     "add-rows",
... ])
```
