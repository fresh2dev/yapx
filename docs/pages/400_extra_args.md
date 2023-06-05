# Extra Arguments

By default, when an unrecognized argument is given, an error is raised.

```python title="Unknown Args" linenums="0" hl_lines="3 6-7"
>>> import yapx
...
>>> def say_hello():
...     print('hello world')
...
>>> yapx.run(say_hello, _args=['--foo', 'bar'])
error: unrecognized arguments: --foo bar
```

However, if the function accepts `*args` or `**kwargs`, then any unrecognized arguments are accepted and passed along.

Important things to note about the use of `*args` and `**kwargs`:

1. values are always passed as *strings*, regardless of type annotation.
2. The CLI help text will not indicate that extra arguments are accepted; this should be mentioned function docstring.

```python title="Extra Args" linenums="0" hl_lines="3 6-7"
>>> import yapx
...
>>> def say_hello(*args):
...     print('args:', args)
...
>>> yapx.run(say_hello, _args=['--foo', 'bar'])
args: ('--foo', 'bar')
```

```python title="Extra Keyword-Args" linenums="0" hl_lines="3 6-7"
>>> import yapx
...
>>> def say_hello(**kwargs):
...     print('kwargs:', kwargs)
...
>>> yapx.run(say_hello, _args=['--foo', 'bar'])
kwargs: {'--foo': 'bar'}
```

```python title="Extra Args and Keyword-Args" linenums="0" hl_lines="3 7-9"
>>> import yapx
...
>>> def say_hello(*args, **kwargs):
...     print('args:', args)
...     print('kwargs:', kwargs)
...
>>> yapx.run(say_hello, _args=['--foo', 'bar', 'baz'])
args: ('baz',)
kwargs: {'--foo': 'bar'}
```
