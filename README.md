# yapx

```python
import yapx

@dataclass
class ArgsModel:
    ...

parser = yapx.ArgumentParser()

parser.add_arguments(ArgsModel)

parser.print_help()
```

```python
@dataclass
class CmdArgsModel:
    ...

parser.add_command('run-command', CmdArgsModel)

parser.print_help()
```

```python
import yapx

def setup(...):
    ...

def run_it(...):
    ...


yapx.run(setup, run_it)
# or
yapx.run(setup, run_command=run_it)
# or
yapx.run(setup, **{
    'run-command': run_it
})
```

```python
yapx.run(setup, run_it, _print_help=True)
```

```python
yapx.run(lambda value: value * 5)
```
