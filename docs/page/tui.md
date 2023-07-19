# CLI to TUI

Yapx offers experimental support for displaying a Textual User Interface (TUI) of your CLI.

It does this using my fork of the Trogon library. To use this experimental feature:

```
pip install --pre trogon-yapx
```

Yapx will detect the module and add a `--tui` flag to your CLI.

## Example

To create a functional app with a CLI *and* TUI, write this script to a file:

```python title="example-tui.py" linenums="1" hl_lines="0"
#!/usr/bin/env python3

import yapx
from yapx.types import Literal


def setup(log_level: Literal["error", "info", "debug"] = "info"):
    print(f"Log level: {log_level}")


def _say_greeting(greeting, name, uppercase=False):
    msg = f"{greeting} {name}!"
    if uppercase:
        msg = msg.upper()
    print(msg)


def say_hello(name, uppercase=False):
    _say_greeting(greeting="Hello", name=name, uppercase=uppercase)


def say_goodbye(name, uppercase=False):
    _say_greeting(greeting="Goodbye", name=name, uppercase=uppercase)


yapx.run(setup, [say_hello, say_goodbye])
```

Then, make the script executable:

```sh
chmod +x ./example-tui.py
```

Now invoke the script:

```sh
./example-tui.py --tui
```

If you want the TUI to open when no arguments are provided:

```python
yapx.run(
    setup,
    [say_hello, say_goodbye],
    default_args=["--tui"],
)
```

Now you can view the TUI without giving any args:

```sh
./example-tui.py
```

By default, the TUI is invoked with the parameter `--tui`. To use a command instead of a parameter:

```python
yapx.run(
    setup,
    [say_hello, say_goodbye],
    tui_flags="tui",
    default_args=["tui"],
)
```

An example TUI could look like:
<video width="800" height="600" controls>
  <source src="https://user-images.githubusercontent.com/554369/239734211-c9e5dabb-5624-45cb-8612-f6ecfde70362.mov" type="video/mp4">
</video>

*gif source: https://github.com/Textualize/trogon/README.md*
