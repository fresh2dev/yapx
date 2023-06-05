# CLI to TUI

When the [trogon](https://github.com/Textualize/trogon) library is present, yapx will allow your app to be presented as a terminal user interface (TUI) simply by using the command-line flag `--tui`.

Install it using: `pip install yapx[tui]`

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


yapx.run(
    setup,
    say_hello,
    say_goodbye,
    _tui_flags=["--tui"],
)
```

Then, make the script executable:

```sh
chmod +x ./example-tui.py
```

Finally, invoke the script:

```sh
./example-tui.py --tui
```

If you want the TUI to open by default when no arguments are provided, set `_tui_flags=[None]`, e.g.:

```python
yapx.run(
    setup,
    say_hello,
    say_goodbye,
    _tui_flags=[None],
)
```

Now you can open the TUI by simply invoking the script:

```sh
./example-tui.py
```

An example TUI could look like:
<video width="800" height="600" controls>
  <source src="https://user-images.githubusercontent.com/554369/239734211-c9e5dabb-5624-45cb-8612-f6ecfde70362.mov" type="video/mp4">
</video>

*gif source: https://github.com/Textualize/trogon/README.md*
