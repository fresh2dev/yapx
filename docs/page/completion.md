# Shell Completion

When the [shtab](https://github.com/iterative/shtab) library is present, yapx CLIs gain the ability to export shell-completion scripts using the flag `--print-shell-completion`.

Install `shtab` using: `pip install yapx[shtab]`

For example:

```sh title="Shell-completion Example"
awesome-app --print-shell-completion bash
```

The above will just print the completion script, but the output needs to be written to a file in order to provide completions.

To do this for the `bash` shell:

```sh title="Install shell-completions for bash"
awesome-app --print-shell-completion bash | \
    sudo tee "$BASH_COMPLETION_COMPAT_DIR/awesome-app"
```

To do this for the `zsh` shell:

```sh title="Install shell-completions for zsh"
awesome-app --print-shell-completion zsh | \
    sudo tee /usr/local/share/zsh/site-functions/_awesome-app
```

I use the `zsh` shell and have this function in my profile:

```sh
install-yapx-completion() {
  $1 --print-shell-completion zsh | \
    sudo tee /usr/local/share/zsh/site-functions/_$1
}
```

This allows me to install shell-completions for yapx CLI apps by calling:

```sh
install-yapx-completion <app-name>
```
