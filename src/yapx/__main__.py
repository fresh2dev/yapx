import argparse
import sys

from .__version__ import __version__


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="_command")

    subparser_version = subparsers.add_parser("version")
    subparser_version.set_defaults(func=print, args=[__version__], kwargs={})

    parsed_args: argparse.Namespace = parser.parse_args(sys.argv[1:])

    parsed_args.func(*parsed_args.args, **parsed_args.kwargs)


if __name__ == "__main__":
    main()
