#!/usr/bin/env python3
"""Archival shim for the retired numbered-experiment batch launcher."""

from __future__ import annotations

import sys


MIGRATION_MESSAGE = (
    "This archival batch launcher is retired and performs no experiments. "
    "Use python run.py formal --help and select an explicit formal command."
)


def main(argv: list[str] | None = None) -> int:
    del argv
    print(MIGRATION_MESSAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
