#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Console/runtime helpers."""

from __future__ import annotations

import os
import sys


def ensure_utf8_console() -> None:
    """Best-effort UTF-8 console setup for Windows terminals."""
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

