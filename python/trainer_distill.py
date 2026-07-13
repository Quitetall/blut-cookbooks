#!/usr/bin/env python3
"""Compatibility entry point for :mod:`blut_standard.trainer_distill`."""

import sys

from blut_standard.trainer_distill import *  # noqa: F403
from blut_standard.trainer_distill import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
