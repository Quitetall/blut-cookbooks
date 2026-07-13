#!/usr/bin/env python3
"""Compatibility entry point for :mod:`blut_standard.trainer_dpo`."""

import sys

from blut_standard.trainer_dpo import *  # noqa: F403
from blut_standard.trainer_dpo import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
