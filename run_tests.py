#!/usr/bin/env python3
"""Run the test suite with stdlib unittest (no pytest needed inside a ComfyUI environment).

    python run_tests.py [-v]

The ComfyUI-backed tests need the Python that runs ComfyUI and COMFYUI_PATH pointing at its checkout; without them
they are skipped and only the pure tests run.
"""

import sys
import unittest


def main():
    suite = unittest.TestLoader().discover(start_dir="tests", top_level_dir=".")
    result = unittest.TextTestRunner(verbosity=2 if "-v" in sys.argv else 1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
