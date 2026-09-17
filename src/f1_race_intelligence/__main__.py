"""``python -m f1_race_intelligence``: the batch entrypoint. See :mod:`f1_race_intelligence.cli`."""

import sys

from f1_race_intelligence.cli import main

if __name__ == "__main__":
    sys.exit(main())
