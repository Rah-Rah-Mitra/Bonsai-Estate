"""Entry point for the estate pipeline. Run through estate.cmd (Blender's python.exe -I -B estate.py <command>)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from estate.env import bootstrap  # noqa: E402

bootstrap()

if __name__ == "__main__":
    from estate.cli import main

    sys.exit(main())
