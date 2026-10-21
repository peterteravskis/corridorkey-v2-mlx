"""Compatibility entry point; prefer the installed corridorkey-mlx command."""
from corridorkey_mlx.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
