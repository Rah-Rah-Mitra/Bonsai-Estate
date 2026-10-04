#!/usr/bin/env sh
# Git Bash wrapper for the estate pipeline (same as estate.cmd).
exec "/c/Program Files/Blender Foundation/Blender 5.2/5.2/python/bin/python.exe" -I -B "$(dirname "$0")/estate.py" "$@"
