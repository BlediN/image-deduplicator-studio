#!/usr/bin/env python3
"""Launcher kept so `python photo_review_app.py` still works. Same as `python -m photo_review`."""

from photo_review.cli import main

if __name__ == "__main__":
    main()
