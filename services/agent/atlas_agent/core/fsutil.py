from __future__ import annotations

import os
import stat
from pathlib import Path

COUNT_CAP = 10_000
ALLOWED_TEXT_EXTENSIONS = frozenset({".txt", ".md", ".json", ".csv", ".sql", ".html", ".css"})
WRITE_LIMIT = 200 * 1024


def format_size(n: int) -> str:
    if n >= 1024 * 1024:
        mb = n / (1024 * 1024)
        return f"{mb:.1f} MB"
    if n >= 1024:
        kb = n / 1024
        return f"{kb:.1f} KB"
    return f"{n} B"


def count_tree(path: Path) -> tuple[int, int, bool]:
    items = 0
    size = 0
    stack = [path]
    while stack:
        current = stack.pop()
        items += 1
        if items > COUNT_CAP:
            return COUNT_CAP, size, True
        try:
            info = current.lstat()
        except OSError:
            continue
        if stat.S_ISLNK(info.st_mode):
            size += info.st_size
            continue
        if stat.S_ISREG(info.st_mode):
            size += info.st_size
            continue
        if stat.S_ISDIR(info.st_mode):
            try:
                with os.scandir(current) as entries:
                    for entry in entries:
                        stack.append(Path(entry.path))
            except OSError:
                continue
    return items, size, False


def entry_kind(path: Path) -> str:
    if path.is_symlink():
        return "file"
    if path.is_dir():
        return "dir"
    return "file"
