from pathlib import Path

from .type_alist import PathLike


def path(path: PathLike) -> Path:
    if isinstance(path, str):
        return Path(path)
    else:
        return path