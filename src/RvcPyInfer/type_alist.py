from pathlib import Path
from typing import BinaryIO, Literal

import numpy as np
from numpy.typing import NDArray

type PathLike = str | Path
type FileLike = PathLike | BinaryIO
type Audio = tuple[NDArray[np.float32], int]
type AudioLike = FileLike | Audio

type F0ExtractAlgorithm = Literal["dio", "harvest", "rmvpe"]