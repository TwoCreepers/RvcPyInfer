from typing import Tuple, BinaryIO, Literal
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

type PathLike = str | Path
type FileLike = PathLike | BinaryIO
type Audio = Tuple[NDArray[np.float32], int]
type AudioLike = FileLike | Audio

F0ExtractAlgorithmList = ["dio", "harvest"]
type F0ExtractAlgorithm = Literal["dio", "harvest"]