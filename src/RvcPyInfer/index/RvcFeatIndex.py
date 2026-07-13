import numpy as np
from numpy.typing import NDArray

from ..path_utils import path
from ..type_alist import PathLike


class RvcFeatIndex:
    def __init__(self, index: PathLike) -> None:
        import faiss  # pyright: ignore[reportMissingImports]
        self.faiss_index = faiss.read_index(str(path(index).resolve()))
        self.faiss_data: NDArray = self.faiss_index.reconstruct_n(0, self.faiss_index.ntotal)

    def apply_index(self, feats: NDArray[np.float32], index_rate: float = 0.66, k: int = 8) -> NDArray[np.float32]:
        score, ix = self.faiss_index.search(feats, k=k)
        weight = np.square(1 / score)
        weight /= weight.sum(axis=1, keepdims=True)
        index_feats = np.sum(self.faiss_data[ix] * np.expand_dims(weight, axis=2), axis=1)
        return index_feats * index_rate + (1 - index_rate) * feats