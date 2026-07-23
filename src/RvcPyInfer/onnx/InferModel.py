from abc import ABC, abstractmethod
from collections import OrderedDict

from numpy.typing import NDArray


class InferModel(ABC):
    @abstractmethod
    def infer(self, output_names: list[str], inputs: dict[str, NDArray]) -> OrderedDict[str, NDArray]:
        pass