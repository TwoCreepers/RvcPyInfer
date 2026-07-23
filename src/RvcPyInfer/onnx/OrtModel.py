from collections import OrderedDict
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..provider.OrtProviders import OrtProviders
from .InferModel import InferModel


class OrtModel(InferModel):
    def __init__(self, model: Path, providers: OrtProviders) -> None:
        model = model.resolve()
        
        import onnxruntime as ort
        self.session = ort.InferenceSession(
            str(model), providers=providers.to_providers()
        )

    def infer(self, output_names: list[str], inputs: dict[str, NDArray]) -> OrderedDict[str, NDArray]:
        infed = self.session.run(
            output_names=output_names, input_feed=inputs
        )
        result = OrderedDict[str, NDArray]()
        for n, i in zip(output_names, infed):
            assert isinstance(i, np.ndarray), "ORT 输出应为 ndarray"
            result[n] = i
        return result