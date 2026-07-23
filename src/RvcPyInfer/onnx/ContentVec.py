import numpy as np
from numpy.typing import NDArray

from ..audio.audio_utils import reSR
from ..path_utils import path as pathf
from ..provider.InferProviders import InferProviders
from ..type_alist import Audio, PathLike
from .model_utils import build_model


class ContentVec:
    def __init__(self, path: PathLike, providers: InferProviders) -> None:
        self.model = build_model(pathf(path), providers)

    def infer(self, audio: Audio) -> NDArray[np.float32]:
        data, sr = audio
        assert len(data.shape) == 1, "输入音频应为单通道"
        data, sr = reSR(audio, target_sr=16000)
        data = np.expand_dims(data, axis=(0, 1))
        input = {
            "source": data
        }
        logits = self.model.infer(["embed"], input)["embed"]
        # (1, frames, embed_size)
        return logits.squeeze(axis=0).astype(np.float32)