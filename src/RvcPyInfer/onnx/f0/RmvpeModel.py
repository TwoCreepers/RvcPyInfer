import numpy as np
from numpy.typing import NDArray

from ...audio.audio_utils import reSR
from ...path_utils import path as pathf
from ...provider.InferProviders import InferProviders
from ...type_alist import Audio, PathLike
from ..model_utils import build_model


class RmvpeModel:
    def __init__(self, model: PathLike, providers: InferProviders) -> None:
        self.model = build_model(pathf(model), providers)

    def infer(self, audio: Audio, p_len: int, threshold: float = 0.03) -> NDArray[np.float32]:
        data, sr = audio
        assert len(data.shape) == 1, "输入音频应为单通道"
        data, sr = reSR(audio, target_sr=16000)
        data = np.expand_dims(data, axis=0)
        input = {
            "waveform": data,
            "threshold": np.array(threshold, dtype=np.float32)
        }
        f0 = self.model.infer(["f0"], inputs=input)["f0"]
        # (1, frames)
        f0 = f0.squeeze(axis=0)
        f0 = f0.astype(np.float32)
        pad_len = p_len - len(f0)
        if pad_len == 0:
            return f0
        elif pad_len < 0:
            return f0[:pad_len]
        else:
            return np.pad(f0, [0, pad_len], mode="constant", constant_values=f0[-1])