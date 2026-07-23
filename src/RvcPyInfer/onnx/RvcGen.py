import numpy as np
from numpy.typing import NDArray

from ..path_utils import path as pathf
from ..provider.InferProviders import InferProviders
from ..type_alist import Audio, PathLike
from .model_utils import build_model
from .OrtModel import OrtModel
from .OvModel import OvModel


class RvcGen:
    def __init__(self, path: PathLike, model_out_sr: int, providers: InferProviders) -> None:
        self.sr = model_out_sr
        self.model = build_model(pathf(path), providers)
        if isinstance(self.model, OrtModel): # 这是一个抽象泄漏，但我不想管它，因为就这一处
            self.channels = next(filter(lambda arg: arg.name == "rnd", self.model.session.get_inputs())).shape[1]
        elif isinstance(self.model, OvModel):
            self.channels = next(filter(lambda arg: arg.any_name == "rnd", self.model.model.inputs)).partial_shape[1].get_length()

    def infer(self, phone: NDArray[np.float32], mel_pitch: NDArray[np.int64], f0_pitch: NDArray[np.float32], sid: int = 0, seed: int | None = 1234) -> Audio:
        assert phone.shape[0] == mel_pitch.shape[0] == f0_pitch.shape[0], "帧数应统一"
        phone_len = phone.shape[0]
        ds = np.array([sid], dtype=np.int64)
        rnd = np.random.default_rng(seed).standard_normal((1, self.channels, phone_len), dtype=np.float32)
        phone = np.expand_dims(phone, axis=0)
        mel_pitch = np.expand_dims(mel_pitch, axis=0)
        f0_pitch = np.expand_dims(f0_pitch, axis=0)

        model_input = {
            "rnd": rnd,
            "ds": ds,
            "phone": phone,
            "phone_lengths": np.array([phone_len], dtype=np.int64),
            "pitch": mel_pitch,
            "pitchf": f0_pitch
        }
        output = self.model.infer(["audio"], model_input)["audio"]
        # (1, 1, sampling)
        return output.squeeze(axis=1).squeeze(axis=0).astype(np.float32), self.sr